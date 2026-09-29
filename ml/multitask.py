# -*- coding: utf-8 -*-
"""Multi-task joint training: mortality (PhysioNet 2012) + sepsis (Challenge 2019).

Shared AttentionLSTM trunk (same arch as the serving model: 2-layer LSTM-96
+ additive temporal attention), two linear heads. Each source keeps its own
labels and its own normalization stats - no label blending (see
ml/challenge2019_to_features.py docstring, option 2).

Eval is on the IDENTICAL E1 slices (val seed-42 stride-15, working holdout
seed-123 excl. locked), so mortality-head AUCs are directly comparable to
the E1 baselines. Outputs are CANDIDATE-only (challenge2019 is not
deploy_train-cleared). Locked slices are never re-run.

Usage:
  .\\.venv\\Scripts\\python.exe -m ml.multitask --smoke
  .\\.venv\\Scripts\\python.exe -m ml.multitask --seeds 61 62 --tag mt
"""
import argparse
import datetime
import gc
import itertools
import json
import os
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from ml.paths import physionet2012_root as _pn_root
from ml.dataset import SERVING_FEATURES, load_and_create_sequences
from ml.c19_pretrain import build_c19, patient_split_masks
from ml.improve_results import (
    FEATURES_12, EPOCHS, PATIENCE, MIN_DELTA,
    norm_fit, norm_apply, preds_of, eval_logits,
    build_all, split_holdout, acquire_lock, release_lock,
)
from ml.train_lstm import AttentionLSTMModel

BASE = _pn_root()
C19_SPLIT_SEED = 7
C19_VAL_FRAC = 0.10
HIDDEN = 96
LR = 1e-4


class MultiTaskAttentionLSTM(AttentionLSTMModel):
    """Shared trunk + two heads. forward(x) -> (mortality_logits, sepsis_logits)."""

    def __init__(self, input_size, hidden_size=96, num_layers=2, dropout=0.3):
        super().__init__(input_size, hidden_size, num_layers, dropout,
                         bidirectional=False)
        ctx = hidden_size
        self.fc_mort = nn_Linear(ctx)
        self.fc_sep = nn_Linear(ctx)
        # ignore the single-head fc from the parent
        del self.fc

    def _context(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)
        return self.batch_norm(context)

    def forward(self, x):
        c = self._context(x)
        return self.fc_mort(c).squeeze(-1), self.fc_sep(c).squeeze(-1)

    def mortality_logits(self, x):
        return self.fc_mort(self._context(x)).squeeze(-1)


def nn_Linear(ctx):
    return torch.nn.Linear(ctx, 1)


def build_e1_sets(smoke=False):
    (Xva, yva), (Xa, ya, pa), (Xb, yb, pb) = build_all(
        False, train_stride=30, smoke=smoke, window=90, horizon=12.0)
    _, _, val_pids = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'),
        outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12.0, gap_channels=False)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, v_idx = next(gss.split(np.zeros(len(val_pids)), np.zeros(len(val_pids)),
                              groups=np.array(val_pids)))
    val_set = set(np.array(val_pids)[v_idx])
    locked_set = set()
    import glob as _glob
    for lock_path in sorted(_glob.glob(os.path.join('ml', 'locked_holdout*.json'))):
        try:
            locked_set |= set(json.load(open(lock_path))['patients'])
        except Exception as e:
            print(f'WARNING: could not read {lock_path} ({e})', flush=True)
    if locked_set:
        print(f'locked slices honored: {len(locked_set)} set-b patients excluded', flush=True)
    pb_arr = np.array(pb)
    tr_mask_b = split_holdout(pb_arr) & np.array([p not in locked_set for p in pb_arr])
    ho_mask_b = ~split_holdout(pb_arr) & np.array([p not in locked_set for p in pb_arr])
    X_tr = np.concatenate([Xa[np.array([p not in val_set for p in pa])],
                           Xb[tr_mask_b]], axis=0)
    y_tr = np.concatenate([ya[np.array([p not in val_set for p in pa])],
                           yb[tr_mask_b]], axis=0)
    return (Xva, yva), (X_tr, y_tr), (Xb[ho_mask_b], yb[ho_mask_b])


def to_loader(X, y, batch):
    return DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                      torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))),
        batch_size=batch, shuffle=True,
        pin_memory=torch.cuda.is_available())


def train_seed(seed, arrays, run_base, smoke=False):
    (X_tr, y_tr, m_mean, m_std, pw_m), (X_c19, y_c19, c_mean, c_std, pw_c), \
        (Xva, yva), (Xfho, yfho) = arrays
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_trn = norm_apply(X_tr, m_mean, m_std)
    X_c19n = norm_apply(X_c19, c_mean, c_std)
    X_van = norm_apply(Xva, m_mean, m_std)
    X_fhon = norm_apply(Xfho, m_mean, m_std)
    print(f'  seed {seed} [multitask]: P12 {X_trn.shape} + C19 {X_c19n.shape}',
          flush=True)
    model = MultiTaskAttentionLSTM(input_size=12, hidden_size=HIDDEN,
                                   dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_m = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pw_m), dtype=torch.float32, device=device))
    loss_c = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pw_c), dtype=torch.float32, device=device))
    best_auc, best_state, patience = 0.0, None, 0
    max_ep = 2 if smoke else EPOCHS
    epoch = 0
    for epoch in range(max_ep):
        dl_m = to_loader(X_trn, y_tr, 128)
        dl_c = to_loader(X_c19n, y_c19, 128)
        model.train()
        total = 0.0
        n_steps = 0
        for (xb_m, yb_m), (xb_c, yb_c) in zip(itertools.cycle(dl_m), dl_c):
            xb_m = xb_m.to(device, non_blocking=True)
            yb_m = yb_m.to(device, non_blocking=True)
            xb_c = xb_c.to(device, non_blocking=True)
            yb_c = yb_c.to(device, non_blocking=True)
            lm, _ = model(xb_m)
            _, lc = model(xb_c)
            loss = loss_m(lm, yb_m) + loss_c(lc, yb_c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_steps += 1
        for pg in optimizer.param_groups:
            pg['lr'] = max(LR * (0.5 ** (epoch // 6)), 1e-6)
        mets = eval_logits(mortality_preds(model, X_van, device), yva)
        print(f'  epoch {epoch} loss={total / max(1, n_steps):.4f} '
              f'mort-val={mets["auc"]:.4f}', flush=True)
        if mets['auc'] - best_auc > MIN_DELTA:
            best_auc, patience = mets['auc'], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    p_va = mortality_preds(model, X_van, device)
    p_fho = mortality_preds(model, X_fhon, device)
    va, fh = eval_logits(p_va, yva), eval_logits(p_fho, yfho)
    out_dir = os.path.join(run_base, f'seed{seed}')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    np.save(os.path.join(out_dir, 'p_va.npy'), p_va)
    np.save(os.path.join(out_dir, 'p_fho.npy'), p_fho)
    m = {'config': f'multitask-seed{seed}', 'model_class': 'attention',
         'features': FEATURES_12, 'window': 90, 'horizon': 12.0,
         'dropout': 0.3, 'lr': LR, 'hidden': HIDDEN, 'causal': True,
         'train': 'E1 P12 set + full C19 setA (joint, per-source labels/scalers)',
         'fresh_holdout': 'E1 20pct set-b (seed 123, excl locked)',
         'best_auc': best_auc, 'val_auc': va['auc'],
         'val_accuracy': va['accuracy'], 'val_recall': va['recall'],
         'fresh_holdout_auc': fh['auc'],
         'fresh_holdout_accuracy': fh['accuracy'],
         'fresh_holdout_recall': fh['recall'],
         'epochs_trained': epoch + 1, 'seed': seed}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    with open(os.path.join(out_dir, 'scaler.json'), 'w') as f:
        json.dump({'features': FEATURES_12, 'mean': m_mean.tolist(),
                   'std': m_std.tolist(),
                   'c19_mean': c_mean.tolist(), 'c19_std': c_std.tolist()}, f, indent=2)
    print(f"  seed {seed}: val={va['auc']:.4f} | fresh-holdout={fh['auc']:.4f}", flush=True)
    return m


def mortality_preds(model, X, device):
    from ml.train_lstm import SimpleLSTMDataset
    model.eval()
    dl = DataLoader(SimpleLSTMDataset(X, np.zeros(len(X))), batch_size=512, shuffle=False)
    pr = np.zeros(len(X))
    off = 0
    with torch.no_grad():
        for xb, _ in dl:
            b = xb.shape[0]
            pr[off:off + b] = model.mortality_logits(xb.to(device)).cpu().numpy().ravel()
            off += b
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[61, 62])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    smoke = args.smoke
    tag = 'smoke' if smoke else (args.tag or datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    acquire_lock()
    try:
        mp = 200 if smoke else None
        Xc, yc, pc = build_c19(max_patients=mp)
        tr_m, _ = patient_split_masks(pc, C19_SPLIT_SEED, C19_VAL_FRAC)
        X_c19, y_c19 = Xc[tr_m], yc[tr_m]
        del Xc, yc, pc
        gc.collect()
        (Xva, yva), (X_tr, y_tr), (Xfho, yfho) = build_e1_sets(smoke=smoke)
        m_mean, m_std = norm_fit(X_tr)[0], norm_fit(X_tr)[1]
        c_mean, c_std = norm_fit(X_c19)[0], norm_fit(X_c19)[1]
        pw_m = int((y_tr == 0).sum()) / max(1, int((y_tr == 1).sum()))
        pw_c = int((y_c19 == 0).sum()) / max(1, int((y_c19 == 1).sum()))
        arrays = ((X_tr, y_tr, m_mean, m_std, pw_m),
                  (X_c19, y_c19, c_mean, c_std, pw_c),
                  (Xva, yva), (Xfho, yfho))
        run_base = os.path.join('ml', 'training_runs', f'improve_multitask_{tag}')
        os.makedirs(run_base, exist_ok=True)
        np.save(os.path.join(run_base, 'y_va.npy'), yva)
        np.save(os.path.join(run_base, 'y_fho.npy'), yfho)
        seeds = [61] if smoke else args.seeds
        results = []
        for seed in seeds:
            done_marker = os.path.join(run_base, f'seed{seed}', 'metrics.json')
            ckpt_marker = os.path.join(run_base, f'seed{seed}', 'model.pt')
            if os.path.exists(done_marker) and os.path.exists(ckpt_marker):
                m = json.load(open(done_marker))
                print(f"  seed {seed}: SKIPPED (done: val={m['val_auc']:.4f})", flush=True)
                results.append((seed, m))
                continue
            results.append((seed, train_seed(seed, arrays, run_base, smoke=smoke)))
        results.sort(key=lambda r: r[1]['val_auc'], reverse=True)
        print('\nRanking:', flush=True)
        for seed, m in results:
            print(f"  seed{seed}: val={m['val_auc']:.4f} "
                  f"fresh-holdout={m['fresh_holdout_auc']:.4f}", flush=True)
        print(f'Runs: {run_base}', flush=True)
        print(f'\nNext: .\\.venv\\Scripts\\python.exe -m ml.improve_results '
              f'--ensemble {run_base}', flush=True)
    finally:
        release_lock()


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        print('\nCAMPAIGN CRASHED:', flush=True)
        traceback.print_exc()
        try:
            release_lock()
        except Exception:
            pass
        raise SystemExit(1)
