"""Round 23 - combined set-a + set-b training (~7000 patients, 2x data).

Train: full set-a (minus original val patients) + 80% of set-b (patient split seed 123).
Val: original stride-15 split (comparable to 0.8371 baseline).
Holdout: fresh unseen 20% of set-b (honest external number).
Deploys if val > 0.8371 and fresh holdout >= 0.78.
"""
import datetime
import json
import os
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

from ml.dataset import SERVING_FEATURES
from ml.paths import physionet2012_root as _pn_root

BASE = _pn_root()
FEATURES_12 = SERVING_FEATURES
BASELINE_VAL = 0.8371
FRESH_HO_MIN = 0.78
EPOCHS = 40
PATIENCE = 14
MIN_DELTA = 0.0005
SEEDS = [91, 92, 93, 94]


def load_orig_val_pids():
    from ml.dataset import load_and_create_sequences
    _, _, val_pids = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, v_idx = next(gss.split(np.zeros(len(val_pids)), np.zeros(len(val_pids)), groups=val_pids))
    return set(np.array(val_pids)[v_idx])


def build_all():
    from ml.dataset import load_and_create_sequences
    print('loading original val (stride 15)...', flush=True)
    Xva, yva, _ = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    X_all, y_all, pid_all = Xva, yva, None
    from ml.dataset import load_physionet_batch
    _, _, pids_a = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    _, v_idx = next(gss.split(X_all, y_all, groups=pids_a))
    Xva, yva = X_all[v_idx], y_all[v_idx]

    print('loading full set-a (stride 30)...', flush=True)
    Xa, ya, pa = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a_full', 'set-a'),
        outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
    print('loading full set-b (stride 30)...', flush=True)
    Xb, yb, pb = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-b_full', 'set-b'),
        outcomes_file=os.path.join(BASE, 'Outcomes-b.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
    return (Xva, yva), (Xa, ya, pa), (Xb, yb, pb)


def preds_of(model, X, device):
    from ml.train_lstm import SimpleLSTMDataset
    from torch.utils.data import DataLoader
    model.eval()
    dl = DataLoader(SimpleLSTMDataset(X, np.zeros(len(X))), batch_size=512, shuffle=False)
    pr = np.zeros(len(X)); off = 0
    with torch.no_grad():
        for xb, _ in dl:
            b = xb.shape[0]
            pr[off:off+b] = model(xb.to(device)).cpu().numpy().ravel(); off += b
    return pr


def eval_logits(p, y):
    pr = 1 / (1 + np.exp(-p))
    return {'auc': float(roc_auc_score(y, pr)), 'accuracy': float(accuracy_score(y, pr > 0.5)),
            'recall': float(recall_score(y, pr > 0.5))}


def train_seed(seed, X_tr, y_tr, X_va, y_va, X_fho, y_fho, run_base):
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mean = np.nanmean(flat, axis=0); std = np.nanstd(flat, axis=0) + 1e-6
    mean = np.where(np.isnan(mean), 0.0, mean); std = np.where(np.isnan(std), 1.0, std)
    def norm(X):
        return ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32)
    X_tr = norm(X_tr); X_va = norm(X_va); X_fho = norm(X_fho)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pw = n_neg / max(1, n_pos)
    print(f'  seed {seed}: train {X_tr.shape} dist={np.bincount(y_tr.astype(int))}', flush=True)
    model = AttentionLSTMModel(input_size=12, hidden_size=96, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_auc = 0.0; best_state = None; patience = 0
    for epoch in range(EPOCHS):
        model, optimizer = quick_train(
            X_tr, y_tr, epochs=1, batch_size=128, learning_rate=1e-4, pos_weight=pw,
            model_class=AttentionLSTMModel, model=model, optimizer=optimizer,
            dropout=0.3, hidden_size=96, weight_decay=1e-4)
        for pg in optimizer.param_groups:
            pg['lr'] = max(1e-4 * (0.5 ** (epoch // 6)), 1e-6)
        mets = eval_logits(preds_of(model, X_va, device), y_va)
        if mets['auc'] - best_auc > MIN_DELTA:
            best_auc = mets['auc']; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    model.load_state_dict(best_state)
    p_va = preds_of(model, X_va, device)
    p_fho = preds_of(model, X_fho, device)
    va = eval_logits(p_va, y_va); fh = eval_logits(p_fho, y_fho)
    out_dir = os.path.join(run_base, f'seed{seed}'); os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    np.save(os.path.join(out_dir, 'p_va.npy'), p_va)
    np.save(os.path.join(out_dir, 'p_fho.npy'), p_fho)
    m = {'config': f'combo8k-lr1e4-seed{seed}', 'features': FEATURES_12, 'window': 90,
         'train': 'set-a_full(excl orig val) + 80pct set-b', 'fresh_holdout': '20pct set-b (seed 123)',
         'best_auc': best_auc, 'val_auc': va['auc'], 'val_accuracy': va['accuracy'],
         'val_recall': va['recall'], 'fresh_holdout_auc': fh['auc'],
         'fresh_holdout_accuracy': fh['accuracy'], 'fresh_holdout_recall': fh['recall'],
         'epochs_trained': epoch + 1}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    print(f"  seed {seed}: val={va['auc']:.4f} | fresh-holdout={fh['auc']:.4f}", flush=True)
    return m, {'mean': mean.tolist(), 'std': std.tolist()}


def main():
    val_pack, a_pack, b_pack = build_all()
    Xva, yva = val_pack
    Xa, ya, pa = a_pack
    Xb, yb, pb = b_pack
    val_set = load_orig_val_pids()

    upb = np.unique(pb)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    dummy = np.zeros(len(upb))
    _, ho_idx = next(gss.split(dummy, dummy, groups=upb))
    ho_patients = set(upb[ho_idx])
    tr_mask_b = np.array([p not in ho_patients for p in pb])
    ho_mask_b = ~tr_mask_b
    Xb_tr, yb_tr = Xb[tr_mask_b], yb[tr_mask_b]
    X_fho, y_fho = Xb[ho_mask_b], yb[ho_mask_b]

    tr_mask_a = np.array([p not in val_set for p in pa])
    X_tr = np.concatenate([Xa[tr_mask_a], Xb_tr], axis=0)
    y_tr = np.concatenate([ya[tr_mask_a], yb_tr], axis=0)
    print(f'TRAIN combined: {X_tr.shape} dist={np.bincount(y_tr.astype(int))}', flush=True)
    print(f'VAL(orig): {Xva.shape} | FRESH HOLDOUT(20% set-b): {X_fho.shape} '
          f'dist={np.bincount(y_fho.astype(int))}', flush=True)

    run_base = os.path.join('ml', 'training_runs', 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_base, exist_ok=True)
    results = []
    for seed in SEEDS:
        m, scaler = train_seed(seed, X_tr, y_tr, Xva, yva, X_fho, y_fho, run_base)
        results.append((seed, m, scaler))
    results.sort(key=lambda r: r[1]['val_auc'], reverse=True)
    print('\nRanking:', flush=True)
    for seed, m, _ in results:
        print(f'  seed{seed}: val={m["val_auc"]:.4f} fresh-holdout={m["fresh_holdout_auc"]:.4f}', flush=True)
    best_seed, best_mets, best_scaler = results[0]
    if best_mets['val_auc'] > BASELINE_VAL and best_mets['fresh_holdout_auc'] >= FRESH_HO_MIN:
        import shutil
        shutil.copy2(os.path.join(run_base, f'seed{best_seed}', 'model.pt'),
                     os.path.join('ml', 'models', 'combo8k_best.pt'))
        scaler_out = {'features': FEATURES_12, 'mean': best_scaler['mean'], 'std': best_scaler['std']}
        with open(os.path.join('ml', 'scaler_combo8k.json'), 'w') as f:
            json.dump(scaler_out, f, indent=2)
        print(f'\n  SAVED combo8k candidate seed{best_seed}: val={best_mets["val_auc"]:.4f} '
              f'fresh-holdout={best_mets["fresh_holdout_auc"]:.4f} (deploy decision pending)', flush=True)
    else:
        print('\n  No combo8k config beat deployed ensemble. Keeping existing.', flush=True)
    print(f'  Runs: {run_base}', flush=True)


if __name__ == '__main__':
    main()