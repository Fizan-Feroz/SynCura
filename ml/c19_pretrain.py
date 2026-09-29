# -*- coding: utf-8 -*-
"""Challenge-2019 pretraining experiment (round 3, option 3 from the loader docstring).

Phase 1 (pretrain): train an AttentionLSTM trunk on Challenge-2019 windows
(sepsis-onset flag) - scale lever: ~20k stays vs PhysioNet's ~7k train.
Phase 2 (finetune): warm-start from the trunk and fine-tune on the standard
E1 PhysioNet train set at low LR, evaluated on the IDENTICAL val split
(seed-42 stride-15) and working holdout (seed-123 excl. locked) so AUCs are
directly comparable to the E1 baselines (seed12/13, ensemble 0.8350).

Caveats (see ml/challenge2019_to_features.py): sepsis-onset != mortality;
GCS is all-NaN in Challenge 2019; hourly data is ffill-upsampled to
per-minute. Outputs are CANDIDATE-only: challenge2019 is registry-cleared
for 'train', NOT 'deploy_train'. Locked slices are never re-run; compare
to the recorded locked numbers by reference only.

Usage:
  .\\.venv\\Scripts\\python.exe -m ml.c19_pretrain --smoke
  .\\.venv\\Scripts\\python.exe -m ml.c19_pretrain --phase all --tag c19
"""
import argparse
import datetime
import gc
import json
import os
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from ml.paths import physionet2012_root as _pn_root
from ml.paths import challenge2019_setA as _c19_root
from ml.dataset import SERVING_FEATURES, create_sequences_from_physionet
from ml.challenge2019_to_features import load_challenge2019_batch
from ml.improve_results import (
    CONFIGS, FEATURES_12, EPOCHS, PATIENCE, MIN_DELTA,
    norm_fit, norm_apply, preds_of, eval_logits, model_class_for,
    build_all, split_holdout, acquire_lock, release_lock,
)

BASE = _pn_root()
PRETRAIN_SPLIT_SEED = 7
PRETRAIN_VAL_FRAC = 0.10


def build_c19(max_patients=None, stride=30, window=90, horizon=12.0):
    print('loading Challenge-2019 setA...', flush=True)
    data = load_challenge2019_batch(_c19_root(), max_patients=max_patients)
    print(f'  {len(data)} patients loaded; windowing...', flush=True)
    X, y, p = create_sequences_from_physionet(
        data, vital_features=FEATURES_12, window_minutes=window,
        stride=stride, label_mode='proximity', horizon_hours=horizon,
        gap_channels=False)
    print(f'  C19 windows: {X.shape} dist={np.bincount(y.astype(int))}', flush=True)
    return X, y, np.array(p)


def patient_split_masks(pids, seed, test_size):
    up = np.unique(pids)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, te_idx = next(gss.split(np.zeros(len(up)), np.zeros(len(up)), groups=up))
    te = set(up[te_idx])
    return np.array([p not in te for p in pids]), np.array([p in te for p in pids])


def train_loop(seed, cfg_name, X_tr, y_tr, X_va, y_va, run_base, tag,
               train_desc, eval_desc, init_state=None, lr=1e-4,
               dropout=0.3, window=90, horizon=12.0, epochs=EPOCHS, smoke=False):
    from ml.train_lstm import train as quick_train
    cfg = CONFIGS[cfg_name]
    ModelCls = model_class_for(cfg)
    n_feat = X_tr.shape[-1]
    mean, std = norm_fit(X_tr)
    X_trn, X_van = (norm_apply(X, mean, std) for X in (X_tr, X_va))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum())
    pw = int((y_tr == 0).sum()) / max(1, n_pos)
    print(f'  seed {seed} [{tag} w{window} h{horizon} lr{lr}]: '
          f'train {X_trn.shape} dist={np.bincount(y_tr.astype(int))}', flush=True)
    model = ModelCls(input_size=n_feat, hidden_size=cfg['hidden'],
                     dropout=dropout).to(device)
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
        print('  warm-started from trunk weights', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auc, best_state, patience = 0.0, None, 0
    max_ep = 2 if smoke else epochs
    epoch = 0
    for epoch in range(max_ep):
        model, optimizer = quick_train(
            X_trn, y_tr, epochs=1, batch_size=128, learning_rate=lr,
            pos_weight=pw, model_class=ModelCls, model=model,
            optimizer=optimizer, dropout=dropout, hidden_size=cfg['hidden'],
            weight_decay=1e-4)
        for pg in optimizer.param_groups:
            pg['lr'] = max(lr * (0.5 ** (epoch // 6)), 1e-6)
        mets = eval_logits(preds_of(model, X_van, device), y_va)
        if mets['auc'] - best_auc > MIN_DELTA:
            best_auc, patience = mets['auc'], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    p_va = preds_of(model, X_van, device)
    va = eval_logits(p_va, y_va)
    out_dir = os.path.join(run_base, f'seed{seed}')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    np.save(os.path.join(out_dir, 'p_va.npy'), p_va)
    m = {'config': f'{tag}-seed{seed}', 'model_class': cfg['model'],
         'features': FEATURES_12, 'window': window, 'horizon': horizon,
         'dropout': dropout, 'lr': lr, 'hidden': cfg['hidden'], 'causal': True,
         'train': train_desc, 'eval': eval_desc,
         'best_auc': best_auc, 'val_auc': va['auc'],
         'val_accuracy': va['accuracy'], 'val_recall': va['recall'],
         'epochs_trained': epoch + 1, 'seed': seed}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    with open(os.path.join(out_dir, 'scaler.json'), 'w') as f:
        json.dump({'features': FEATURES_12,
                   'mean': mean.tolist(), 'std': std.tolist()}, f, indent=2)
    print(f"  seed {seed}: val={va['auc']:.4f}", flush=True)
    return m, {k: v.cpu() for k, v in model.state_dict().items()}


def phase_pretrain(tag, seed, smoke=False):
    mp = 200 if smoke else None
    X, y, p = build_c19(max_patients=mp)
    tr_m, va_m = patient_split_masks(p, PRETRAIN_SPLIT_SEED, PRETRAIN_VAL_FRAC)
    run_base = os.path.join('ml', 'training_runs', f'improve_c19pretrain_{tag}')
    os.makedirs(run_base, exist_ok=True)
    m, state = train_loop(
        seed, 'baseline', X[tr_m], y[tr_m], X[va_m], y[va_m], run_base,
        tag='c19pretrain',
        train_desc='challenge2019-setA 90pct (sepsis-onset flag; GCS all-NaN)',
        eval_desc='challenge2019-setA 10pct (seed 7, NOT comparable to mortality AUC)',
        lr=1e-4, smoke=smoke)
    del X, y, p
    gc.collect()
    return os.path.join(run_base, f'seed{seed}', 'model.pt')


def phase_finetune(tag, seeds, trunk_path, smoke=False):
    (Xva, yva), (Xa, ya, pa), (Xb, yb, pb) = build_all(
        False, train_stride=30, smoke=smoke, window=90, horizon=12.0)
    from ml.dataset import load_and_create_sequences
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
    X_fho, y_fho = Xb[ho_mask_b], yb[ho_mask_b]
    print(f'TRAIN combined: {X_tr.shape} dist={np.bincount(y_tr.astype(int))}', flush=True)
    print(f'VAL: {Xva.shape} | FRESH HOLDOUT: {X_fho.shape} '
          f'dist={np.bincount(y_fho.astype(int))}', flush=True)

    run_base = os.path.join('ml', 'training_runs', f'improve_c19finetune_{tag}')
    os.makedirs(run_base, exist_ok=True)
    np.save(os.path.join(run_base, 'y_va.npy'), yva)
    np.save(os.path.join(run_base, 'y_fho.npy'), y_fho)
    trunk = {k: v for k, v in torch.load(trunk_path, map_location='cpu').items()}
    results = []
    for seed in seeds:
        done_marker = os.path.join(run_base, f'seed{seed}', 'metrics.json')
        ckpt_marker = os.path.join(run_base, f'seed{seed}', 'model.pt')
        if os.path.exists(done_marker) and os.path.exists(ckpt_marker):
            m = json.load(open(done_marker))
            print(f"  seed {seed}: SKIPPED (done: val={m['val_auc']:.4f})", flush=True)
            results.append((seed, m))
            continue
        m, _ = train_loop(
            seed, 'baseline', X_tr, y_tr, Xva, yva, run_base,
            tag='c19finetune',
            train_desc='E1 set (set-a_full excl val + 80pct set-b excl locked)',
            eval_desc='E1 val (seed-42 stride-15) - DIRECTLY comparable to E1 baselines',
            init_state=trunk, lr=3e-5, smoke=smoke)
        p_fho = preds_of_seed(run_base, seed, X_fho)
        fh = eval_logits(p_fho, y_fho)
        m.update({'fresh_holdout_auc': fh['auc'],
                  'fresh_holdout_accuracy': fh['accuracy'],
                  'fresh_holdout_recall': fh['recall']})
        np.save(os.path.join(run_base, f'seed{seed}', 'p_fho.npy'), p_fho)
        with open(done_marker, 'w') as f:
            json.dump(m, f, indent=2)
        print(f"  seed {seed}: val={m['val_auc']:.4f} | "
              f"fresh-holdout={fh['auc']:.4f}", flush=True)
        results.append((seed, m))
    results.sort(key=lambda r: r[1]['val_auc'], reverse=True)
    print('\nRanking:', flush=True)
    for seed, m in results:
        print(f"  seed{seed}: val={m['val_auc']:.4f} "
              f"fresh-holdout={m.get('fresh_holdout_auc', float('nan')):.4f}", flush=True)
    print(f'Runs: {run_base}', flush=True)
    return run_base


def preds_of_seed(run_base, seed, X_fho):
    import torch as _t
    from ml.train_lstm import AttentionLSTMModel
    from ml.improve_results import norm_apply
    sd = os.path.join(run_base, f'seed{seed}')
    sc = json.load(open(os.path.join(sd, 'scaler.json')))
    mean = np.array(sc['mean'])
    std = np.array(sc['std'])
    device = _t.device('cuda' if _t.cuda.is_available() else 'cpu')
    model = AttentionLSTMModel(input_size=12, hidden_size=96, dropout=0.3).to(device)
    model.load_state_dict(_t.load(os.path.join(sd, 'model.pt'), map_location='cpu'))
    return preds_of(model, norm_apply(X_fho, mean, std), device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['smoke', 'pretrain', 'finetune', 'all'],
                    default='all')
    ap.add_argument('--seeds', nargs='+', type=int, default=[51, 52, 53])
    ap.add_argument('--pretrain-seed', type=int, default=50)
    ap.add_argument('--trunk', default=None,
                    help='existing trunk model.pt (skips pretrain)')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    smoke = args.phase == 'smoke'
    tag = 'smoke' if smoke else (args.tag or datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    acquire_lock()
    try:
        trunk_path = args.trunk
        if args.phase in ('smoke', 'pretrain', 'all'):
            trunk_path = phase_pretrain(tag, args.pretrain_seed, smoke=smoke)
        if args.phase in ('smoke', 'finetune', 'all'):
            if trunk_path is None:
                raise SystemExit('finetune needs --trunk (no pretrain ran)')
            seeds = [51] if smoke else args.seeds
            run_base = phase_finetune(tag, seeds, trunk_path, smoke=smoke)
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
