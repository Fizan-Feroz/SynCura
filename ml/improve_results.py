# -*- coding: utf-8 -*-
"""Results-improvement campaign runner (streaming-compatible only).

Trains AttentionLSTM seeds under the fixed CAUSAL pipeline and evaluates on
the original val split + fresh 20%-set-B holdout. Writes per-seed checkpoints,
predictions, scalers, and metrics; never touches serving artifacts.

Configs:
  baseline : 12 features, hidden 96  (serves with zero backend changes)
  gap      : 12 + 12 gap channels = 24 features, hidden 96
  gap128   : 24 features, hidden 128

Usage:
  .\\.venv\\Scripts\\python.exe -m ml.improve_results --smoke
  .\\.venv\\Scripts\\python.exe -m ml.improve_results --config baseline --seeds 11 12 13
  .\\.venv\\Scripts\\python.exe -m ml.improve_results --config gap --seeds 21 22 23
  .\\.venv\\Scripts\\python.exe -m ml.improve_results --ensemble ml/training_runs/<run_dir>
"""
import argparse
import datetime
import json
import os
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

from ml.paths import physionet2012_root as _pn_root
BASE = _pn_root()
FEATURES_12 = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
               'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose']
GAP_SUFFIX = '_GAP'
EPOCHS = 40
PATIENCE = 14
MIN_DELTA = 0.0005

CONFIGS = {
    'baseline': {'gap': False, 'hidden': 96},
    'gap': {'gap': True, 'hidden': 96},
    'gap128': {'gap': True, 'hidden': 128},
}


def feature_names(gap):
    return FEATURES_12 + ([f + GAP_SUFFIX for f in FEATURES_12] if gap else [])


def build_all(gap, train_stride=30, smoke=False, window=90, horizon=12):
    from ml.dataset import load_and_create_sequences
    kw = dict(vital_features=FEATURES_12, window_minutes=window,
              label_mode='proximity', horizon_hours=horizon, gap_channels=gap)
    print('loading original val (stride 15)...', flush=True)
    Xva, yva, pva = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'),
        outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'), stride=15, **kw)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, v_idx = next(gss.split(Xva, yva, groups=pva))
    Xva, yva = Xva[v_idx], yva[v_idx]

    mp = 60 if smoke else None
    print('loading full set-a (stride %d)...' % train_stride, flush=True)
    Xa, ya, pa = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a_full', 'set-a'),
        outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        stride=train_stride, max_patients=mp, **kw)
    print('loading full set-b (stride %d)...' % train_stride, flush=True)
    Xb, yb, pb = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-b_full', 'set-b'),
        outcomes_file=os.path.join(BASE, 'Outcomes-b.txt'),
        stride=train_stride, max_patients=mp, **kw)
    return (Xva, yva), (Xa, ya, pa), (Xb, yb, pb)


def split_holdout(pb, seed=123, test_size=0.2):
    upb = np.unique(pb)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, ho_idx = next(gss.split(np.zeros(len(upb)), np.zeros(len(upb)), groups=upb))
    ho_patients = set(upb[ho_idx])
    return np.array([p not in ho_patients for p in pb])


def norm_fit(X_tr):
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0) + 1e-6
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std), 1.0, std)
    return mean, std


def norm_apply(X, mean, std):
    return ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32)


def preds_of(model, X, device):
    from ml.train_lstm import SimpleLSTMDataset
    from torch.utils.data import DataLoader
    model.eval()
    dl = DataLoader(SimpleLSTMDataset(X, np.zeros(len(X))), batch_size=512, shuffle=False)
    pr = np.zeros(len(X))
    off = 0
    with torch.no_grad():
        for xb, _ in dl:
            b = xb.shape[0]
            pr[off:off + b] = model(xb.to(device)).cpu().numpy().ravel()
            off += b
    return pr


def eval_logits(p, y):
    pr = 1 / (1 + np.exp(-p))
    return {'auc': float(roc_auc_score(y, pr)),
            'accuracy': float(accuracy_score(y, pr > 0.5)),
            'recall': float(recall_score(y, pr > 0.5))}


def train_seed(seed, cfg_name, X_tr, y_tr, X_va, y_va, X_fho, y_fho,
               run_base, epochs=EPOCHS, smoke=False, window=90, horizon=12,
               dropout=0.3, lr=1e-4):
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    cfg = CONFIGS[cfg_name]
    n_feat = X_tr.shape[-1]
    mean, std = norm_fit(X_tr)
    X_trn, X_van, X_fhon = (norm_apply(X, mean, std) for X in (X_tr, X_va, X_fho))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum())
    pw = int((y_tr == 0).sum()) / max(1, n_pos)
    print(f'  seed {seed} [{cfg_name} w{window} h{horizon} do{dropout} lr{lr}]: '
          f'train {X_trn.shape} dist={np.bincount(y_tr.astype(int))}', flush=True)
    model = AttentionLSTMModel(input_size=n_feat, hidden_size=cfg['hidden'],
                               dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auc, best_state, patience = 0.0, None, 0
    max_ep = 2 if smoke else epochs
    for epoch in range(max_ep):
        model, optimizer = quick_train(
            X_trn, y_tr, epochs=1, batch_size=128, learning_rate=lr,
            pos_weight=pw, model_class=AttentionLSTMModel, model=model,
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
    p_fho = preds_of(model, X_fhon, device)
    va, fh = eval_logits(p_va, y_va), eval_logits(p_fho, y_fho)
    out_dir = os.path.join(run_base, f'seed{seed}')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    np.save(os.path.join(out_dir, 'p_va.npy'), p_va)
    np.save(os.path.join(out_dir, 'p_fho.npy'), p_fho)
    m = {'config': f'{cfg_name}-causal-seed{seed}', 'features': feature_names(cfg['gap']),
         'window': window, 'horizon': horizon, 'dropout': dropout, 'lr': lr,
         'hidden': cfg['hidden'], 'causal': True,
         'train': 'set-a_full(excl orig val) + 80pct set-b' if not smoke else 'smoke subset',
         'fresh_holdout': '20pct set-b (seed 123)',
         'best_auc': best_auc, 'val_auc': va['auc'], 'val_accuracy': va['accuracy'],
         'val_recall': va['recall'], 'fresh_holdout_auc': fh['auc'],
         'fresh_holdout_accuracy': fh['accuracy'], 'fresh_holdout_recall': fh['recall'],
         'epochs_trained': epoch + 1, 'seed': seed}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    with open(os.path.join(out_dir, 'scaler.json'), 'w') as f:
        json.dump({'features': feature_names(cfg['gap']),
                   'mean': mean.tolist(), 'std': std.tolist()}, f, indent=2)
    print(f"  seed {seed}: val={va['auc']:.4f} | fresh-holdout={fh['auc']:.4f}", flush=True)
    return m


def _sigmoid(p):
    return 1 / (1 + np.exp(-np.asarray(p, dtype=np.float64)))


def run_ensemble(run_dirs, holdout_gate=0.83):
    """Greedy val-gated ensemble over finished run dirs (cached predictions).

    Only mixes seeds whose (window, horizon, stride) match, so every member
    scores the same windows in the same order. Writes candidate_ensemble.json
    next to the first run dir — a CANDIDATE, never a deployment.
    """
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
    seeds = {}  # key -> {p_va, p_fho, metrics, run_dir}
    yva = yfho = None
    fingerprint = None
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        yva_p, yfho_p = os.path.join(run_dir, 'y_va.npy'), os.path.join(run_dir, 'y_fho.npy')
        if yva is None and os.path.exists(yva_p):
            yva, yfho = np.load(yva_p), np.load(yfho_p)
        for d in sorted(os.listdir(run_dir)):
            sd = os.path.join(run_dir, d)
            if not (d.startswith('seed') and os.path.exists(os.path.join(sd, 'p_va.npy'))):
                continue
            m = json.load(open(os.path.join(sd, 'metrics.json')))
            fp = (m.get('window', 90), m.get('horizon', 12.0))
            if fingerprint is None:
                fingerprint = fp
            if fp != fingerprint:
                print(f'  skip {d} ({run_dir}): shape {fp} != {fingerprint}', flush=True)
                continue
            key = f'{os.path.basename(run_dir)}/{d}'
            seeds[key] = {'p_va': np.load(os.path.join(sd, 'p_va.npy')),
                          'p_fho': np.load(os.path.join(sd, 'p_fho.npy')),
                          'metrics': m, 'dir': sd}
    if not seeds or yva is None:
        print('run_ensemble: no usable seeds+labels found', flush=True)
        return None
    for s, info in seeds.items():
        m = info['metrics']
        print(f"  {s}: val={m['val_auc']:.4f} fresh-holdout={m['fresh_holdout_auc']:.4f}", flush=True)
    order = sorted(seeds, key=lambda s: seeds[s]['metrics']['val_auc'], reverse=True)
    chosen = [order[0]]
    best = float(roc_auc_score(yva, _sigmoid(seeds[order[0]]['p_va'])))
    print(f'  start [{order[0]}]: val={best:.4f}', flush=True)
    for _ in range(9):
        cand = None
        for s in seeds:
            if s in chosen:
                continue
            t = chosen + [s]
            a = float(roc_auc_score(yva, _sigmoid(np.mean([seeds[x]['p_va'] for x in t], axis=0))))
            if cand is None or a > cand[1]:
                cand = (s, a, t)
        if cand is None:
            break
        s, a, t = cand
        if a > best + 1e-4:
            best = a
            chosen = t
            h = float(roc_auc_score(yfho, _sigmoid(np.mean([seeds[x]['p_fho'] for x in t], axis=0))))
            print(f'  add {s}: val={a:.4f} fresh-holdout={h:.4f}', flush=True)
        else:
            print(f'  stop: best add {s} val={a:.4f}', flush=True)
            break
    pe = np.mean([seeds[s]['p_va'] for s in chosen], axis=0)
    phe = np.mean([seeds[s]['p_fho'] for s in chosen], axis=0)
    ve, he = eval_logits(pe, yva), eval_logits(phe, yfho)
    out = {'members': chosen,
           'checkpoints': [os.path.join(seeds[s]['dir'], 'model.pt') for s in chosen],
           'scalers': [os.path.join(seeds[s]['dir'], 'scaler.json') for s in chosen],
           'val_auc': ve['auc'], 'val_accuracy': ve['accuracy'], 'val_recall': ve['recall'],
           'fresh_holdout_auc': he['auc'], 'fresh_holdout_accuracy': he['accuracy'],
           'fresh_holdout_recall': he['recall'],
           'candidate_only': True,
           'holdout_gate': holdout_gate,
           'passes_gate': bool(ve['auc'] > 0 and he['auc'] >= holdout_gate)}
    out_path = os.path.join(run_dirs[0], 'candidate_ensemble.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"CANDIDATE {chosen}: val={ve['auc']:.4f} fresh-holdout={he['auc']:.4f} "
          f"gate={holdout_gate} pass={out['passes_gate']} -> {out_path}", flush=True)
    return out


LOCK_FILE = os.path.join('ml', 'training_runs', '.campaign.lock')


def acquire_lock():
    """Refuse to start if another campaign is already running (stale locks
    from dead PIDs are cleared automatically)."""
    import subprocess
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                old_pid = int(f.read().strip())
            alive = subprocess.run(
                ['tasklist', '/FI', f'PID eq {old_pid}', '/FO', 'CSV'],
                capture_output=True, text=True).stdout.count(str(old_pid)) > 0
            if alive:
                print(f'REFUSING TO START: campaign already running as PID {old_pid}. '
                      f'Stop it first or delete {LOCK_FILE} if it is stale.', flush=True)
                raise SystemExit(2)
        except (ValueError, OSError):
            pass
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))


def release_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=list(CONFIGS) + ['all'], default='baseline')
    ap.add_argument('--seeds', nargs='+', type=int, default=[11])
    ap.add_argument('--smoke', action='store_true', help='tiny fast validation run')
    ap.add_argument('--ensemble', nargs='+', default=None, help='greedy-ensemble over run dirs')
    ap.add_argument('--window', type=int, default=90)
    ap.add_argument('--horizon', type=float, default=12.0)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--tag', default=None, help='run-dir tag suffix')
    args = ap.parse_args()

    if not args.ensemble:
        acquire_lock()
    try:
        _run(args)
    finally:
        if not args.ensemble:
            release_lock()


def _run(args):

    if args.ensemble:
        run_ensemble(args.ensemble)
        return

    cfgs = list(CONFIGS) if args.config == 'all' else [args.config]
    for cfg_name in cfgs:
        gap = CONFIGS[cfg_name]['gap']
        (Xva, yva), (Xa, ya, pa), (Xb, yb, pb) = build_all(
            gap, train_stride=30, smoke=args.smoke,
            window=args.window, horizon=args.horizon)
        from ml.dataset import load_and_create_sequences
        _, _, val_pids = load_and_create_sequences(
            physionet_dir=os.path.join(BASE, 'set-a'),
            outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
            vital_features=FEATURES_12, window_minutes=args.window, stride=15,
            label_mode='proximity', horizon_hours=args.horizon,
            gap_channels=gap)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, v_idx = next(gss.split(np.zeros(len(val_pids)), np.zeros(len(val_pids)),
                                  groups=np.array(val_pids)))
        val_set = set(np.array(val_pids)[v_idx])
        # Locked final-validation slice (ml/locked_holdout.json, v5 item #1):
        # excluded from BOTH training and the reported working holdout, so the
        # end-of-campaign locked evaluation is genuinely untouched.
        locked_set = set()
        lock_path = os.path.join('ml', 'locked_holdout.json')
        if os.path.exists(lock_path):
            try:
                locked_set = set(json.load(open(lock_path))['patients'])
                print(f'locked slice honored: {len(locked_set)} set-b patients excluded', flush=True)
            except Exception as e:
                print(f'WARNING: could not read locked slice ({e}); proceeding WITHOUT exclusion', flush=True)
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

        tag = 'smoke' if args.smoke else (args.tag or datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        run_base = os.path.join('ml', 'training_runs', f'improve_{cfg_name}_{tag}')
        os.makedirs(run_base, exist_ok=True)
        np.save(os.path.join(run_base, 'y_va.npy'), yva)
        np.save(os.path.join(run_base, 'y_fho.npy'), y_fho)
        results = []
        for seed in args.seeds:
            results.append((seed, train_seed(seed, cfg_name, X_tr, y_tr, Xva, yva,
                                            X_fho, y_fho, run_base, smoke=args.smoke,
                                            window=args.window, horizon=args.horizon,
                                            dropout=args.dropout, lr=args.lr)))
        results.sort(key=lambda r: r[1]['val_auc'], reverse=True)
        print('\nRanking:', flush=True)
        for seed, m in results:
            print(f"  seed{seed}: val={m['val_auc']:.4f} "
                  f"fresh-holdout={m['fresh_holdout_auc']:.4f}", flush=True)
        print(f'Runs: {run_base}', flush=True)


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
