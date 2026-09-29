"""Round 14 - low-LR warm-start fine-tune on FULL set-a (12 features, stride 30).

Deployed 12-feature model has stable 0.798 val / 0.764 holdout on full-data
splits. Warm-start keeps learned features, small steps adapt to more data.
Deploys only if val > baseline and holdout >= baseline.
"""
import datetime
import json
import os
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

from ml.paths import physionet2012_root as _pn_root

BASE = _pn_root()
TRAIN_DIR = os.path.join(BASE, "set-a_full", "set-a")
TRAIN_OUTCOMES = os.path.join(BASE, "Outcomes-a.txt")
HOLDOUT_DIR = os.path.join(BASE, "set-b_full", "set-b")
HOLDOUT_OUTCOMES = os.path.join(BASE, "Outcomes-b.txt")

from ml.dataset import SERVING_FEATURES

FEATURES_12 = SERVING_FEATURES
WARM_START = os.path.join('ml', 'models', 'lstm_baseline.pt')
BASELINE_VAL = 0.7978
BASELINE_HOLDOUT = 0.7640

CONFIGS = {
    'ws-lr5e5-s8':    dict(lr=5e-5, wd=1e-4, step_every=8, dropout=0.3),
    'ws-lr3e5-s10':   dict(lr=3e-5, wd=1e-4, step_every=10, dropout=0.3),
    'ws-lr5e5-s8-do4': dict(lr=5e-5, wd=1e-4, step_every=8, dropout=0.4),
    'ws-lr1e4-s10':   dict(lr=1e-4, wd=1e-4, step_every=10, dropout=0.3),
    'ws-lr5e5-s8-wd2': dict(lr=5e-5, wd=2e-4, step_every=8, dropout=0.3),
}
EPOCHS = 40
PATIENCE = 12
MIN_DELTA = 0.0004


def load_split(seed=42):
    from ml.dataset import load_and_create_sequences
    print(f'loading full set-a (stride 30, seed {seed})...')
    X, y, pid = load_and_create_sequences(
        physionet_dir=TRAIN_DIR, outcomes_file=TRAIN_OUTCOMES,
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    idx, v_idx = next(gss.split(X, y, groups=pid))
    X_tr, y_tr = X[idx], y[idx]
    X_va, y_va = X[v_idx], y[v_idx]
    print(f'X_tr={X_tr.shape} X_va={X_va.shape} dist_tr={np.bincount(y_tr.astype(int))}')
    return X_tr, y_tr, X_va, y_va


def apply_scaler(X):
    s = json.load(open('ml/scaler.json'))
    mean = np.array(s['mean']); std = np.array(s['std'])
    return ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32), mean, std


def eval_model(model, X, y, device):
    from ml.train_lstm import SimpleLSTMDataset
    from torch.utils.data import DataLoader
    model.eval()
    dl = DataLoader(SimpleLSTMDataset(X, y), batch_size=512)
    preds = np.zeros(len(y)); ys = np.zeros(len(y)); off = 0
    with torch.no_grad():
        for xb, yb in dl:
            b = xb.shape[0]
            preds[off:off+b] = model(xb.to(device)).cpu().numpy().ravel()
            ys[off:off+b] = yb.numpy(); off += b
    pr = 1 / (1 + np.exp(-preds))
    return {'auc': float(roc_auc_score(ys, pr)), 'accuracy': float(accuracy_score(ys, pr > 0.5)),
            'recall': float(recall_score(ys, pr > 0.5))}


def run(name, cfg, X_tr, y_tr, X_va, y_va, X_ho, y_ho, run_base):
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    X_tr, mean, std = apply_scaler(X_tr)
    X_va, _, _ = apply_scaler(X_va)
    X_ho, _, _ = apply_scaler(X_ho)
    print(f"\n=== {name} lr={cfg['lr']} wd={cfg['wd']} do={cfg['dropout']} step={cfg['step_every']} ===")
    torch.manual_seed(cfg.get('seed', 21)); np.random.seed(cfg.get('seed', 21))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pw = n_neg / max(1, n_pos)
    model = AttentionLSTMModel(input_size=12, hidden_size=96, dropout=cfg['dropout'])
    model.load_state_dict(torch.load(WARM_START, map_location='cpu'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    best_auc = 0.0; best_state = None; patience = 0; hist = []
    for epoch in range(EPOCHS):
        model, optimizer = quick_train(
            X_tr, y_tr, epochs=1, batch_size=128, learning_rate=cfg['lr'], pos_weight=pw,
            model_class=AttentionLSTMModel, model=model, optimizer=optimizer,
            dropout=cfg['dropout'], hidden_size=96, weight_decay=cfg['wd'])
        for pg in optimizer.param_groups:
            pg['lr'] = max(cfg['lr'] * (0.5 ** (epoch // cfg['step_every'])), 1e-6)
        mets = eval_model(model, X_va, y_va, device)
        hist.append(mets['auc'])
        if mets['auc'] - best_auc > MIN_DELTA:
            best_auc = mets['auc']; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    model.load_state_dict(best_state)
    va = eval_model(model, X_va, y_va, device)
    ho = eval_model(model, X_ho, y_ho, device)
    out_dir = os.path.join(run_base, name); os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    m = {'config': name, 'features': FEATURES_12, 'window': 90, 'hidden_size': 96,
         'dropout': cfg['dropout'], 'weight_decay': cfg['wd'], 'lr': cfg['lr'],
         'step_every': cfg['step_every'], 'seed': cfg.get('seed', 21), 'stride': 30,
         'warm_start': True, 'scaler': 'ml/scaler.json',
         'best_auc': best_auc, 'val_auc': va['auc'], 'val_accuracy': va['accuracy'],
         'val_recall': va['recall'], 'holdout_auc': ho['auc'],
         'holdout_accuracy': ho['accuracy'], 'holdout_recall': ho['recall'],
         'epochs_trained': epoch + 1, 'auc_history': hist}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    print(f"  => {name}: val AUC={va['auc']:.4f} | holdout AUC={ho['auc']:.4f}")
    return m, os.path.join(out_dir, 'model.pt'), hist


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    from ml.dataset import load_and_create_sequences
    print('loading set-b holdout...')
    X_ho, y_ho, _ = load_and_create_sequences(
        physionet_dir=HOLDOUT_DIR, outcomes_file=HOLDOUT_OUTCOMES,
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
    print(f'holdout: X={X_ho.shape} dist={np.bincount(y_ho.astype(int))}')
    run_base = os.path.join('ml', 'training_runs', 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_base, exist_ok=True)
    results = []
    for name, cfg in CONFIGS.items():
        m, path, hist = run(name, cfg, X_tr, y_tr, X_va, y_va, X_ho, y_ho, run_base)
        results.append((name, m, path))
    results.sort(key=lambda r: r[1]['val_auc'], reverse=True)
    print(f"\nRanking by val AUC (val > {BASELINE_VAL}, holdout >= {BASELINE_HOLDOUT}):")
    for name, m, p in results:
        flag = '' if m['holdout_auc'] >= BASELINE_HOLDOUT else ' (holdout regression)'
        print(f"  {name}: val={m['val_auc']:.4f} holdout={m['holdout_auc']:.4f}{flag}")
    best_name, best_mets, best_path = results[0]
    if best_mets['val_auc'] > BASELINE_VAL and best_mets['holdout_auc'] >= BASELINE_HOLDOUT:
        import shutil
        shutil.copy2(best_path, os.path.join('ml', 'models', 'lstm_baseline.pt'))
        deploy = {k: v for k, v in best_mets.items() if k != 'auc_history'}
        with open(os.path.join('ml', 'metrics.json'), 'w') as f:
            json.dump(deploy, f, indent=2)
        print(f"\n  DEPLOYED {best_name}: val={best_mets['val_auc']:.4f} holdout={best_mets['holdout_auc']:.4f}")
    else:
        print('\n  No config beat baseline. Keeping existing deploy.')
    print(f'  Runs: {run_base}')


if __name__ == '__main__':
    main()