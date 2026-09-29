"""Round 12 - warm-start fine-tune of deployed 0.8072 model on FULL PhysioNet set-a (4000 patients).

Warm-starts from ml/models/lstm_baseline.pt, stride 30 (halves sequences,
deployment uses window=90 regardless of training stride). Evaluates each
config on the set-b holdout and deploys the best internal-val config that
does not regress holdout AUC below baseline. Recomputes scaler from full set-a train split.
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
HID = 96
DROP = 0.3
WD = 1e-4
BS = 128
EPOCHS = 40
PATIENCE = 14
MIN_DELTA = 0.0005
BASELINE_HOLDOUT = 0.7640
BASELINE_VAL = 0.7978

CONFIGS = {
    'full-lr1e4-s8':    dict(lr=1e-4, wd=1e-4, step_every=8),
    'full-lr1e4-s5':    dict(lr=1e-4, wd=1e-4, step_every=5),
    'full-lr15e4-s8':   dict(lr=1.5e-4, wd=1e-4, step_every=8),
    'full-lr5e5-s8':    dict(lr=5e-5, wd=1e-4, step_every=8),
    'full-lr1e4-wd0':   dict(lr=1e-4, wd=0.0, step_every=8),
    'ws2-full-lr2e4':   dict(lr=2e-4, wd=1e-4, step_every=8, warm=True),
}


def apply_scaler(X, mean, std):
    return ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32)


def load_scaler_from_file():
    s = json.load(open('ml/scaler.json'))
    return np.array(s['mean']), np.array(s['std'])


def load_split(seed=42):
    from ml.dataset import load_and_create_sequences
    print(f'loading full set-a (stride 30, split seed {seed})...')
    X, y, patient_ids = load_and_create_sequences(
        physionet_dir=TRAIN_DIR, outcomes_file=TRAIN_OUTCOMES,
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    idx, v_idx = next(gss.split(X, y, groups=patient_ids))
    X_tr, y_tr = X[idx], y[idx]
    X_va, y_va = X[v_idx], y[v_idx]
    print(f'X_tr={X_tr.shape} X_va={X_va.shape} dist_tr={np.bincount(y_tr.astype(int))}')
    return X_tr, y_tr, X_va, y_va


def load_holdout():
    from ml.dataset import load_and_create_sequences
    print('loading set-b holdout...')
    X, y, _ = load_and_create_sequences(
        physionet_dir=HOLDOUT_DIR, outcomes_file=HOLDOUT_OUTCOMES,
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
    return X, y


def eval_model(model, X, y, device):
    from ml.train_lstm import SimpleLSTMDataset
    from torch.utils.data import DataLoader
    model.eval()
    dl = DataLoader(SimpleLSTMDataset(X, y), batch_size=512)
    preds = np.zeros(len(y)); ys = np.zeros(len(y))
    with torch.no_grad():
        off = 0
        for xb, yb in dl:
            b = xb.shape[0]
            preds[off:off+b] = model(xb.to(device)).cpu().numpy().ravel()
            ys[off:off+b] = yb.numpy(); off += b
    pr = 1 / (1 + np.exp(-preds))
    return {'auc': float(roc_auc_score(ys, pr)),
            'accuracy': float(accuracy_score(ys, pr > 0.5)),
            'recall': float(recall_score(ys, pr > 0.5))}


def train_config(name, cfg, X_tr, y_tr, X_va, y_va, X_ho, y_ho, mean, std, run_base):
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    X_tr = apply_scaler(X_tr, mean, std)
    X_va = apply_scaler(X_va, mean, std)
    X_ho = apply_scaler(X_ho, mean, std)
    print(f"\n=== CONFIG {name} lr={cfg['lr']} wd={cfg.get('wd', WD)} bs={cfg.get('bs', BS)} warm={cfg.get('warm', False)} ===")
    torch.manual_seed(cfg.get('seed', 9))
    np.random.seed(cfg.get('seed', 9))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pw = n_neg / max(1, n_pos)

    model = AttentionLSTMModel(input_size=12, hidden_size=HID, dropout=DROP)
    if cfg.get('warm'):
        model.load_state_dict(torch.load(WARM_START, map_location='cpu'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('wd', WD))

    best_auc = 0.0; best_state = None; patience = 0
    hist = []
    step_every = cfg.get('step_every', 6)
    for epoch in range(EPOCHS):
        model, optimizer = quick_train(
            X_tr, y_tr, epochs=1, batch_size=cfg.get('bs', BS), learning_rate=cfg['lr'],
            pos_weight=pw, model_class=AttentionLSTMModel, model=model, optimizer=optimizer,
            dropout=DROP, hidden_size=HID, weight_decay=cfg.get('wd', WD))
        for pg in optimizer.param_groups:
            pg['lr'] = max(cfg['lr'] * (0.5 ** (epoch // step_every)), 1e-6)
        mets = eval_model(model, X_va, y_va, device)
        hist.append(mets['auc'])
        print(f"  E{epoch+1:02d} val AUC={mets['auc']:.4f} acc={mets['accuracy']:.4f} rec={mets['recall']:.4f}")
        if mets['auc'] - best_auc > MIN_DELTA:
            best_auc = mets['auc']; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            break

    model.load_state_dict(best_state)
    val = eval_model(model, X_va, y_va, device)
    ho = eval_model(model, X_ho, y_ho, device) if X_ho is not None else {'auc': None}
    out_dir = os.path.join(run_base, name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    final = {'config': name, 'features': FEATURES_12, 'window': 90, 'hidden_size': HID,
             'dropout': DROP, 'weight_decay': cfg['wd'], 'batch_size': cfg.get('bs', BS),
             'lr': cfg['lr'], 'seed': cfg.get('seed', 9), 'stride': 30,
             'warm_start': cfg.get('warm', False), 'scaler': 'ml/scaler.json',
             'best_auc': best_auc, 'val_auc': val['auc'], 'val_accuracy': val['accuracy'],
             'val_recall': val['recall'], 'holdout_auc': ho['auc'],
             'holdout_accuracy': ho['accuracy'], 'holdout_recall': ho['recall'],
             'epochs_trained': epoch + 1, 'auc_history': hist}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(final, f, indent=2)
    print(f"  => {name}: val AUC={final['val_auc']:.4f} | set-b holdout AUC={final['holdout_auc']:.4f}")
    return final, os.path.join(out_dir, 'model.pt')


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    X_ho, y_ho = load_holdout()
    print(f'holdout: X={X_ho.shape} dist={np.bincount(y_ho.astype(int))}')
    mean, std = load_scaler_from_file()
    # strided/oversampled sequences may exceed the deployed scaler's range;
    # recompute from full set-a train split but keep feature order stable.
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    m2 = np.nanmean(flat, axis=0); s2 = np.nanstd(flat, axis=0) + 1e-6
    m2 = np.where(np.isnan(m2), 0.0, m2); s2 = np.where(np.isnan(s2), 1.0, s2)
    mean, std = m2, s2
    run_base = os.path.join('ml', 'training_runs', 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_base, exist_ok=True)

    results = []
    for name, cfg in CONFIGS.items():
        final, path = train_config(name, cfg, X_tr, y_tr, X_va, y_va, X_ho, y_ho, mean, std, run_base)
        results.append((name, final, path))

    results.sort(key=lambda r: r[1]['val_auc'], reverse=True)
    best_name, best_mets, best_path = results[0]
    print(f"\nRanking by val AUC (holdout >= {BASELINE_HOLDOUT}, val > {BASELINE_VAL}):")
    for name, mets, p in results:
        flag = '' if mets['holdout_auc'] >= BASELINE_HOLDOUT else ' (holdout regression)'
        print(f"  {name}: val={mets['val_auc']:.4f} holdout={mets['holdout_auc']:.4f}{flag}")

    if best_mets['holdout_auc'] >= BASELINE_HOLDOUT and best_mets['val_auc'] > BASELINE_VAL:
        import shutil
        shutil.copy2(best_path, os.path.join('ml', 'models', 'lstm_baseline.pt'))
        scaler_out = {'features': FEATURES_12, 'mean': mean.tolist(), 'std': std.tolist()}
        with open(os.path.join('ml', 'scaler.json'), 'w') as f:
            json.dump(scaler_out, f, indent=2)
        deploy = {k: v for k, v in best_mets.items() if k != 'auc_history'}
        with open(os.path.join('ml', 'metrics.json'), 'w') as f:
            json.dump(deploy, f, indent=2)
        print(f"\n  DEPLOYED {best_name}: val AUC={best_mets['val_auc']:.4f} holdout AUC={best_mets['holdout_auc']:.4f}")
    else:
        print(f"\n  No config beat current deploy. Keeping existing (val 0.8072).")
    print(f"  Runs: {run_base}")


if __name__ == '__main__':
    main()