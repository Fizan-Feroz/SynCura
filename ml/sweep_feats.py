"""Round 13 - feature-count A/B on the fast 1519-patient subset.

12-feature (deployed) vs 20-feature (labs: K Na HCO3 Mg HCT pH PaO2 PaCO2)
vs 16-feature (K Na HCO3 Mg HCT pH PaO2). Same splits, same schedule.
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
TRAIN_DIR = os.path.join(BASE, "set-a")
TRAIN_OUTCOMES = os.path.join(BASE, "Outcomes-a.txt")

from ml.dataset import SERVING_FEATURES

F12 = SERVING_FEATURES
ADD6 = ['K', 'Na', 'HCO3', 'Mg', 'HCT', 'pH']
ADD8 = ADD6 + ['PaO2', 'PaCO2']
F16 = F12 + ADD6
F20 = F12 + ADD8

EPOCHS = 40
PATIENCE = 14
MIN_DELTA = 0.0005

CONFIGS = [
    ('f20-lr2e4-s6', F20, dict(lr=2e-4, wd=1e-4, step_every=6, seed=11)),
    ('f20-lr1e4-s6', F20, dict(lr=1e-4, wd=1e-4, step_every=6, seed=12)),
    ('f16-lr2e4-s6', F16, dict(lr=2e-4, wd=1e-4, step_every=6, seed=13)),
]


def load_split(features):
    from ml.dataset import load_and_create_sequences
    X, y, patient_ids = load_and_create_sequences(
        physionet_dir=TRAIN_DIR, outcomes_file=TRAIN_OUTCOMES,
        vital_features=features, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx, v_idx = next(gss.split(X, y, groups=patient_ids))
    X_tr, y_tr = X[idx], y[idx]
    X_va, y_va = X[v_idx], y[v_idx]
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mean = np.nanmean(flat, axis=0); std = np.nanstd(flat, axis=0) + 1e-6
    mean = np.where(np.isnan(mean), 0.0, mean); std = np.where(np.isnan(std), 1.0, std)
    X_tr = ((np.where(np.isnan(X_tr), mean, X_tr) - mean) / std).astype(np.float32)
    X_va = ((np.where(np.isnan(X_va), mean, X_va) - mean) / std).astype(np.float32)
    print(f'  f={len(features)} X_tr={X_tr.shape} X_va={X_va.shape} dist_tr={np.bincount(y_tr.astype(int))} nan%={np.isnan(flat).mean()*100:.1f}')
    return X_tr, y_tr, X_va, y_va


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


def train(name, features, cfg):
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    X_tr, y_tr, X_va, y_va = load_split(features)
    print(f"=== {name} cfg={cfg} ===")
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pw = n_neg / max(1, n_pos)
    model = AttentionLSTMModel(input_size=len(features), hidden_size=96, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    best_auc = 0.0; best_state = None; patience = 0
    for epoch in range(EPOCHS):
        model, optimizer = quick_train(X_tr, y_tr, epochs=1, batch_size=128, learning_rate=cfg['lr'],
            pos_weight=pw, model_class=AttentionLSTMModel, model=model, optimizer=optimizer,
            dropout=0.3, hidden_size=96, weight_decay=cfg['wd'])
        for pg in optimizer.param_groups:
            pg['lr'] = max(cfg['lr'] * (0.5 ** (epoch // cfg['step_every'])), 1e-6)
        mets = eval_model(model, X_va, y_va, device)
        if mets['auc'] - best_auc > MIN_DELTA:
            best_auc = mets['auc']; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    model.load_state_dict(best_state)
    va = eval_model(model, X_va, y_va, device)
    out = os.path.join('ml', 'training_runs', 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), name)
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))
    m = {'config': name, 'features': features, 'best_auc': best_auc, 'val_auc': va['auc'],
         'val_accuracy': va['accuracy'], 'val_recall': va['recall'], 'lr': cfg['lr'],
         'wd': cfg['wd'], 'step_every': cfg['step_every'], 'seed': cfg['seed']}
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    print(f"  => {name}: val AUC={va['auc']:.4f} acc={va['accuracy']:.4f} rec={va['recall']:.4f} (best {best_auc:.4f})")


if __name__ == '__main__':
    for name, features, cfg in CONFIGS:
        train(name, features, cfg)