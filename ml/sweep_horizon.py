"""Round 21 - label-horizon test: 24h proximity (paper's dynamic task) vs 12h.

Trains 2 seeds of the winning lr1e4 recipe at horizon 24 on full set-a
(minus original val patients), evaluates on 24h val + 24h holdout.
Also evaluates the deployed xval-lr1e4 single at 24h as reference.
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
EPOCHS = 40
PATIENCE = 14
MIN_DELTA = 0.0005
HORIZON = 24
SEEDS = [71, 72]


def load_orig_split(horizon, seed=42):
    from ml.dataset import load_and_create_sequences
    X, y, pid = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=horizon)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, v_idx = next(gss.split(X, y, groups=pid))
    return X[v_idx], y[v_idx]


def load_train_excluding(val_pids, horizon):
    from ml.dataset import load_and_create_sequences
    X, y, pid = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a_full', 'set-a'),
        outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=horizon)
    mask = np.array([p not in val_pids for p in pid])
    return X[mask], y[mask]


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


def train_seed(seed, X_tr, y_tr, X_va, y_va, X_ho, y_ho, run_base):
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mean = np.nanmean(flat, axis=0); std = np.nanstd(flat, axis=0) + 1e-6
    mean = np.where(np.isnan(mean), 0.0, mean); std = np.where(np.isnan(std), 1.0, std)
    def norm(X):
        return ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32)
    X_tr = norm(X_tr); X_va = norm(X_va); X_ho = norm(X_ho)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pw = n_neg / max(1, n_pos)
    print(f'  train dist={np.bincount(y_tr.astype(int))} pw={pw:.2f}', flush=True)
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
    p_ho = preds_of(model, X_ho, device)
    va = eval_logits(p_va, y_va); ho = eval_logits(p_ho, y_ho)
    out_dir = os.path.join(run_base, f'seed{seed}'); os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    m = {'config': f'h{HORIZON}-lr1e4-seed{seed}', 'horizon_hours': HORIZON, 'features': FEATURES_12,
         'best_auc': best_auc, 'val_auc': va['auc'], 'val_accuracy': va['accuracy'],
         'val_recall': va['recall'], 'holdout_auc': ho['auc'],
         'holdout_accuracy': ho['accuracy'], 'holdout_recall': ho['recall'],
         'epochs_trained': epoch + 1}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    print(f"  seed {seed} (h{HORIZON}): val AUC={va['auc']:.4f} | holdout AUC={ho['auc']:.4f}", flush=True)
    return m


def main():
    from ml.dataset import load_and_create_sequences
    from ml.train_lstm import AttentionLSTMModel
    X_va, y_va = load_orig_split(HORIZON)
    _, _, val_pids = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=HORIZON)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, v_idx = next(gss.split(np.zeros(len(val_pids)), np.zeros(len(val_pids)), groups=val_pids))
    val_set = set(np.array(val_pids)[v_idx])
    X_tr, y_tr = load_train_excluding(val_set, HORIZON)
    print(f'train: {X_tr.shape} | val(h{HORIZON}): {X_va.shape} dist_va={np.bincount(y_va.astype(int))}', flush=True)
    X_ho, y_ho, _ = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-b_full', 'set-b'), outcomes_file=os.path.join(BASE, 'Outcomes-b.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=HORIZON)
    print(f'holdout(h{HORIZON}): {X_ho.shape} dist={np.bincount(y_ho.astype(int))}', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s = json.load(open('ml/scaler.json'))
    mean = np.array(s['mean']); std = np.array(s['std'])
    Xva_n = ((np.where(np.isnan(X_va), mean, X_va) - mean) / std).astype(np.float32)
    Xho_n = ((np.where(np.isnan(X_ho), mean, X_ho) - mean) / std).astype(np.float32)
    ref = AttentionLSTMModel(input_size=12, hidden_size=96, dropout=0.3)
    ref.load_state_dict(torch.load('ml/training_runs/exp_20260916_193507/full-xval-lr1e4/model.pt',
                                   map_location='cpu'))
    ref.eval().to(device)
    rv = eval_logits(preds_of(ref, Xva_n, device), y_va)
    rh = eval_logits(preds_of(ref, Xho_n, device), y_ho)
    print(f'  REFERENCE xval-lr1e4 (12h-trained) @24h labels: val={rv["auc"]:.4f} holdout={rh["auc"]:.4f}', flush=True)

    run_base = os.path.join('ml', 'training_runs', 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_base, exist_ok=True)
    for seed in SEEDS:
        train_seed(seed, X_tr, y_tr, X_va, y_va, X_ho, y_ho, run_base)
    print(f'  Runs: {run_base}', flush=True)


if __name__ == '__main__':
    main()