"""Evaluate candidate ensembles precisely using saved seed logits + fresh xval inference."""
import json
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

from ml.dataset import SERVING_FEATURES
from ml.paths import physionet2012_root as _pn_root

BASE = _pn_root()

R16 = 'ml/training_runs/exp_20260916_213919'
R17 = 'ml/training_runs/exp_20260916_220047'
XVAL = 'ml/training_runs/exp_20260916_193507'
XVAL_MAP = {'xval-lr1e4': 'full-xval-lr1e4', 'xval-lr2e4': 'full-xval-lr2e4',
            'xval-lr2e4b': 'full-xval-lr2e4b', 'xval-lr3e4': 'full-xval-lr3e4'}

CANDIDATES = [
    ['s48', 'xval-lr1e4'],
    ['s48', 'xval-lr1e4', 's45'],
    ['s48', 'xval-lr1e4', 's45', 's52'],
    ['s48', 's52', 'xval-lr1e4'],
    ['s48', 'xval-lr1e4', 's45', 's52', 's50'],
    ['s48', 'xval-lr1e4', 's51'],
]


def seed_dir(s):
    n = int(s[1:])
    return os.path.join(R16 if 41 <= n <= 46 else R17, f'seed{n}')


def load_val_labels():
    from ml.dataset import load_and_create_sequences
    from sklearn.model_selection import GroupShuffleSplit
    X, y, pid = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=SERVING_FEATURES,
        window_minutes=90, stride=15, label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, v_idx = next(gss.split(X, y, groups=pid))
    return y[v_idx]


def load_holdout_labels():
    from ml.dataset import load_and_create_sequences
    _, y, _ = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-b_full', 'set-b'),
        outcomes_file=os.path.join(BASE, 'Outcomes-b.txt'),
        vital_features=SERVING_FEATURES,
        window_minutes=90, stride=15, label_mode='proximity', horizon_hours=12)
    return y


def fresh_logits(member, X, device):
    from ml.train_lstm import AttentionLSTMModel, SimpleLSTMDataset
    from torch.utils.data import DataLoader
    m = AttentionLSTMModel(input_size=12, hidden_size=96, dropout=0.3)
    m.load_state_dict(torch.load(os.path.join(XVAL, XVAL_MAP[member], 'model.pt'), map_location='cpu'))
    m.eval().to(device)
    dl = DataLoader(SimpleLSTMDataset(X, np.zeros(len(X))), batch_size=512, shuffle=False)
    pr = np.zeros(len(X)); off = 0
    with torch.no_grad():
        for xb, _ in dl:
            b = xb.shape[0]
            pr[off:off+b] = m(xb.to(device)).cpu().numpy().ravel(); off += b
    return pr


def normed_arrays(which):
    from ml.dataset import load_and_create_sequences
    s = json.load(open('ml/scaler.json'))
    mean = np.array(s['mean']); std = np.array(s['std'])
    feats = list(SERVING_FEATURES)
    if which == 'va':
        X, y, pid = load_and_create_sequences(
            physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
            vital_features=feats, window_minutes=90, stride=15,
            label_mode='proximity', horizon_hours=12)
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, v_idx = next(gss.split(X, y, groups=pid))
        X = X[v_idx]
    else:
        X, _, _ = load_and_create_sequences(
            physionet_dir=os.path.join(BASE, 'set-b_full', 'set-b'),
            outcomes_file=os.path.join(BASE, 'Outcomes-b.txt'),
            vital_features=feats, window_minutes=90, stride=15,
            label_mode='proximity', horizon_hours=12)
    return ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32)


def metrics(p, y):
    pr = 1 / (1 + np.exp(-p))
    return (float(roc_auc_score(y, pr)), float(accuracy_score(y, pr > 0.5)),
            float(recall_score(y, pr > 0.5)))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yva = load_val_labels()
    yho = load_holdout_labels()
    Xva = normed_arrays('va')
    Xho = normed_arrays('ho')
    P = {}
    PH = {}
    for s in [f's{n}' for n in list(range(41, 53))]:
        P[s] = np.load(os.path.join(seed_dir(s), 'p_va.npy'))
        PH[s] = np.load(os.path.join(seed_dir(s), 'p_ho.npy'))
    for m in XVAL_MAP:
        P[m] = fresh_logits(m, Xva, device)
        PH[m] = fresh_logits(m, Xho, device)
    print('members ready', flush=True)
    for combo in CANDIDATES:
        pe = np.mean([P[k] for k in combo], axis=0)
        phe = np.mean([PH[k] for k in combo], axis=0)
        a, acc, rec = metrics(pe, yva)
        ha, hacc, hrec = metrics(phe, yho)
        print(f'{combo}: val={a:.4f} acc={acc:.4f} rec={rec:.4f} | holdout={ha:.4f} acc={hacc:.4f} rec={hrec:.4f}',
              flush=True)


if __name__ == '__main__':
    main()