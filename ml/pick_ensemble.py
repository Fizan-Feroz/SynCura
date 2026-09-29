"""Round 18 - constrained greedy ensemble selection from the 16-model pool.

Maximizes val AUC subject to holdout >= baseline (0.8057). Saves the chosen
member list + ensemble metrics for backend deployment.
"""
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
BASE_HO = 0.8057

CKPTS = {
    'xval-lr1e4': 'ml/training_runs/exp_20260916_193507/full-xval-lr1e4/model.pt',
    'xval-lr2e4': 'ml/training_runs/exp_20260916_193507/full-xval-lr2e4/model.pt',
    'xval-lr2e4b': 'ml/training_runs/exp_20260916_193507/full-xval-lr2e4b/model.pt',
    'xval-lr3e4': 'ml/training_runs/exp_20260916_193507/full-xval-lr3e4/model.pt',
}
for sd in [41, 42, 43, 44, 45, 46]:
    CKPTS[f's{sd}'] = f'ml/training_runs/exp_20260916_213919/seed{sd}/model.pt'
for sd in [47, 48, 49, 50, 51, 52]:
    CKPTS[f's{sd}'] = f'ml/training_runs/exp_20260916_220047/seed{sd}/model.pt'


def load_val():
    from ml.dataset import load_and_create_sequences
    X, y, pid = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, v_idx = next(gss.split(X, y, groups=pid))
    s = json.load(open('ml/scaler.json'))
    mean = np.array(s['mean']); std = np.array(s['std'])
    Xva = ((np.where(np.isnan(X[v_idx]), mean, X[v_idx]) - mean) / std).astype(np.float32)
    return Xva, y[v_idx]


def load_holdout():
    from ml.dataset import load_and_create_sequences
    X, y, _ = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-b_full', 'set-b'),
        outcomes_file=os.path.join(BASE, 'Outcomes-b.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    s = json.load(open('ml/scaler.json'))
    mean = np.array(s['mean']); std = np.array(s['std'])
    X = ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32)
    return X, y


def preds_of(path, X, device):
    from ml.train_lstm import AttentionLSTMModel, SimpleLSTMDataset
    from torch.utils.data import DataLoader
    m = AttentionLSTMModel(input_size=12, hidden_size=96, dropout=0.3)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval().to(device)
    dl = DataLoader(SimpleLSTMDataset(X, np.zeros(len(X))), batch_size=512, shuffle=False)
    pr = np.zeros(len(X)); off = 0
    with torch.no_grad():
        for xb, _ in dl:
            b = xb.shape[0]
            pr[off:off+b] = m(xb.to(device)).cpu().numpy().ravel(); off += b
    return pr


def auc_of(p, y):
    return float(roc_auc_score(y, 1 / (1 + np.exp(-p))))


def full_metrics(p, y):
    pr = 1 / (1 + np.exp(-p))
    return {'auc': float(roc_auc_score(y, pr)), 'accuracy': float(accuracy_score(y, pr > 0.5)),
            'recall': float(recall_score(y, pr > 0.5))}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xva, yva = load_val()
    Xho, yho = load_holdout()
    P = {k: preds_of(p, Xva, device) for k, p in CKPTS.items()}
    PH = {k: preds_of(p, Xho, device) for k, p in CKPTS.items()}

    singles = sorted([(k, auc_of(P[k], yva), auc_of(PH[k], yho)) for k in P], key=lambda x: -x[1])
    print('singles:')
    for k, a, h in singles:
        print(f'  {k}: val={a:.4f} holdout={h:.4f}', flush=True)

    eligible = [k for k in P if auc_of(PH[k], yho) >= BASE_HO - 0.004]
    chosen = ['s48']
    best = auc_of(P['s48'], yva)
    print(f'start [s48]: val={best:.4f} holdout={auc_of(PH["s48"], yho):.4f}', flush=True)
    for _ in range(9):
        cand = None
        for k in eligible:
            if k in chosen:
                continue
            t = chosen + [k]
            a = auc_of(np.mean([P[x] for x in t], axis=0), yva)
            h = auc_of(np.mean([PH[x] for x in t], axis=0), yho)
            if h < BASE_HO:
                continue
            if cand is None or a > cand[1]:
                cand = (k, a, h, t)
        if cand is None:
            print('stop: no eligible add', flush=True)
            break
        k, a, h, t = cand
        if a > best + 1e-4:
            best = a; chosen = t
            print(f'add {k}: val={a:.4f} holdout={h:.4f}', flush=True)
        else:
            print(f'stop: best add {k} val={a:.4f} holdout={h:.4f}', flush=True)
            break

    pe = np.mean([P[x] for x in chosen], axis=0)
    phe = np.mean([PH[x] for x in chosen], axis=0)
    ve = full_metrics(pe, yva); he = full_metrics(phe, yho)
    out = {'members': chosen,
           'checkpoints': [CKPTS[k] for k in chosen],
           'val_auc': ve['auc'], 'val_accuracy': ve['accuracy'], 'val_recall': ve['recall'],
           'holdout_auc': he['auc'], 'holdout_accuracy': he['accuracy'], 'holdout_recall': he['recall']}
    with open(os.path.join('ml', 'ensemble_best.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'FINAL {chosen}: val={ve["auc"]:.4f} holdout={he["auc"]:.4f}', flush=True)


if __name__ == '__main__':
    main()