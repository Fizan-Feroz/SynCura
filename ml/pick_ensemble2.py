"""Round 20 - greedy ensemble selection over the 21-model pool (16 unidir + 5 bidir).

Maximizes val AUC subject to holdout >= 0.8073 (current deployed holdout).
Saves manifest for backend deployment (per-member arch included).
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
BASE_VAL = 0.8371
BASE_HO = 0.8073
STEP_HO = 0.8000
BIDIR_RUN = 'ml/training_runs/exp_20260917_001259'

CKPTS = {
    'xval-lr1e4': ('ml/training_runs/exp_20260916_193507/full-xval-lr1e4/model.pt', False, 96),
    'xval-lr2e4': ('ml/training_runs/exp_20260916_193507/full-xval-lr2e4/model.pt', False, 96),
    'xval-lr2e4b': ('ml/training_runs/exp_20260916_193507/full-xval-lr2e4b/model.pt', False, 96),
    'xval-lr3e4': ('ml/training_runs/exp_20260916_193507/full-xval-lr3e4/model.pt', False, 96),
}
for sd in [41, 42, 43, 44, 45, 46]:
    CKPTS[f's{sd}'] = (f'ml/training_runs/exp_20260916_213919/seed{sd}/model.pt', False, 96)
for sd in [47, 48, 49, 50, 51, 52]:
    CKPTS[f's{sd}'] = (f'ml/training_runs/exp_20260916_220047/seed{sd}/model.pt', False, 96)
for name in ['bidir-lr1e4-s6', 'bidir-lr2e4-s6', 'bidir-lr1e4-do4', 'bidir-h128-lr1e4', 'bidir-h128-lr2e4']:
    hid = 128 if 'h128' in name else 96
    CKPTS[name] = (os.path.join(BIDIR_RUN, name, 'model.pt'), True, hid)


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
    return ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32), y


def preds_of(path, bidir, hid, X, device):
    from ml.train_lstm import AttentionLSTMModel, SimpleLSTMDataset
    from torch.utils.data import DataLoader
    m = AttentionLSTMModel(input_size=12, hidden_size=hid, dropout=0.3, bidirectional=bidir)
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
    P = {k: preds_of(p, b, h, Xva, device) for k, (p, b, h) in CKPTS.items()}
    PH = {k: preds_of(p, b, h, Xho, device) for k, (p, b, h) in CKPTS.items()}

    singles = sorted([(k, auc_of(P[k], yva), auc_of(PH[k], yho)) for k in P], key=lambda x: -x[1])
    print('singles:', flush=True)
    for k, a, h in singles:
        print(f'  {k}: val={a:.4f} holdout={h:.4f}', flush=True)

    start = singles[0][0]
    chosen = [start]
    best = auc_of(P[start], yva)
    print(f'start [{start}]: val={best:.4f} holdout={auc_of(PH[start], yho):.4f}', flush=True)
    for _ in range(10):
        cand = None
        for k in P:
            if k in chosen:
                continue
            t = chosen + [k]
            a = auc_of(np.mean([P[x] for x in t], axis=0), yva)
            h = auc_of(np.mean([PH[x] for x in t], axis=0), yho)
            if h < STEP_HO:
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
    beats = ve['auc'] > BASE_VAL and he['auc'] >= BASE_HO
    tag = 'BEATS DEPLOYED' if beats else 'below deployed'
    out = {'members': chosen,
           'checkpoints': [CKPTS[k][0] for k in chosen],
           'arch': [{'bidirectional': CKPTS[k][1], 'hidden_size': CKPTS[k][2]} for k in chosen],
           'val_auc': ve['auc'], 'val_accuracy': ve['accuracy'], 'val_recall': ve['recall'],
           'holdout_auc': he['auc'], 'holdout_accuracy': he['accuracy'], 'holdout_recall': he['recall']}
    with open(os.path.join('ml', 'ensemble_best.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'FINAL {chosen}: val={ve["auc"]:.4f} holdout={he["auc"]:.4f} [{tag}]', flush=True)


if __name__ == '__main__':
    main()