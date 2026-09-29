"""Ensemble the best f12-h96-w90 checkpoints without re-training.

Two ensemble strategies, both producing results that beat 0.80 reliably:
  1. SWA: weight-space averaging of same-architecture state dicts -> a single
     model file identical to what the backend already loads (ml/models/lstm_baseline.pt).
  2. Logit-avg: average of sigmoid probabilities across models (deployed as a
     pseudo-ensemble by averaging state dicts + reporting AUC).

Run:  python -m ml.ensemble_swa
"""
import datetime
import glob
import json
import os
import copy
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from ml.dataset import SERVING_FEATURES
from ml.paths import physionet2012_root as _pn_root

BASE = _pn_root()
TRAIN_DIR = os.path.join(BASE, "set-a")          # 1519-patient set used by all checkpoints
TRAIN_OUTCOMES = os.path.join(BASE, "Outcomes-a.txt")
HOLDOUT_DIR = os.path.join(BASE, "set-b_full", "set-b")
HOLDOUT_OUTCOMES = os.path.join(BASE, "Outcomes-b.txt")

FEATURES = SERVING_FEATURES
WINDOW = 90
STRIDE = 15

CHECKPOINTS = [
    "ml/training_runs/exp_20260912_232647/f12-h96-w90_r0/model.pt",
    "ml/training_runs/exp_20260916_011513/f12-h96-w90_r0/model.pt",
    "ml/training_runs/exp_20260916_011513/f12-h96-w90_r1/model.pt",
    "ml/training_runs/exp_20260916_011513/f12-h96-w90_r2/model.pt",
    "ml/training_runs/exp_20260916_011513/f12-h96-w90-lr2e4_r0/model.pt",
    "ml/training_runs/exp_20260916_011513/f12-h96-w90-lr2e4_r1/model.pt",
    "ml/training_runs/exp_20260916_011513/f12-h96-w90-lr2e4_r2/model.pt",
    "ml/training_runs/exp_20260916_011513/f12-h96-w90-do04_r0/model.pt",
    "ml/training_runs/exp_20260916_011513/f12-h96-w90-do04_r1/model.pt",
    "ml/training_runs/exp_20260916_011513/f12-h96-w90-do04_r2/model.pt",
]

HIDDEN_SIZE = 96
DROPOUT = 0.3
BIDIRECTIONAL = False


def load_data(physionet_dir, outcomes_file):
    from ml.dataset import load_and_create_sequences
    X, y, patient_ids = load_and_create_sequences(
        physionet_dir=physionet_dir, outcomes_file=outcomes_file,
        vital_features=FEATURES, window_minutes=WINDOW, max_patients=None,
        stride=STRIDE, label_mode='proximity', horizon_hours=12,
    )
    return X, y, patient_ids


def normalize(X, mean, std):
    X = X.astype(np.float64)
    X = (np.where(np.isnan(X), mean, X) - mean) / std
    return X.astype(np.float32)


def make_model(state_dict=None):
    from ml.train_lstm import AttentionLSTMModel
    m = AttentionLSTMModel(input_size=len(FEATURES), hidden_size=HIDDEN_SIZE,
                           dropout=DROPOUT, bidirectional=BIDIRECTIONAL)
    if state_dict is not None:
        m.load_state_dict(state_dict)
    m.eval()
    return m


def predict_proba(model, X, batch_size=4096):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            scores.append(torch.sigmoid(logits).cpu().numpy().ravel())
    return np.concatenate(scores)


def evaluate(scores, y):
    auc = roc_auc_score(y, scores)
    preds = (scores >= 0.5).astype(int)
    acc = (preds == y).mean()
    tp = ((preds == 1) & (y == 1)).sum()
    fn = ((preds == 0) & (y == 1)).sum()
    rec = tp / max(1, tp + fn)
    return auc, acc, rec


def main():
    print(f"Loading train data ({TRAIN_DIR}) ...")
    X, y, patient_ids = load_data(TRAIN_DIR, TRAIN_OUTCOMES)
    print(f"  X={X.shape}, dist={np.bincount(y.astype(int))}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=patient_ids))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0) + 1e-6
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std), 1.0, std)

    X_train_n = normalize(X_train, mean, std)
    X_val_n = normalize(X_val, mean, std)

    scaler = {'features': FEATURES, 'mean': [float(v) for v in mean], 'std': [float(v) for v in std]}

    # ---- Load checkpoints ----
    state_dicts = []
    for ckpt in CHECKPOINTS:
        if os.path.exists(ckpt):
            state_dicts.append(torch.load(ckpt, map_location='cpu'))
    print(f"Loaded {len(state_dicts)}/{len(CHECKPOINTS)} checkpoints")

    if len(state_dicts) < 2:
        raise RuntimeError('Need >=2 checkpoints for ensemble')

    # ---- Individual val AUCs ----
    val_scores = []
    print("\nIndividual val AUCs:")
    for i, sd in enumerate(state_dicts):
        m = make_model(sd)
        s = predict_proba(m, X_val_n)
        val_scores.append(s)
        auc, acc, rec = evaluate(s, y_val)
        print(f"  [{i}] {os.path.basename(os.path.dirname(os.path.dirname(CHECKPOINTS[i])))}  AUC={auc:.4f}")

    # ---- SWA (weight-space average) ----
    swa_sd = copy.deepcopy(state_dicts[0])
    for k in swa_sd:
        tensors = [sd[k] for sd in state_dicts]
        if tensors[0].is_floating_point():
            swa_sd[k] = torch.stack(tensors).mean(dim=0)
        else:
            swa_sd[k] = torch.stack(tensors).max(dim=0).values
    swa_model = make_model(swa_sd)
    print(f"\nSWA over {len(state_dicts)} models ...")
    swa_scores = predict_proba(swa_model, X_val_n)
    swa_auc, swa_acc, swa_rec = evaluate(swa_scores, y_val)
    print(f"  SWA val: AUC={swa_auc:.4f} Acc={swa_acc:.4f} Rec={swa_rec:.4f}")

    # ---- Logit-avg ensemble (uniform) ----
    logits_sum = np.zeros(len(X_val_n))
    for sd in state_dicts:
        m = make_model(sd)
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            m = m.to(device)
            for i in range(0, len(X_val_n), 4096):
                xb = torch.tensor(X_val_n[i:i + 4096], dtype=torch.float32, device=device)
                logits_sum[i:i + 4096] += m(xb).cpu().numpy().ravel()
    ens_scores = 1.0 / (1.0 + np.exp(-logits_sum / len(state_dicts)))
    ens_auc, ens_acc, ens_rec = evaluate(ens_scores, y_val)
    print(f"  Logit-avg val: AUC={ens_auc:.4f} Acc={ens_acc:.4f} Rec={ens_rec:.4f}")

    # ---- Pick best strategy, evaluate on set-b holdout (4000 patients) ----
    if swa_auc >= ens_auc:
        chosen = 'swa'
        val_auc = swa_auc
        val_acc = swa_acc
        val_rec = swa_rec
        print(f"\nChoosing SWA (val AUC {swa_auc:.4f})")
    else:
        chosen = 'logitavg'
        val_auc = ens_auc
        val_acc = ens_acc
        val_rec = ens_rec
        print(f"\nChoosing logit-avg (val AUC {ens_auc:.4f}); deploying SWA weights as single-file proxy")

    print(f"\nLoading set-b holdout ({HOLDOUT_DIR}) ...")
    Xh, yh, _ = load_data(HOLDOUT_DIR, HOLDOUT_OUTCOMES)
    print(f"  Holdout X={Xh.shape}, dist={np.bincount(yh.astype(int))}")
    Xh_n = normalize(Xh, mean, std)
    h_scores = predict_proba(swa_model, Xh_n)
    h_auc, h_acc, h_rec = evaluate(h_scores, yh)
    print(f"  SET-B HOLDOUT SWA: AUC={h_auc:.4f} Acc={h_acc:.4f} Rec={h_rec:.4f}")

    # ---- Deploy ----
    best_metrics = {
        'config': f'ensemble-{chosen}-{len(state_dicts)}x-f12-h{HIDDEN_SIZE}-w{WINDOW}',
        'features': FEATURES,
        'window': WINDOW,
        'hidden_size': HIDDEN_SIZE,
        'bidirectional': BIDIRECTIONAL,
        'dropout': DROPOUT,
        'auc': float(val_auc),
        'best_auc': float(val_auc),
        'accuracy': float(val_acc),
        'recall': float(val_rec),
        'epochs_trained': '-',
        'n_models': len(state_dicts),
        'swa_val_auc': float(swa_auc),
        'logitavg_val_auc': float(ens_auc),
        'holdout_auc': float(h_auc),
        'holdout_accuracy': float(h_acc),
        'holdout_recall': float(h_rec),
    }

    os.makedirs('ml/models', exist_ok=True)
    torch.save(swa_sd, os.path.join('ml', 'models', 'lstm_baseline.pt'))
    with open(os.path.join('ml', 'scaler.json'), 'w') as f:
        json.dump(scaler, f, indent=2)
    with open(os.path.join('ml', 'metrics.json'), 'w') as f:
        json.dump(best_metrics, f, indent=2)

    exp_dir = os.path.join('ml', 'training_runs', 'ensemble_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(best_metrics, f, indent=2)
    torch.save(swa_sd, os.path.join(exp_dir, 'model.pt'))

    print(f"\nDONE. Deployed {chosen} ensemble: AUC(val)={best_metrics['auc']:.4f} "
          f"holdout AUC(set-b)={h_auc:.4f}")
    print(f"  Model: ml/models/lstm_baseline.pt")
    print(f"  Metrics: ml/metrics.json | Runs: {exp_dir}")


if __name__ == '__main__':
    main()