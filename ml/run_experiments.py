"""Iterative grid search over model hyperparameters.

Loads PhysioNet data ONCE, then loops through configs training from scratch
each time. Overall-best model is deployed to ml/models/lstm_baseline.pt.

Run:  python -m ml.run_experiments
"""
import datetime
import json
import os
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from ml.dataset import SERVING_FEATURES
from ml.paths import physionet2012_root as _pn_root

BASE_DIR = _pn_root()
PHYSIONET_DIR = os.path.join(BASE_DIR, "set-a_full", "set-a")
OUTCOMES_FILE = os.path.join(BASE_DIR, "Outcomes-a.txt")
HOLDOUT_DIR = os.path.join(BASE_DIR, "set-b_full", "set-b")
HOLDOUT_OUTCOMES = os.path.join(BASE_DIR, "Outcomes-b.txt")
VITAL_FEATURES_6 = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2']
VITAL_FEATURES_12 = SERVING_FEATURES
VITAL_FEATURES_16 = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
                      'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose',
                      'Lactate', 'pH', 'FiO2', 'MechVent']
WINDOW = 60

EPOCHS = 40
PATIENCE = 12
MIN_DELTA = 0.001

# ── Config grid ──────────────────────────────────────────────────────────────
# Round 7: full set-a (4000 patients) train + set-b (4000 patients) holdout eval.
# More data -> bigger models should win. Focus on 16 features (paper's vent params).
CONFIGS = [
    # Previous best zone rerun on full data
    ("f12-h96-w90", dict(hidden_size=96, batch_size=128, lr=3e-4, stride=15, dropout=0.3, features=VITAL_FEATURES_12, window=90)),
    ("f12-h128-w90", dict(hidden_size=128, batch_size=128, lr=3e-4, stride=15, dropout=0.3, features=VITAL_FEATURES_12, window=90)),
    ("f12-h192-w90", dict(hidden_size=192, batch_size=128, lr=3e-4, stride=15, dropout=0.3, features=VITAL_FEATURES_12, window=90)),
    # 16 features (ventilator params) — paper's approach
    ("f16-h96-w90", dict(hidden_size=96, batch_size=128, lr=3e-4, stride=15, dropout=0.3, features=VITAL_FEATURES_16, window=90)),
    ("f16-h128-w90", dict(hidden_size=128, batch_size=128, lr=3e-4, stride=15, dropout=0.3, features=VITAL_FEATURES_16, window=90)),
    ("f16-h128-w90-do04", dict(hidden_size=128, batch_size=128, lr=3e-4, stride=15, dropout=0.4, features=VITAL_FEATURES_16, window=90)),
    ("f16-h128-w90-lr2e4", dict(hidden_size=128, batch_size=128, lr=2e-4, stride=15, dropout=0.3, features=VITAL_FEATURES_16, window=90)),
    ("f16-h128-w90-bi", dict(hidden_size=128, batch_size=128, lr=3e-4, stride=15, dropout=0.3, features=VITAL_FEATURES_16, window=90, bidirectional=True)),
    ("f16-h192-w90", dict(hidden_size=192, batch_size=128, lr=3e-4, stride=15, dropout=0.3, features=VITAL_FEATURES_16, window=90)),
    ("f16-h192-w90-do04", dict(hidden_size=192, batch_size=128, lr=3e-4, stride=15, dropout=0.4, features=VITAL_FEATURES_16, window=90)),
    # Longer window
    ("f16-h128-w120", dict(hidden_size=128, batch_size=128, lr=3e-4, stride=15, dropout=0.3, features=VITAL_FEATURES_16, window=120)),
    ("f12-h128-w120", dict(hidden_size=128, batch_size=128, lr=3e-4, stride=15, dropout=0.3, features=VITAL_FEATURES_12, window=120)),
]

REPEATS = 1


def load_and_split(vital_features, window):
    """Load data with given features/window, compute normalization from train split."""
    from ml.dataset import load_and_create_sequences

    print(f"Loading set-a data: {len(vital_features)} features, window={window}min ...")
    X, y, patient_ids = load_and_create_sequences(
        physionet_dir=PHYSIONET_DIR,
        outcomes_file=OUTCOMES_FILE,
        vital_features=vital_features,
        window_minutes=window,
        max_patients=None,
        stride=15,
        label_mode='proximity',
        horizon_hours=12,
    )
    print(f"  X={X.shape} y={y.shape}, dist={np.bincount(y.astype(int))}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=patient_ids))
    X_train, y_train = X[train_idx].astype(np.float64), y[train_idx]
    X_val, y_val = X[val_idx].astype(np.float64), y[val_idx]
    print(f"  Split: train={len(X_train)} val={len(X_val)} (patient-level)")

    flat = X_train.reshape(-1, X_train.shape[-1])
    train_mean = np.nanmean(flat, axis=0)
    train_std = np.nanstd(flat, axis=0) + 1e-6
    train_mean = np.where(np.isnan(train_mean), 0.0, train_mean)
    train_std = np.where(np.isnan(train_std), 1.0, train_std)

    X_train = (np.where(np.isnan(X_train), train_mean, X_train) - train_mean) / train_std
    X_val = (np.where(np.isnan(X_val), train_mean, X_val) - train_mean) / train_std
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    scaler = {'features': list(vital_features),
              'mean': [float(v) for v in train_mean],
              'std': [float(v) for v in train_std]}
    return X_train, y_train, X_val, y_val, scaler


HOLDOUT_CACHE = {}


def load_holdout(vital_features, window, scaler):
    """Load the set-b holdout (4000 patients) with matching features/window."""
    key = (tuple(vital_features), window)
    if key in HOLDOUT_CACHE:
        return HOLDOUT_CACHE[key]

    from ml.dataset import load_and_create_sequences

    print(f"Loading set-b holdout: {len(vital_features)} features, window={window}min ...")
    X, y, patient_ids = load_and_create_sequences(
        physionet_dir=HOLDOUT_DIR,
        outcomes_file=HOLDOUT_OUTCOMES,
        vital_features=vital_features,
        window_minutes=window,
        max_patients=None,
        stride=15,
        label_mode='proximity',
        horizon_hours=12,
    )
    print(f"  Holdout X={X.shape} y={y.shape}, dist={np.bincount(y.astype(int))}")

    X = X.astype(np.float64)
    mean = np.array(scaler['mean'])
    std = np.array(scaler['std'])
    X = (np.where(np.isnan(X), mean, X) - mean) / std
    X = X.astype(np.float32)
    HOLDOUT_CACHE[key] = (X, y, patient_ids)
    return HOLDOUT_CACHE[key]


def train_single(X_train, y_train, X_val, y_val, cfg_name, cfg_kwargs, run_dir, features, window):
    """Train one config, return metrics dict."""
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    from ml.train import evaluate_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = n_neg / max(1, n_pos)

    hidden_size = cfg_kwargs.pop('hidden_size', 64)
    bidirectional = cfg_kwargs.pop('bidirectional', False)
    batch_size = cfg_kwargs.pop('batch_size', 128)
    lr = cfg_kwargs.pop('lr', 3e-4)
    dropout = cfg_kwargs.pop('dropout', 0.3)
    weight_decay = cfg_kwargs.pop('weight_decay', 0.0)
    stride = cfg_kwargs.pop('stride', 15)

    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    model = None
    opt = None

    for epoch in range(EPOCHS):
        model, opt = quick_train(
            X_train, y_train,
            epochs=1, batch_size=batch_size, learning_rate=lr,
            pos_weight=pos_weight, model_class=AttentionLSTMModel,
            model=model, optimizer=opt,
            dropout=dropout, hidden_size=hidden_size,
            bidirectional=bidirectional, weight_decay=weight_decay,
        )

        metrics = evaluate_model(model, X_val, y_val)
        auc = metrics['auc']

        if auc - best_auc > MIN_DELTA:
            best_auc = auc
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final = evaluate_model(model, X_val, y_val)
    final['best_auc'] = best_auc
    final['epochs_trained'] = epoch + 1
    final['config'] = cfg_name
    final['features'] = list(features)
    final['window'] = window
    final['hidden_size'] = hidden_size
    final['bidirectional'] = bool(bidirectional)
    final['batch_size'] = batch_size
    final['lr'] = lr
    final['dropout'] = dropout

    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(final, f, indent=2)

    return final


def main():
    run_base = os.path.join('ml', 'training_runs', 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_base, exist_ok=True)

    overall_best = None
    all_results = []

    # Group configs by (features, window) to avoid reloading data
    from collections import OrderedDict
    data_groups = OrderedDict()
    for cfg_name, cfg_kwargs in CONFIGS:
        feats = tuple(cfg_kwargs.pop('features', VITAL_FEATURES_6))
        win = cfg_kwargs.pop('window', WINDOW)
        key = (feats, win)
        if key not in data_groups:
            data_groups[key] = []
        data_groups[key].append((cfg_name, cfg_kwargs))

    total_runs = sum(len(v) for v in data_groups.values()) * REPEATS
    run_idx = 0

    for (feats, win), configs in data_groups.items():
        X_train, y_train, X_val, y_val, scaler = load_and_split(list(feats), win)
        n_feat = len(feats)

        for cfg_name, cfg_kwargs in configs:
            aucs = []
            for rep in range(REPEATS):
                run_idx += 1
                rep_name = f"{cfg_name}_r{rep}"
                print(f"\n{'='*60}")
                print(f"  [{run_idx}/{total_runs}] Config: {rep_name}  (features={n_feat}, window={win})")
                print(f"{'='*60}")
                run_dir = os.path.join(run_base, rep_name)

                metrics = train_single(X_train, y_train, X_val, y_val, rep_name, dict(cfg_kwargs), run_dir, list(feats), win)
                all_results.append((rep_name, metrics['auc'], metrics))

                auc = metrics['auc']
                aucs.append(auc)
                print(f"  => AUC={auc:.4f}  Acc={metrics['accuracy']:.4f}  Rec={metrics['recall']:.4f}  Epochs={metrics['epochs_trained']}")

                if overall_best is None or auc > overall_best['auc']:
                    overall_best = dict(metrics)
                    overall_best['run_dir'] = run_dir
                    overall_best['scaler'] = scaler
                    overall_best['features'] = list(feats)
                    print(f"  *** NEW BEST ***")

            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            print(f"\n  {cfg_name}: mean AUC={mean_auc:.4f} +/- {std_auc:.4f}  (runs: {[f'{a:.4f}' for a in aucs]})")

    # Summary
    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY (grouped by config, sorted by mean AUC)")
    print(f"{'='*60}")
    config_stats = {}
    for rep_name, auc, metrics in all_results:
        base_name = rep_name.rsplit('_r', 1)[0]
        if base_name not in config_stats:
            config_stats[base_name] = []
        config_stats[base_name].append((auc, metrics))

    summary = []
    for name, runs in config_stats.items():
        aucs = [a for a, _ in runs]
        best_m = max(runs, key=lambda x: x[0])[1]
        summary.append((name, np.mean(aucs), np.std(aucs), max(aucs), best_m))
    summary.sort(key=lambda x: x[1], reverse=True)

    for i, (name, mean, std, best, m) in enumerate(summary):
        marker = " <-- BEST" if overall_best and name in overall_best.get('config', '') else ""
        print(f"  {i+1}. {name:35s}  mean={mean:.4f}+/-{std:.4f}  best={best:.4f}  Rec={m['recall']:.4f}{marker}")

    # Deploy overall best
    import shutil
    best_model_src = os.path.join(overall_best['run_dir'], 'model.pt')
    best_model_dst = os.path.join('ml', 'models', 'lstm_baseline.pt')
    os.makedirs(os.path.dirname(best_model_dst), exist_ok=True)
    shutil.copy2(best_model_src, best_model_dst)

    scaler_path = os.path.join('ml', 'scaler.json')
    with open(scaler_path, 'w') as f:
        json.dump(overall_best['scaler'], f, indent=2)

    deploy_metrics = {k: v for k, v in overall_best.items() if k not in ('run_dir', 'scaler')}

    # Evaluate deployed best on set-b holdout (4000 patients, unseen set)
    from ml.train_lstm import AttentionLSTMModel
    from ml.train import evaluate_model

    holdout_X, holdout_y, _ = load_holdout(overall_best['features'], overall_best['window'], overall_best['scaler'])
    model = AttentionLSTMModel(input_size=len(overall_best['features']),
                               hidden_size=overall_best.get('hidden_size', 96),
                               dropout=overall_best.get('dropout', 0.3),
                               bidirectional=overall_best.get('bidirectional', False))
    model.load_state_dict(torch.load(best_model_src))
    model.eval()
    holdout_metrics = evaluate_model(model, holdout_X, holdout_y)
    holdout_metrics = {f'holdout_{k}': v for k, v in holdout_metrics.items()}
    print(f"\n  SET-B HOLDOUT (4000 patients): AUC={holdout_metrics['holdout_auc']:.4f} "
          f"Acc={holdout_metrics['holdout_accuracy']:.4f} Rec={holdout_metrics['holdout_recall']:.4f}")
    deploy_metrics.update(holdout_metrics)

    with open(os.path.join('ml', 'metrics.json'), 'w') as f:
        json.dump(deploy_metrics, f, indent=2)

    print(f"\n  DEPLOYED: {overall_best['config']}  AUC={overall_best['auc']:.4f}")
    print(f"  Features: {overall_best['features']}")
    print(f"  Model: {best_model_dst}")
    print(f"  Metrics: ml/metrics.json")
    print(f"  All runs: {run_base}")

    return overall_best


if __name__ == '__main__':
    main()
