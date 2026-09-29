"""Round 9 — push past 0.80 by fine-tuning the 0.7997 winner with low LR.

Warm-start from exp_20260916_050232/f12-h96-sched-step (AUC 0.7997), continue
training with small LR and long patience so val AUC keeps climbing.
"""
import datetime
import json
import os
import copy
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from ml.paths import physionet2012_root as _pn_root

BASE = _pn_root()
TRAIN_DIR = os.path.join(BASE, "set-a")
TRAIN_OUTCOMES = os.path.join(BASE, "Outcomes-a.txt")

from ml.dataset import SERVING_FEATURES

FEATURES_12 = SERVING_FEATURES

EPOCHS = 50
PATIENCE = 18
MIN_DELTA = 0.0003

WARM = "ml/training_runs/exp_20260916_050232/f12-h96-sched-step/model.pt"

CONFIGS = [
    ("win7ft-cos-5e5",  dict(hidden_size=96, lr=5e-5, dropout=0.3, weight_decay=1e-5, batch_size=128, schedule='cos')),
    ("win7ft-cos-7e5",  dict(hidden_size=96, lr=7e-5, dropout=0.3, weight_decay=1e-5, batch_size=128, schedule='cos')),
    ("win7ft-cos-1e4",  dict(hidden_size=96, lr=1e-4, dropout=0.3, weight_decay=1e-5, batch_size=128, schedule='cos')),
    ("win7ft-plat",     dict(hidden_size=96, lr=1e-4, dropout=0.3, weight_decay=1e-5, batch_size=128, schedule='plat')),
    ("win7ft-cos-64",   dict(hidden_size=96, lr=7e-5, dropout=0.3, weight_decay=1e-5, batch_size=64,  schedule='cos')),
]

def load_split():
    from ml.dataset import load_and_create_sequences
    X, y, patient_ids = load_and_create_sequences(
        physionet_dir=TRAIN_DIR, outcomes_file=TRAIN_OUTCOMES,
        vital_features=FEATURES_12, window_minutes=90, max_patients=None,
        stride=15, label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx, v_idx = next(gss.split(X, y, groups=patient_ids))
    X_tr, y_tr = X[idx], y[idx]
    X_va, y_va = X[v_idx], y[v_idx]
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0) + 1e-6
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std), 1.0, std)
    X_tr = ((np.where(np.isnan(X_tr), mean, X_tr) - mean) / std).astype(np.float32)
    X_va = ((np.where(np.isnan(X_va), mean, X_va) - mean) / std).astype(np.float32)
    return X_tr, y_tr, X_va, y_va, {'features': FEATURES_12, 'mean': mean.tolist(), 'std': std.tolist()}


def train_one(g, X_tr, y_tr, X_va, y_va, out_dir):
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    from ml.train import evaluate_model

    name, cfg = g
    hid = cfg['hidden_size']; lr = cfg['lr']; do = cfg['dropout']
    wd = cfg['weight_decay']; bs = cfg['batch_size']; sched = cfg['schedule']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pw = n_neg / max(1, n_pos)

    print(f"  [{name}] warm-start {WARM} lr={lr} wd={wd} bs={bs} sched={sched}")
    try:
        warm_sd = torch.load(WARM, map_location='cpu', weights_only=True)
    except TypeError:  # old torch without weights_only
        warm_sd = torch.load(WARM, map_location='cpu')
    model = AttentionLSTMModel(input_size=len(FEATURES_12), hidden_size=hid, dropout=do).to(device)
    model.load_state_dict(warm_sd)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_auc = 0.0; best_state = None; patience = 0
    lr_plateau = None
    if sched == 'plat':
        lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    for epoch in range(EPOCHS):
        model, opt = quick_train(
            X_tr, y_tr, epochs=1, batch_size=bs, learning_rate=lr,
            pos_weight=pw, model_class=AttentionLSTMModel, model=model, optimizer=opt,
            dropout=do, hidden_size=hid, weight_decay=wd)
        if sched == 'cos':
            for pg in opt.param_groups:
                progress = (epoch + 1) / EPOCHS
                pg['lr'] = lr * (0.5 * (1 + np.cos(np.pi * progress)))

        mets = evaluate_model(model, X_va, y_va)
        auc = mets['auc']
        if lr_plateau is not None:
            lr_plateau.step(auc)
        if auc - best_auc > MIN_DELTA:
            best_auc = auc; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            break

    model.load_state_dict(best_state)
    final = evaluate_model(model, X_va, y_va)
    final['best_auc'] = best_auc
    final['epochs_trained'] = epoch + 1
    final['config'] = name
    final['features'] = FEATURES_12
    final['window'] = 90
    final['hidden_size'] = hid
    final['dropout'] = do
    final['weight_decay'] = wd
    final['batch_size'] = bs
    final['schedule'] = sched

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(final, f, indent=2)
    print(f"  => AUC={final['best_auc']:.4f} Acc={final['accuracy']:.4f} Rec={final['recall']:.4f} E={epoch+1}")
    return final


def main():
    X_tr, y_tr, X_va, y_va, scaler = load_split()
    print(f"X_tr={X_tr.shape} X_va={X_va.shape} dist_tr={np.bincount(y_tr.astype(int))}")

    run_base = os.path.join('ml', 'training_runs', 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_base, exist_ok=True)

    overall_best = json.load(open('ml/metrics.json'))
    for i, g in enumerate(CONFIGS):
        print(f"\n{'='*60}\n  [{i+1}/{len(CONFIGS)}] Config: {g[0]}\n{'='*60}")
        out_dir = os.path.join(run_base, g[0])
        mets = train_one(g, X_tr, y_tr, X_va, y_va, out_dir)
        if overall_best is None or mets['best_auc'] > overall_best['best_auc']:
            overall_best = mets
            overall_best['run_dir'] = out_dir
            overall_best['scaler'] = scaler
            print("  *** NEW BEST ***")

    import shutil
    shutil.copy2(os.path.join(overall_best['run_dir'], 'model.pt'), os.path.join('ml', 'models', 'lstm_baseline.pt'))
    with open(os.path.join('ml', 'scaler.json'), 'w') as f:
        json.dump(overall_best['scaler'], f, indent=2)
    deploy = {k: v for k, v in overall_best.items() if k not in ('run_dir', 'scaler')}
    with open(os.path.join('ml', 'metrics.json'), 'w') as f:
        json.dump(deploy, f, indent=2)

    print(f"\n  DEPLOYED: {overall_best['config']}  AUC={overall_best['best_auc']:.4f}")
    print(f"  Runs: {run_base}")


if __name__ == '__main__':
    main()