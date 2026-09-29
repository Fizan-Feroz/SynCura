"""Round 10 — last push past 0.80 with step-decay cadence variations.

0.7997 came from step-schedule halving every 6 epochs. Try faster/slower decay,
higher starting LR, and longer training with bigger patience.
"""
import datetime
import json
import os
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

CONFIGS = [
    # (name, lr, step_every, dropout, weight_decay, batch_size)
    # Baseline winner: stacked 6 -> lr3e-4/every6 reached 0.7997
    ("stp6-lr3e4-wd1e4",   (3e-4, 6, 0.3, 1e-4, 128)),
    ("stp4-lr3e4-wd1e4",   (3e-4, 4, 0.3, 1e-4, 128)),
    ("stp8-lr3e4-wd1e4",   (3e-4, 8, 0.3, 1e-4, 128)),
    ("stp6-lr5e4-wd1e4",   (5e-4, 6, 0.3, 1e-4, 128)),
    ("stp6-lr3e4-wd0",     (3e-4, 6, 0.3, 0.0,  128)),
    ("stp6-lr3e4-wd1e4-b256", (3e-4, 6, 0.3, 1e-4, 256)),
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


def train_one(name, p, X_tr, y_tr, X_va, y_va, out_dir):
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    from ml.train import evaluate_model
    lr, step_every, dropout, wd, bs = p
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pw = n_neg / max(1, n_pos)
    print(f"  [{name}] lr={lr} step={step_every} do={dropout} wd={wd} bs={bs}")

    best_auc = 0.0; best_state = None; patience = 0
    model = None; opt = None
    for epoch in range(EPOCHS):
        model, opt = quick_train(
            X_tr, y_tr, epochs=1, batch_size=bs, learning_rate=lr,
            pos_weight=pw, model_class=AttentionLSTMModel, model=model, optimizer=opt,
            dropout=dropout, hidden_size=96, weight_decay=wd)
        for pg in opt.param_groups:
            pg['lr'] = max(lr * (0.5 ** ((epoch + 1) // step_every)), 1e-6)
        mets = evaluate_model(model, X_va, y_va)
        auc = mets['auc']
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
    final['hidden_size'] = 96
    final['dropout'] = dropout
    final['weight_decay'] = wd
    final['batch_size'] = bs
    final['step_every'] = step_every
    final['lr'] = lr
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
    overall_best = None
    for i, (name, p) in enumerate(CONFIGS):
        print(f"\n{'='*60}\n  [{i+1}/{len(CONFIGS)}] Config: {name}\n{'='*60}")
        out_dir = os.path.join(run_base, name)
        mets = train_one(name, p, X_tr, y_tr, X_va, y_va, out_dir)
        if overall_best is None or mets['best_auc'] > overall_best['best_auc']:
            overall_best = mets
            overall_best['run_dir'] = out_dir
            overall_best['scaler'] = scaler
            print("  *** NEW BEST ***")

    if overall_best['best_auc'] > 0.80:
        import shutil
        shutil.copy2(os.path.join(overall_best['run_dir'], 'model.pt'), os.path.join('ml', 'models', 'lstm_baseline.pt'))
        with open(os.path.join('ml', 'scaler.json'), 'w') as f:
            json.dump(overall_best['scaler'], f, indent=2)
        deploy = {k: v for k, v in overall_best.items() if k not in ('run_dir', 'scaler')}
        with open(os.path.join('ml', 'metrics.json'), 'w') as f:
            json.dump(deploy, f, indent=2)
        print(f"\n  DEPLOYED (>0.80): {overall_best['config']}  AUC={overall_best['best_auc']:.4f}")
    else:
        print(f"\n  No config >0.80. Best={overall_best['best_auc']:.4f} — keeping existing deploy.")
    print(f"  Runs: {run_base}")


if __name__ == '__main__':
    main()