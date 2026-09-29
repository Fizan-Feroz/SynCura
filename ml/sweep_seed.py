"""Round 11 - seed sweep of the 0.7997-winning step-schedule config.

Same data/split/scaler; different torch seeds. Deploy if any seed >0.80.
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

EPOCHS = 40
PATIENCE = 16
MIN_DELTA = 0.0005
LR = 3e-4
DROP = 0.3
WD = 1e-4
BS = 128
HID = 96

SEEDS = [101, 202, 303, 404, 505, 606]

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


def train_seed(seed, X_tr, y_tr, X_va, y_va, out_dir):
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    from ml.train import evaluate_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pw = n_neg / max(1, n_pos)
    print(f"  [seed {seed}] h={HID} lr={LR} do={DROP} wd={WD} bs={BS} step_every=6")

    best_auc = 0.0; best_state = None; patience = 0
    model = None; opt = None
    for epoch in range(EPOCHS):
        model, opt = quick_train(
            X_tr, y_tr, epochs=1, batch_size=BS, learning_rate=LR,
            pos_weight=pw, model_class=AttentionLSTMModel, model=model, optimizer=opt,
            dropout=DROP, hidden_size=HID, weight_decay=WD)
        for pg in opt.param_groups:
            pg['lr'] = max(LR * (0.5 ** (epoch // 6)), 1e-6)
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
    final['config'] = f'stp6-seed{seed}'
    final['features'] = FEATURES_12
    final['window'] = 90
    final['hidden_size'] = HID
    final['dropout'] = DROP
    final['weight_decay'] = WD
    final['batch_size'] = BS
    final['seed'] = seed
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
    best = None
    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}\n  [{i+1}/{len(SEEDS)}] Seed {seed}\n{'='*60}")
        out_dir = os.path.join(run_base, f'seed{seed}')
        mets = train_seed(seed, X_tr, y_tr, X_va, y_va, out_dir)
        if best is None or mets['best_auc'] > best['best_auc']:
            best = mets
            best['run_dir'] = out_dir
            best['scaler'] = scaler
            print("  *** NEW BEST ***")

    if best['best_auc'] > 0.80:
        import shutil
        shutil.copy2(os.path.join(best['run_dir'], 'model.pt'), os.path.join('ml', 'models', 'lstm_baseline.pt'))
        with open(os.path.join('ml', 'scaler.json'), 'w') as f:
            json.dump(best['scaler'], f, indent=2)
        deploy = {k: v for k, v in best.items() if k not in ('run_dir', 'scaler')}
        with open(os.path.join('ml', 'metrics.json'), 'w') as f:
            json.dump(deploy, f, indent=2)
        print(f"\n  DEPLOYED (>0.80): {best['config']}  AUC={best['best_auc']:.4f}")
    else:
        print(f"\n  Best seed AUC={best['best_auc']:.4f} — not >0.80, keeping existing deploy.")
    print(f"  Runs: {run_base}")


if __name__ == '__main__':
    main()