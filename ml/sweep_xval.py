"""Round 15 - train on FULL set-a (minus original val patients), evaluate on the
SAME val split / holdout as the deployed 0.8072 model (stride 15) for direct
comparison. More training data (~3200 patients vs ~1219) should lift val.
Deploys if val > 0.8072 and holdout >= 0.7651.
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
BASELINE_VAL = 0.8072
BASELINE_HOLDOUT = 0.7651

CONFIGS = {
    'full-xval-lr3e4':  dict(lr=3e-4, wd=1e-4, step_every=6, seed=31),
    'full-xval-lr2e4':  dict(lr=2e-4, wd=1e-4, step_every=6, seed=32),
    'full-xval-lr2e4b': dict(lr=2e-4, wd=1e-4, step_every=6, seed=33),
    'full-xval-lr1e4':  dict(lr=1e-4, wd=1e-4, step_every=6, seed=34),
}
EPOCHS = 40
PATIENCE = 14
MIN_DELTA = 0.0005


def load_orig_split(seed=42):
    from ml.dataset import load_and_create_sequences
    X, y, pid = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, v_idx = next(gss.split(X, y, groups=pid))
    return X[v_idx], y[v_idx], pid[v_idx]


def load_train_excluding(val_pids):
    from ml.dataset import load_and_create_sequences, load_physionet_batch
    X_tr_list, y_tr_list, keep = [], [], set(val_pids)
    X, y, pid = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a_full', 'set-a'),
        outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
    mask = np.array([p not in keep for p in pid])
    return X[mask], y[mask]


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


def run(name, cfg, X_tr, y_tr, X_va, y_va, X_ho, y_ho, run_base):
    from ml.train_lstm import train as quick_train, AttentionLSTMModel
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mean = np.nanmean(flat, axis=0); std = np.nanstd(flat, axis=0) + 1e-6
    mean = np.where(np.isnan(mean), 0.0, mean); std = np.where(np.isnan(std), 1.0, std)
    def norm(X):
        return ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32)
    X_tr = norm(X_tr); X_va = norm(X_va); X_ho = norm(X_ho)
    print(f"\n=== {name} lr={cfg['lr']} seed={cfg['seed']} | X_tr={X_tr.shape} ===")
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pw = n_neg / max(1, n_pos)
    model = AttentionLSTMModel(input_size=12, hidden_size=96, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    best_auc = 0.0; best_state = None; patience = 0; hist = []
    for epoch in range(EPOCHS):
        model, optimizer = quick_train(
            X_tr, y_tr, epochs=1, batch_size=128, learning_rate=cfg['lr'], pos_weight=pw,
            model_class=AttentionLSTMModel, model=model, optimizer=optimizer,
            dropout=0.3, hidden_size=96, weight_decay=cfg['wd'])
        for pg in optimizer.param_groups:
            pg['lr'] = max(cfg['lr'] * (0.5 ** (epoch // cfg['step_every'])), 1e-6)
        mets = eval_model(model, X_va, y_va, device)
        hist.append(mets['auc'])
        if mets['auc'] - best_auc > MIN_DELTA:
            best_auc = mets['auc']; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    model.load_state_dict(best_state)
    va = eval_model(model, X_va, y_va, device)
    ho = eval_model(model, X_ho, y_ho, device)
    out_dir = os.path.join(run_base, name); os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    m = {'config': name, 'features': FEATURES_12, 'window': 90, 'hidden_size': 96,
         'dropout': 0.3, 'weight_decay': cfg['wd'], 'lr': cfg['lr'],
         'step_every': cfg['step_every'], 'seed': cfg['seed'],
         'val_split': 'original-1519-subset-stride15-seed42', 'val_stride': 15, 'train_stride': 30,
         'best_auc': best_auc, 'val_auc': va['auc'], 'val_accuracy': va['accuracy'],
         'val_recall': va['recall'], 'holdout_auc': ho['auc'],
         'holdout_accuracy': ho['accuracy'], 'holdout_recall': ho['recall'],
         'epochs_trained': epoch + 1, 'auc_history': hist}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    print(f"  => {name}: val AUC={va['auc']:.4f} | holdout AUC={ho['auc']:.4f}")
    return m, os.path.join(out_dir, 'model.pt'), {'mean': mean.tolist(), 'std': std.tolist()}


def main():
    X_va, y_va, val_pids = load_orig_split()
    X_tr, y_tr = load_train_excluding(val_pids)
    print(f'val(orig split, stride15): {X_va.shape} | train(full set-a minus val): {X_tr.shape} dist={np.bincount(y_tr.astype(int))}')
    from ml.dataset import load_and_create_sequences
    X_ho, y_ho, _ = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-b_full', 'set-b'), outcomes_file=os.path.join(BASE, 'Outcomes-b.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    print(f'holdout(orig, stride15): {X_ho.shape} dist={np.bincount(y_ho.astype(int))}')

    run_base = os.path.join('ml', 'training_runs', 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_base, exist_ok=True)
    results = []
    for name, cfg in CONFIGS.items():
        m, path, scaler = run(name, cfg, X_tr, y_tr, X_va, y_va, X_ho, y_ho, run_base)
        results.append((name, m, path, scaler))
    results.sort(key=lambda r: r[1]['val_auc'], reverse=True)
    best_name, best_mets, best_path, best_scaler = results[0]
    print(f"\nRanking by val AUC (val > {BASELINE_VAL}, holdout >= {BASELINE_HOLDOUT}):")
    for name, m, p, s in results:
        flag = '' if m['holdout_auc'] >= BASELINE_HOLDOUT else ' (holdout regression)'
        print(f"  {name}: val={m['val_auc']:.4f} holdout={m['holdout_auc']:.4f}{flag}")
    if best_mets['val_auc'] > BASELINE_VAL and best_mets['holdout_auc'] >= BASELINE_HOLDOUT:
        import shutil
        shutil.copy2(best_path, os.path.join('ml', 'models', 'lstm_baseline.pt'))
        scaler_out = {'features': FEATURES_12, 'mean': best_scaler['mean'], 'std': best_scaler['std']}
        with open(os.path.join('ml', 'scaler.json'), 'w') as f:
            json.dump(scaler_out, f, indent=2)
        deploy = {k: v for k, v in best_mets.items() if k != 'auc_history'}
        with open(os.path.join('ml', 'metrics.json'), 'w') as f:
            json.dump(deploy, f, indent=2)
        print(f"\n  DEPLOYED {best_name}: val={best_mets['val_auc']:.4f} holdout={best_mets['holdout_auc']:.4f}")
    else:
        print('\n  No config beat baseline. Keeping existing deploy.')
    print(f'  Runs: {run_base}')


if __name__ == '__main__':
    main()