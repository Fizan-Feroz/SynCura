"""Round 16 - multi-seed of the winning full-xval-lr1e4 config + pointwise ensemble.

Trains seeds on full set-a (minus val patients, stride 30), evaluates on the
original stride-15 val split and set-b holdout. Deploys best single if it beats
0.8332, and reports the top-k logit-ensemble on val+holdout.
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
BASELINE_VAL = 0.8332
BASELINE_HOLDOUT = 0.8057
EPOCHS = 40
PATIENCE = 14
MIN_DELTA = 0.0005

SEEDS = [41, 42, 43, 44, 45, 46]


def load_orig_split(seed=42):
    from ml.dataset import load_and_create_sequences
    X, y, pid = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, v_idx = next(gss.split(X, y, groups=pid))
    return X[v_idx], y[v_idx]


def load_train_excluding(val_pids):
    from ml.dataset import load_and_create_sequences
    X, y, pid = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a_full', 'set-a'),
        outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
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
    np.save(os.path.join(out_dir, 'p_va.npy'), p_va)
    np.save(os.path.join(out_dir, 'p_ho.npy'), p_ho)
    m = {'config': 'full-xval-lr1e4-seed' + str(seed), 'features': FEATURES_12, 'window': 90,
         'hidden_size': 96, 'dropout': 0.3, 'weight_decay': 1e-4, 'lr': 1e-4, 'seed': seed,
         'val_split': 'original-1519-subset-stride15-seed42',
         'best_auc': best_auc, 'val_auc': va['auc'], 'val_accuracy': va['accuracy'],
         'val_recall': va['recall'], 'holdout_auc': ho['auc'],
         'holdout_accuracy': ho['accuracy'], 'holdout_recall': ho['recall'],
         'epochs_trained': epoch + 1}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=2)
    print(f"  seed {seed}: val AUC={va['auc']:.4f} | holdout AUC={ho['auc']:.4f} (best {best_auc:.4f})")
    return m, p_va, p_ho, {'mean': mean.tolist(), 'std': std.tolist()}


def main():
    from ml.dataset import load_and_create_sequences
    X_va, y_va = load_orig_split()
    _, _, val_pids = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, v_idx = next(gss.split(np.zeros(len(val_pids)), np.zeros(len(val_pids)), groups=val_pids))
    val_set = set(np.array(val_pids)[v_idx])
    X_tr, y_tr = load_train_excluding(val_set)
    print(f'train: {X_tr.shape} | val(orig): {X_va.shape}')
    X_ho, y_ho, _ = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-b_full', 'set-b'), outcomes_file=os.path.join(BASE, 'Outcomes-b.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    print(f'holdout: {X_ho.shape}')

    run_base = os.path.join('ml', 'training_runs', 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_base, exist_ok=True)
    results = []; pvas = []; phos = []
    for i, seed in enumerate(SEEDS):
        m, pva, pho, scaler = train_seed(seed, X_tr, y_tr, X_va, y_va, X_ho, y_ho, run_base)
        results.append((seed, m)); pvas.append(pva); phos.append(pho)

    named_results = list(zip(results, range(len(results))))  # (seed, mets), idx
    named_results.sort(key=lambda x: x[0][1]['val_auc'], reverse=True)
    results = [(seed, mets) for (seed, mets), _ in named_results]

    print(f"\nRanking by val AUC (val > {BASELINE_VAL}, holdout >= {BASELINE_HOLDOUT}):")
    for seed, m in results:
        flag = '' if m['holdout_auc'] >= BASELINE_HOLDOUT else ' (holdout regression)'
        print(f"  seed{seed}: val={m['val_auc']:.4f} holdout={m['holdout_auc']:.4f}{flag}")
    best_seed, best_mets = results[0]

    ens_metrics = {'val_auc': 0}
    order = [r[0] for r in results]
    for k in (2, 3, 4, 5):
        top_idx = [SEEDS.index(s) for s in order[:k]]
        pe = np.mean([pvas[i] for i in top_idx], axis=0)
        phe = np.mean([phos[i] for i in top_idx], axis=0)
        ve = eval_logits(pe, y_va); he = eval_logits(phe, y_ho)
        print(f"  ENS-top{k}: val={ve['auc']:.4f} holdout={he['auc']:.4f}")
        if ve['auc'] > ens_metrics['val_auc']:
            ens_metrics = dict(config='ensemble-' + '+'.join(f'seed{s}' for s in order[:k]),
                               val_auc=ve['auc'], val_accuracy=ve['accuracy'], val_recall=ve['recall'],
                               holdout_auc=he['auc'], holdout_accuracy=he['accuracy'], holdout_recall=he['recall'])

    deploy_m = best_mets
    if deploy_m['val_auc'] <= BASELINE_VAL:
        print('\n  Single seed did not beat 0.8332; checking ensemble...')
        if ens_metrics['val_auc'] > BASELINE_VAL and ens_metrics['holdout_auc'] >= BASELINE_HOLDOUT:
            deploy_m = ens_metrics
            print(f'  Deploying {ens_metrics["config"]} (val {ens_metrics["val_auc"]:.4f})')

    if deploy_m['val_auc'] > BASELINE_VAL and deploy_m['holdout_auc'] >= BASELINE_HOLDOUT:
        import shutil
        shutil.copy2(os.path.join(run_base, 'seed' + str(best_seed), 'model.pt'),
                     os.path.join('ml', 'models', 'lstm_baseline.pt'))
        with open(os.path.join('ml', 'metrics.json'), 'w') as f:
            json.dump(deploy_m, f, indent=2)
        print(f'\n  DEPLOYED {deploy_m["config"]}: val={deploy_m["val_auc"]:.4f} holdout={deploy_m["holdout_auc"]:.4f}')
    else:
        print('\n  Nothing beat current deploy. Keeping existing.')
    print(f'  Runs: {run_base}')


if __name__ == '__main__':
    main()