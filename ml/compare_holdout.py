"""Round 24 - fair comparison on the SAME fresh holdout (unseen 20% of set-b).

- OLD deployed ensemble [s48,xval-lr1e4,s45,s52] (never saw set-b): val + fresh holdout
- NEW combo-seed ensemble (trained on 80% set-b): val + fresh holdout
- MIXED ensemble if complementary.
Deploys only on strict improvement of val with sane fresh holdout.
"""
import json
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

from ml.paths import physionet2012_root as _pn_root
BASE = _pn_root()
FEATURES_12 = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
               'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose']
COMBO_RUN = 'ml/training_runs/exp_20260917_005359'
OLD = {
    's48': 'ml/training_runs/exp_20260916_220047/seed48/model.pt',
    'xval-lr1e4': 'ml/training_runs/exp_20260916_193507/full-xval-lr1e4/model.pt',
    's45': 'ml/training_runs/exp_20260916_213919/seed45/model.pt',
    's52': 'ml/training_runs/exp_20260916_220047/seed52/model.pt',
}
BASE_VAL = 0.8371


def build_fresh_holdout_with_ids():
    from ml.dataset import load_and_create_sequences
    from sklearn.model_selection import GroupShuffleSplit
    Xb, yb, pb = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-b_full', 'set-b'),
        outcomes_file=os.path.join(BASE, 'Outcomes-b.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
    upb = np.unique(pb)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    _, ho_idx = next(gss.split(np.zeros(len(upb)), np.zeros(len(upb)), groups=upb))
    ho_patients = set(upb[ho_idx])
    m = np.array([p in ho_patients for p in pb])
    return Xb[m], yb[m], np.array(pb)[m]


def build_fresh_holdout():
    X, y, _ = build_fresh_holdout_with_ids()
    return X, y


def load_orig_val():
    from ml.dataset import load_and_create_sequences
    from sklearn.model_selection import GroupShuffleSplit
    X, y, pid = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, v_idx = next(gss.split(X, y, groups=pid))
    return X[v_idx], y[v_idx]


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


def metrics(p, y):
    pr = 1 / (1 + np.exp(-p))
    return (float(roc_auc_score(y, pr)), float(accuracy_score(y, pr > 0.5)),
            float(recall_score(y, pr > 0.5)))


def combo_scaler():
    """Reconstruct the combo-run scaler deterministically (same X_tr mask)."""
    from ml.dataset import load_and_create_sequences
    from sklearn.model_selection import GroupShuffleSplit
    Xa, ya, pa = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a_full', 'set-a'),
        outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
    Xb, yb, pb = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-b_full', 'set-b'),
        outcomes_file=os.path.join(BASE, 'Outcomes-b.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)
    _, _, val_pids = load_and_create_sequences(
        physionet_dir=os.path.join(BASE, 'set-a'), outcomes_file=os.path.join(BASE, 'Outcomes-a.txt'),
        vital_features=FEATURES_12, window_minutes=90, stride=15,
        label_mode='proximity', horizon_hours=12)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, v_idx = next(gss.split(np.zeros(len(val_pids)), np.zeros(len(val_pids)), groups=val_pids))
    val_set = set(np.array(val_pids)[v_idx])
    upb = np.unique(pb)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    _, ho_idx = next(gss2.split(np.zeros(len(upb)), np.zeros(len(upb)), groups=upb))
    ho_patients = set(upb[ho_idx])
    mask = np.array([p not in val_set for p in pa])
    maskb = np.array([p not in ho_patients for p in pb])
    X_tr = np.concatenate([Xa[mask], Xb[maskb]], axis=0)
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mean = np.nanmean(flat, axis=0); std = np.nanstd(flat, axis=0) + 1e-6
    mean = np.where(np.isnan(mean), 0.0, mean); std = np.where(np.isnan(std), 1.0, std)
    return mean, std


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xva_raw, yva = load_orig_val()
    Xfho_raw, yfho = build_fresh_holdout()
    print(f'val: {Xva_raw.shape} | fresh holdout: {Xfho_raw.shape}', flush=True)

    old_scaler = json.load(open('ml/scaler.json'))
    ome, ost = np.array(old_scaler['mean']), np.array(old_scaler['std'])
    Xva_old = ((np.where(np.isnan(Xva_raw), ome, Xva_raw) - ome) / ost).astype(np.float32)
    Xfho_old = ((np.where(np.isnan(Xfho_raw), ome, Xfho_raw) - ome) / ost).astype(np.float32)

    combo_scaler_path = os.path.join('ml', 'scaler_combo8k.json')
    if os.path.exists(combo_scaler_path):
        combo_scaler_json = json.load(open(combo_scaler_path))
        cme, cst = np.array(combo_scaler_json['mean']), np.array(combo_scaler_json['std'])
    else:
        print('reconstructing combo scaler...', flush=True)
        cme, cst = combo_scaler()
        json.dump({'features': FEATURES_12, 'mean': cme.tolist(), 'std': cst.tolist()},
                  open(combo_scaler_path, 'w'), indent=2)
    Xva_c = ((np.where(np.isnan(Xva_raw), cme, Xva_raw) - cme) / cst).astype(np.float32)
    Xfho_c = ((np.where(np.isnan(Xfho_raw), cme, Xfho_raw) - cme) / cst).astype(np.float32)

    P_old_va, P_old_fho = {}, {}
    for k, p in OLD.items():
        P_old_va[k] = preds_of(p, Xva_old, device)
        P_old_fho[k] = preds_of(p, Xfho_old, device)
    P_new_va = {f'c{s}': np.load(os.path.join(COMBO_RUN, f'seed{s}', 'p_va.npy')) for s in [91, 92, 93, 94]}
    P_new_fho = {f'c{s}': np.load(os.path.join(COMBO_RUN, f'seed{s}', 'p_fho.npy')) for s in [91, 92, 93, 94]}

    def show(name, pva, pfho):
        a, acc, rec = metrics(pva, yva)
        h, hacc, hrec = metrics(pfho, yfho)
        print(f'{name}: val={a:.4f} acc={acc:.4f} rec={rec:.4f} | fresh-ho={h:.4f} acc={hacc:.4f} rec={hrec:.4f}',
              flush=True)
        return a, h

    print('--- singles ---', flush=True)
    for k in P_old_va:
        show(f'OLD {k}', P_old_va[k], P_old_fho[k])
    for k in P_new_va:
        show(f'NEW {k}', P_new_va[k], P_new_fho[k])

    print('--- ensembles ---', flush=True)
    old_ens_va = np.mean([P_old_va[k] for k in OLD], axis=0)
    old_ens_fho = np.mean([P_old_fho[k] for k in OLD], axis=0)
    show('OLD-ENS[4]', old_ens_va, old_ens_fho)
    new_ens_va = np.mean([P_new_va[k] for k in P_new_va], axis=0)
    new_ens_fho = np.mean([P_new_fho[k] for k in P_new_fho], axis=0)
    va4, ho4 = show('NEW-ENS[4]', new_ens_va, new_ens_fho)

    best = None
    for r in range(1, 5):
        from itertools import combinations
        for combo in combinations(P_new_va.keys(), r):
            pe = np.mean([P_new_va[k] for k in combo], axis=0)
            a, _, _ = metrics(pe, yva)
            if best is None or a > best[1]:
                phe = np.mean([P_new_fho[k] for k in combo], axis=0)
                h, _, _ = metrics(phe, yfho)
                best = (combo, a, h)
    print(f'BEST-NEW-SUBSET {list(best[0])}: val={best[1]:.4f} fresh-ho={best[2]:.4f}', flush=True)

    if va4 > BASE_VAL:
        out = {'members': ['combo91', 'combo92', 'combo93', 'combo94'],
               'checkpoints': [os.path.join(COMBO_RUN, f'seed{s}', 'model.pt') for s in [91, 92, 93, 94]],
               'val_auc': va4, 'fresh_holdout_auc': ho4, 'scaler': 'ml/scaler_combo8k.json'}
        with open(os.path.join('ml', 'ensemble_combo.json'), 'w') as f:
            json.dump(out, f, indent=2)
        print('SAVED ensemble_combo.json (deploy decision pending)', flush=True)
    else:
        print('NEW-ENS val below 0.8371; trying MIXED old+new greedy...', flush=True)
        pool_va = {}
        pool_fho = {}
        for k in P_old_va:
            pool_va[f'old:{k}'] = P_old_va[k]
            pool_fho[f'old:{k}'] = P_old_fho[k]
        for k in P_new_va:
            pool_va[f'new:{k}'] = P_new_va[k]
            pool_fho[f'new:{k}'] = P_new_fho[k]
        order = sorted(pool_va, key=lambda k: roc_auc_score(yva, 1 / (1 + np.exp(-pool_va[k]))),
                       reverse=True)
        chosen = [order[0]]
        best = roc_auc_score(yva, 1 / (1 + np.exp(-pool_va[order[0]])))
        print(f'  start [{order[0]}]: val={best:.4f}', flush=True)
        for _ in range(9):
            cand = None
            for k in pool_va:
                if k in chosen:
                    continue
                t = chosen + [k]
                a = roc_auc_score(yva, 1 / (1 + np.exp(-np.mean([pool_va[x] for x in t], axis=0))))
                if cand is None or a > cand[1]:
                    cand = (k, a, t)
            k, a, t = cand
            if a > best + 1e-4:
                best = a; chosen = t
                h, _, _ = metrics(np.mean([pool_fho[x] for x in t], axis=0), yfho)
                print(f'  add {k}: val={a:.4f} fresh-ho={h:.4f}', flush=True)
            else:
                print(f'  stop: best add {k} val={a:.4f}', flush=True)
                break
        pe = np.mean([pool_va[x] for x in chosen], axis=0)
        phe = np.mean([pool_fho[x] for x in chosen], axis=0)
        va_m, _, _ = metrics(pe, yva)
        ho_m, _, _ = metrics(phe, yfho)
        print(f'MIXED {chosen}: val={va_m:.4f} fresh-ho={ho_m:.4f}', flush=True)
        if va_m > BASE_VAL and ho_m >= 0.83:
            out = {'members': chosen, 'val_auc': va_m, 'fresh_holdout_auc': ho_m,
                   'note': 'mixed old+combo; per-member scalers (old=ml/scaler.json, new=ml/scaler_combo8k.json)'}
            with open(os.path.join('ml', 'ensemble_mixed.json'), 'w') as f:
                json.dump(out, f, indent=2)
            print('SAVED ensemble_mixed.json (deploy decision pending)', flush=True)
        else:
            print('Mixed ensemble does not clear gates; keeping deployed ensemble.', flush=True)


if __name__ == '__main__':
    main()