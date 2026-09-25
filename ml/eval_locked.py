# -*- coding: utf-8 -*-
"""ONE-TIME locked-slice evaluation (v5 Step 3).

Evaluates the Step-2 winning ensemble against ml/locked_holdout.json
EXACTLY ONCE. Do not re-run with different members/thresholds "to
compare" -- this file exists to be executed a single time, and its output
(ml/LOCKED_EVAL_<date>.json) is the paper/report headline number.

Usage:
  .\\.venv\\Scripts\\python.exe -m ml.eval_locked ml/training_runs/improve_baseline_E1/candidate_ensemble.json
"""
import datetime
import json
import os
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from ml.paths import physionet2012_root as _pn_root

BASE = _pn_root()
FEATURES_12 = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
               'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose']


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate')
    ap.add_argument('--locked', default=os.path.join('ml', 'locked_holdout.json'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cand = json.load(open(args.candidate))
    assert cand.get('candidate_only'), 'refusing: not a candidate manifest'
    locked = json.load(open(args.locked))
    locked_set = set(locked['patients'])
    print(f"locked slice: seed={locked['seed']} n={locked['n_patients']}", flush=True)

    from ml.dataset import load_physionet_batch, create_sequences_from_physionet
    data = load_physionet_batch(os.path.join(BASE, 'set-b_full', 'set-b'),
                                os.path.join(BASE, 'Outcomes-b.txt'))
    data = [(df, y, pid) for df, y, pid in data if pid in locked_set]
    print(f'locked patients loaded: {len(data)}', flush=True)
    X, y, pids = create_sequences_from_physionet(
        data, vital_features=FEATURES_12, window_minutes=90, stride=30,
        label_mode='proximity', horizon_hours=12)

    from ml.train_lstm import AttentionLSTMModel, AttentionLSTMFusionModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = cand.get('model_classes', ['attention'] * len(cand['checkpoints']))
    logits = []
    for ckpt, scpath, cls in zip(cand['checkpoints'], cand['scalers'], classes):
        sc = json.load(open(scpath))
        mean, std = np.array(sc['mean']), np.array(sc['std'])
        n_feat = len(sc.get('features', FEATURES_12))
        Xn = ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32)
        ModelCls = AttentionLSTMFusionModel if cls == 'fusion' else AttentionLSTMModel
        m = ModelCls(input_size=n_feat, hidden_size=96, dropout=0.3)
        try:
            state = torch.load(ckpt, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location='cpu')
        m.load_state_dict(state)
        m.eval().to(device)
        pr = np.zeros(len(Xn))
        with torch.no_grad():
            for off in range(0, len(Xn), 512):
                xb = torch.tensor(Xn[off:off + 512], dtype=torch.float32).to(device)
                pr[off:off + len(xb)] = m(xb).cpu().numpy().ravel()
        logits.append(pr)
    ens = np.mean(logits, axis=0)
    prob = 1 / (1 + np.exp(-ens))
    pred = (prob > 0.5).astype(int)
    out = {
        'candidate': args.candidate,
        'members': cand['members'],
        'locked_seed': locked['seed'],
        'locked_n_patients': locked['n_patients'],
        'n_windows': len(y),
        'auc': float(roc_auc_score(y, prob)),
        'accuracy': float(accuracy_score(y, pred)),
        'sensitivity': float(recall_score(y, pred)),
        'specificity': float(recall_score(1 - y, 1 - pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'evaluated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'note': 'ONE-TIME evaluation. Do not re-run with other configs.',
    }
    default_name = 'LOCKED_EVAL_%s.json' % datetime.datetime.now().strftime('%Y%m%d')
    out_path = args.out or os.path.join('ml', default_name)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    # True-positive patients for the attention check (IDs only, no tuning).
    tp = sorted({p for p, yi, pi in zip(pids, y, prob) if yi == 1 and pi > 0.5})[:10]
    out['tp_patients_for_attention_check'] = tp
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('LOCKED RESULT:', {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in out.items() if k != 'tp_patients_for_attention_check'}, flush=True)
    print('TP patients for Step 4:', tp, flush=True)


if __name__ == '__main__':
    main()
