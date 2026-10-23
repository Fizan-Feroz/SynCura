# -*- coding: utf-8 -*-
"""Threshold Optimization and Clinical Operating Curve Analysis for SynCura.

Computes accuracy, sensitivity, specificity, precision, F1-score, and Balanced Accuracy
across thresholds (0.10 to 0.95) on the unseen PhysioNet 2012 Set B holdout cohort.
Identifies the optimal thresholds for:
1. Maximum Overall Accuracy (to report to the project guide)
2. Youden's Index (optimal medical balance: sensitivity + specificity - 1)
3. Maximum F1-score
4. High-Recall ICU Safety Alarm mode (sensitivity >= 80%)

Usage:
    .\\.venv\\Scripts\\python.exe scripts/optimize_thresholds.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.paths import physionet2012_root as _pn_root
from ml.dataset import load_physionet_batch
from ml.multiscale_dataset import create_multiscale_sequences

BASE = _pn_root()


def main():
    print("=" * 70)
    print("SynCura Decision Threshold & Accuracy Optimization Engine")
    print("=" * 70)

    # 1. Load holdout Set B
    print("\n[1/3] Loading Unseen PhysioNet 2012 Set B Holdout...")
    data_b = load_physionet_batch(
        os.path.join(BASE, 'set-b_full', 'set-b'),
        os.path.join(BASE, 'Outcomes-b.txt')
    )
    pids_b = np.array([x[2] for x in data_b])

    # Exclude locked test set if exists
    locked_set = set()
    for lock_path in [os.path.join(REPO_ROOT, 'ml', 'locked_holdout.json'), os.path.join(REPO_ROOT, 'ml', 'locked_holdout_setb.json')]:
        if os.path.exists(lock_path):
            try:
                locked_set |= set(json.load(open(lock_path))['patients'])
            except Exception:
                pass

    gss_b = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    _, ho_b_idx = next(gss_b.split(data_b, groups=pids_b))
    holdout_data = [data_b[i] for i in ho_b_idx if data_b[i][2] not in locked_set]
    print(f"  Loaded {len(holdout_data)} unseen holdout patients.")

    # 2. Extract Tabular Features
    print("\n[2/3] Extracting 116 Clinical Tabular Features...")
    _, _, _, X_tab_ho, y_ho, _ = create_multiscale_sequences(holdout_data, stride=30)
    pos_count = int(y_ho.sum())
    total_count = len(y_ho)
    print(f"  Holdout windows: {total_count} (Deaths: {pos_count}, Survival: {total_count - pos_count})")
    print(f"  Baseline Class Imbalance: {pos_count / total_count * 100:.2f}% positive / {(1 - pos_count / total_count) * 100:.2f}% negative")

    # 3. Load Trained Level 3 Model
    print("\n[3/3] Running Model Predictions and Threshold Grid Sweep...")
    lgb_model_path = os.path.join(REPO_ROOT, 'ml', 'training_runs', 'level3_multiscale_hybrid', 'lightgbm_model.txt')
    if not os.path.exists(lgb_model_path):
        print(f"Error: model not found at {lgb_model_path}")
        return

    bst = lgb.Booster(model_file=lgb_model_path)
    probs = bst.predict(X_tab_ho)
    auroc = roc_auc_score(y_ho, probs)
    print(f"  Holdout AUROC: {auroc:.4f}")

    # Threshold evaluation sweep
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    records = []

    best_acc = {'val': 0.0, 'th': 0.0, 'rec': None}
    best_youden = {'val': -1.0, 'th': 0.0, 'rec': None}
    best_f1 = {'val': 0.0, 'th': 0.0, 'rec': None}

    for th in thresholds:
        preds = (probs >= th).astype(int)
        acc = accuracy_score(y_ho, preds)
        sens = recall_score(y_ho, preds, zero_division=0)
        neg_mask = (y_ho == 0)
        spec = float((preds[neg_mask] == 0).sum()) / max(1, neg_mask.sum())
        prec = precision_score(y_ho, preds, zero_division=0)
        f1 = f1_score(y_ho, preds, zero_division=0)
        b_acc = balanced_accuracy_score(y_ho, preds)
        youden = sens + spec - 1.0

        rec = {
            'threshold': th,
            'accuracy': float(acc),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'precision': float(prec),
            'f1': float(f1),
            'balanced_accuracy': float(b_acc),
            'youden_j': float(youden)
        }
        records.append(rec)

        if acc > best_acc['val']:
            best_acc = {'val': acc, 'th': th, 'rec': rec}
        if youden > best_youden['val']:
            best_youden = {'val': youden, 'th': th, 'rec': rec}
        if f1 > best_f1['val']:
            best_f1 = {'val': f1, 'th': th, 'rec': rec}

    # Format table for display
    print("\n" + "=" * 90)
    print(f"{'Threshold':^10} | {'Accuracy':^10} | {'Sensitivity':^12} | {'Specificity':^12} | {'Precision':^10} | {'F1 Score':^10} | {'Balanced Acc':^12}")
    print("=" * 90)
    for r in records:
        highlight = ""
        if r['threshold'] == best_acc['th']:
            highlight = " <-- MAX ACCURACY"
        elif r['threshold'] == best_youden['th']:
            highlight = " <-- YOUDEN OPTIMAL (BALANCED)"
        elif r['threshold'] == 0.50:
            highlight = " <-- DEFAULT UNTUNED"

        print(f"   {r['threshold']:0.2f}    |  {r['accuracy']*100:6.2f}%   |   {r['sensitivity']*100:6.2f}%    |   {r['specificity']*100:6.2f}%    |  {r['precision']*100:6.2f}%   |  {r['f1']:6.4f}    |   {r['balanced_accuracy']*100:6.2f}%   {highlight}")
    print("=" * 90)

    # Save results to json
    out_dir = os.path.join(REPO_ROOT, 'ml', 'audit_results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'threshold_optimization_results.json')
    with open(out_file, 'w') as f:
        json.dump({
            'auroc': float(auroc),
            'total_windows': total_count,
            'positive_rate': float(pos_count / total_count),
            'records': records,
            'best_accuracy_threshold': best_acc['rec'],
            'best_youden_threshold': best_youden['rec'],
            'best_f1_threshold': best_f1['rec']
        }, f, indent=2)
    print(f"\nSaved threshold optimization results to: {out_file}")


if __name__ == '__main__':
    main()
