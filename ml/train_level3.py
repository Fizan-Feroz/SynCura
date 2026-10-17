# -*- coding: utf-8 -*-
"""Training and Evaluation Script for Levels 1, 2, and 3 (Multi-Scale, Static Fusion & LightGBM Hybrid).

Trains on the expanded PhysioNet 2012 cohort (Set A + Set C + 80% Set B = ~11,000 patients).
Evaluates on:
- Standard Val Split: seed 42 (stride 15)
- Fresh Holdout: 20% Set B (locked excluded)

Usage:
    .\\.venv\\Scripts\\python.exe -m ml.train_level3 --smoke
    .\\.venv\\Scripts\\python.exe -m ml.train_level3
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.paths import physionet2012_root as _pn_root
from ml.dataset import SERVING_FEATURES, load_physionet_batch
from ml.multiscale_dataset import create_multiscale_sequences
from ml.models_level3 import (
    MultiScaleFusionAttentionLSTM,
    train_multiscale_neural,
    train_lightgbm_model,
    predict_multiscale_batched,
)

BASE = _pn_root()


def eval_binary_metrics(p_prob, y_true, threshold=0.5):
    """Compute AUC, accuracy, sensitivity, specificity, precision, and F1."""
    auc = roc_auc_score(y_true, p_prob)
    preds = (p_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    sens = recall_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    # Specificity
    neg_mask = (y_true == 0)
    spec = float((preds[neg_mask] == 0).sum()) / max(1, neg_mask.sum())
    return {
        'auc': float(auc),
        'accuracy': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'precision': float(prec),
        'f1': float(f1)
    }


def fit_norm(arr):
    flat = arr.reshape(-1, arr.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0) + 1e-6
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std), 1.0, std)
    return mean, std


def apply_norm(arr, mean, std):
    out = (arr - mean) / std
    out = np.where(np.isnan(out), 0.0, out)
    return out


def main():
    parser = argparse.ArgumentParser(description='Train Level 1-3 Multi-Scale Hybrid Model.')
    parser.add_argument('--smoke', action='store_true', help='Quick smoke test on small subset')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs for neural model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for neural model')
    parser.add_argument('--seed', type=int, default=88, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Level 1-3 Training Pipeline (Device: {device}) ===")

    mp = 150 if args.smoke else None
    stride_train = 30
    stride_val = 15

    # 1. Load Set A
    print("\n[1/4] Loading PhysioNet 2012 Set A...", flush=True)
    data_a = load_physionet_batch(
        os.path.join(BASE, 'set-a_full', 'set-a'),
        os.path.join(BASE, 'Outcomes-a.txt'),
        max_patients=mp
    )
    # Split Set A for standard validation (seed 42)
    pids_a = np.array([x[2] for x in data_a])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_a_idx, va_a_idx = next(gss.split(data_a, groups=pids_a))
    train_data_a = [data_a[i] for i in tr_a_idx]
    val_data_a = [data_a[i] for i in va_a_idx]

    # 2. Load Set B (Train portion + Fresh Holdout portion)
    print("[2/4] Loading PhysioNet 2012 Set B...", flush=True)
    data_b = load_physionet_batch(
        os.path.join(BASE, 'set-b_full', 'set-b'),
        os.path.join(BASE, 'Outcomes-b.txt'),
        max_patients=mp
    )
    pids_b = np.array([x[2] for x in data_b])

    # Respect locked slices
    locked_set = set()
    import glob
    for lock_path in sorted(glob.glob(os.path.join('ml', 'locked_holdout*.json'))):
        try:
            locked_set |= set(json.load(open(lock_path))['patients'])
        except Exception:
            pass

    # Split 80/20 Set B
    gss_b = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    tr_b_idx, ho_b_idx = next(gss_b.split(data_b, groups=pids_b))
    train_data_b = [data_b[i] for i in tr_b_idx if data_b[i][2] not in locked_set]
    holdout_data_b = [data_b[i] for i in ho_b_idx if data_b[i][2] not in locked_set]

    # 3. Load Set C
    print("[3/4] Loading PhysioNet 2012 Set C...", flush=True)
    data_c = load_physionet_batch(
        os.path.join(BASE, 'set-c'),
        os.path.join(BASE, 'Outcomes-c.txt'),
        max_patients=mp
    )

    # 4. Generate Multi-Scale Feature Sets
    print("\n[4/4] Generating Multi-Scale Representations...", flush=True)
    print("  Processing Training Cohort (Set A train + Set B train + Set C)...", flush=True)
    combined_train_data = train_data_a + train_data_b + data_c
    X_seq_tr, X_st_tr, X_trd_tr, X_tab_tr, y_tr, _ = create_multiscale_sequences(
        combined_train_data, stride=stride_train
    )

    print("  Processing Validation Cohort (Set A val, stride 15)...", flush=True)
    X_seq_va, X_st_va, X_trd_va, X_tab_va, y_va, _ = create_multiscale_sequences(
        val_data_a, stride=stride_val
    )

    print("  Processing Fresh Holdout Cohort (Set B holdout)...", flush=True)
    X_seq_fho, X_st_fho, X_trd_fho, X_tab_fho, y_fho, _ = create_multiscale_sequences(
        holdout_data_b, stride=stride_train
    )

    print(f"\nCohort Shapes:")
    print(f"  Train: Seq {X_seq_tr.shape} | Tabular {X_tab_tr.shape} | Mortality: {int(y_tr.sum())}/{len(y_tr)}")
    print(f"  Val:   Seq {X_seq_va.shape} | Tabular {X_tab_va.shape} | Mortality: {int(y_va.sum())}/{len(y_va)}")
    print(f"  Hold:  Seq {X_seq_fho.shape} | Tabular {X_tab_fho.shape} | Mortality: {int(y_fho.sum())}/{len(y_fho)}")

    # Normalization (Train stats only)
    mean_seq, std_seq = fit_norm(X_seq_tr)
    X_seq_tr_norm = apply_norm(X_seq_tr, mean_seq, std_seq)
    X_seq_va_norm = apply_norm(X_seq_va, mean_seq, std_seq)
    X_seq_fho_norm = apply_norm(X_seq_fho, mean_seq, std_seq)

    # Concatenate Static (8) + Trend (36) = Context (44)
    X_ctx_tr = np.concatenate([X_st_tr, X_trd_tr], axis=1)
    X_ctx_va = np.concatenate([X_st_va, X_trd_va], axis=1)
    X_ctx_fho = np.concatenate([X_st_fho, X_trd_fho], axis=1)

    mean_ctx, std_ctx = fit_norm(X_ctx_tr)
    X_ctx_tr_norm = apply_norm(X_ctx_tr, mean_ctx, std_ctx)
    X_ctx_va_norm = apply_norm(X_ctx_va, mean_ctx, std_ctx)
    X_ctx_fho_norm = apply_norm(X_ctx_fho, mean_ctx, std_ctx)

    # ==========================================
    # LEVEL 1 & 2: Multi-Scale Neural Attention Model
    # ==========================================
    print("\n--- Training Level 1 & 2: Multi-Scale Fusion AttentionLSTM ---", flush=True)
    neural_model = MultiScaleFusionAttentionLSTM(
        input_size=12, hidden_size=96, static_dim=8, trend_dim=36, dropout=0.3
    )

    epochs = 2 if args.smoke else args.epochs
    neural_model, best_val_auc = train_multiscale_neural(
        neural_model,
        X_seq_tr_norm, X_ctx_tr_norm, y_tr,
        X_seq_va_norm, X_ctx_va_norm, y_va,
        epochs=epochs, lr=args.lr, device=device
    )

    # Evaluate Neural Model using memory-safe batched evaluation
    p_neural_va = predict_multiscale_batched(neural_model, X_seq_va_norm, X_ctx_va_norm, batch_size=512, device=device)
    p_neural_fho = predict_multiscale_batched(neural_model, X_seq_fho_norm, X_ctx_fho_norm, batch_size=512, device=device)

    neural_va_mets = eval_binary_metrics(p_neural_va, y_va)
    neural_fho_mets = eval_binary_metrics(p_neural_fho, y_fho)
    print(f"\n[Level 1 & 2 Neural Results]")
    print(f"  Val AUC:  {neural_va_mets['auc']:.4f} | Recall: {neural_va_mets['sensitivity']:.4f}")
    print(f"  Hold AUC: {neural_fho_mets['auc']:.4f} | Recall: {neural_fho_mets['sensitivity']:.4f}")

    # ==========================================
    # LEVEL 3: LightGBM Clinical Tabular Model
    # ==========================================
    print("\n--- Training Level 3: LightGBM on 116 Engineered Clinical Features ---", flush=True)
    gbm_model, p_lgb_va, lgb_va_auc = train_lightgbm_model(
        X_tab_tr, y_tr, X_tab_va, y_va
    )
    p_lgb_fho = gbm_model.predict(X_tab_fho, num_iteration=gbm_model.best_iteration)
    lgb_va_mets = eval_binary_metrics(p_lgb_va, y_va)
    lgb_fho_mets = eval_binary_metrics(p_lgb_fho, y_fho)

    print(f"\n[Level 3 LightGBM Results]")
    print(f"  Val AUC:  {lgb_va_mets['auc']:.4f} | Recall: {lgb_va_mets['sensitivity']:.4f}")
    print(f"  Hold AUC: {lgb_fho_mets['auc']:.4f} | Recall: {lgb_fho_mets['sensitivity']:.4f}")

    # ==========================================
    # LEVEL 3: Hybrid Meta-Stacking (Neural + Tree)
    # ==========================================
    print("\n--- Level 3 Hybrid Meta-Stacking Evaluation ---", flush=True)
    best_blend_alpha = 0.5
    best_blend_val_auc = 0.0

    # Grid search optimal blend alpha on validation
    for alpha in np.linspace(0.1, 0.9, 9):
        blend_va = alpha * p_neural_va + (1.0 - alpha) * p_lgb_va
        score = roc_auc_score(y_va, blend_va)
        if score > best_blend_val_auc:
            best_blend_val_auc = score
            best_blend_alpha = alpha

    # Final Hybrid predictions
    p_hybrid_va = best_blend_alpha * p_neural_va + (1.0 - best_blend_alpha) * p_lgb_va
    p_hybrid_fho = best_blend_alpha * p_neural_fho + (1.0 - best_blend_alpha) * p_lgb_fho

    hybrid_va_mets = eval_binary_metrics(p_hybrid_va, y_va)
    hybrid_fho_mets = eval_binary_metrics(p_hybrid_fho, y_fho)

    print(f"Optimal Ensemble Weight: {best_blend_alpha:.2f} * Neural + {1.0-best_blend_alpha:.2f} * LightGBM")
    print(f"\n==========================================")
    print(f"FINAL LEVEL 1-3 COMPARISON")
    print(f"==========================================")
    print(f"1. Baseline Deployed v2:            Val AUC: 0.8404 | Holdout AUC: 0.8338")
    print(f"2. Level 1 & 2 Neural Multi-Scale:  Val AUC: {neural_va_mets['auc']:.4f} | Holdout AUC: {neural_fho_mets['auc']:.4f}")
    print(f"3. Level 3 LightGBM Tabular:        Val AUC: {lgb_va_mets['auc']:.4f} | Holdout AUC: {lgb_fho_mets['auc']:.4f}")
    print(f"4. Level 3 HYBRID BLEND (WINNER):   Val AUC: {hybrid_va_mets['auc']:.4f} | Holdout AUC: {hybrid_fho_mets['auc']:.4f}")
    print(f"==========================================")

    # Save artifacts
    out_dir = os.path.join('ml', 'training_runs', 'level3_multiscale_hybrid')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(neural_model.state_dict(), os.path.join(out_dir, 'neural_multiscale.pt'))
    gbm_model.save_model(os.path.join(out_dir, 'lightgbm_model.txt'))

    summary = {
        'neural_val_metrics': neural_va_mets,
        'neural_holdout_metrics': neural_fho_mets,
        'lightgbm_val_metrics': lgb_va_mets,
        'lightgbm_holdout_metrics': lgb_fho_mets,
        'hybrid_val_metrics': hybrid_va_mets,
        'hybrid_holdout_metrics': hybrid_fho_mets,
        'optimal_alpha': float(best_blend_alpha),
        'scaler_seq': {'mean': mean_seq.tolist(), 'std': std_seq.tolist()},
        'scaler_ctx': {'mean': mean_ctx.tolist(), 'std': std_ctx.tolist()},
    }
    with open(os.path.join(out_dir, 'level3_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nArtifacts saved to {out_dir}")


if __name__ == '__main__':
    main()
