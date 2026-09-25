# -*- coding: utf-8 -*-
"""Round 25 - produce presentation evidence figures for the DEPLOYED ensemble.

Replicates backend inference exactly (inference.py): 3 deployed checkpoints
(s48, c93, s45), logit-averaged, normalized with ml/scaler.json.
Outputs for the deck:
  ppt/figs/roc.png         ROC curves (val + fresh holdout) with AUC + bootstrap CI
  ppt/figs/metrics_table.png  val/holdout metrics incl. specificity + precision + F1
  ppt/figs/metrics.json    the same numbers for the deck builder
  ppt/figs/shap.png        per-feature SHAP importance (mean-pooled KernelSHAP)
"""
import json
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.compare_holdout import load_orig_val, build_fresh_holdout_with_ids, FEATURES_12
from ml.train_lstm import AttentionLSTMModel


def brier_score(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def expected_calibration_error(y, p, n_bins=10):
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (p > edges[b]) & (p <= edges[b + 1] if b < n_bins - 1 else p <= edges[b + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def patient_level_auc(y, p, pids):
    """One score per patient (max window probability); windows are correlated
    so window-level AUC overstates the independent sample count."""
    df = {}
    for yi, pi, pidi in zip(y, p, pids):
        if pidi not in df or pi > df[pidi][0]:
            df[pidi] = (pi, yi)
    probs = np.array([v[0] for v in df.values()])
    labels = np.array([v[1] for v in df.values()])
    if len(np.unique(labels)) < 2:
        return None, len(df)
    return float(roc_auc_score(labels, probs)), len(df)

from ml.paths import physionet2012_root as _pn_root
BASE = _pn_root()
ENS = {
    "s48": "ml/training_runs/exp_20260916_220047/seed48/model.pt",
    "c93": "ml/training_runs/exp_20260917_005359/seed93/model.pt",
    "s45": "ml/training_runs/exp_20260916_213919/seed45/model.pt",
}
OUT_DIR = "ppt/figs"
FEATS = ["HR", "RespRate", "Temp", "SysBP", "DiasBP", "SpO2", "GCS", "BUN",
         "Creatinine", "WBC", "Platelets", "Glucose"]


def load_ensemble(device):
    models = []
    for name, path in ENS.items():
        m = AttentionLSTMModel(input_size=12, hidden_size=96, dropout=0.3)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval().to(device)
        models.append(m)
    return models


def predict_logits(models, X, device, batch=512):
    """Logit-averaged ensemble predictions (matches backend/inference.py)."""
    out = np.zeros(len(X), dtype=np.float32)
    for off in range(0, len(X), batch):
        xb = torch.tensor(X[off:off + batch], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = np.mean([m(xb).cpu().numpy().ravel() for m in models], axis=0)
        out[off:off + len(xb)] = logits
    return out


def normalize(X):
    sc = json.load(open("ml/scaler.json"))
    mean, std = np.array(sc["mean"]), np.array(sc["std"])
    return ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32)


def ci_auc(p, y, n_boot=2000, seed=7):
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    idx = np.arange(len(y))
    for i in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            aucs[i] = np.nan
        else:
            aucs[i] = roc_auc_score(y[s], p[s])
    return float(np.nanpercentile(aucs, 2.5)), float(np.nanpercentile(aucs, 97.5))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading data...", flush=True)
    Xva, yva = load_orig_val()
    Xho, yho, hopids = build_fresh_holdout_with_ids()
    print(f"val {Xva.shape} | fresh holdout {Xho.shape} ({len(np.unique(hopids))} patients)", flush=True)

    models = load_ensemble(device)
    pva = predict_logits(models, normalize(Xva), device)
    pho = predict_logits(models, normalize(Xho), device)
    prva = 1 / (1 + np.exp(-np.clip(pva, -30, 30)))
    prho = 1 / (1 + np.exp(-np.clip(pho, -30, 30)))

    def block(y, p, thr=0.5):
        return {
            "auc": float(roc_auc_score(y, p)),
            "accuracy": float(accuracy_score(y, p > thr)),
            "sensitivity": float(recall_score(y, p > thr)),
            "specificity": float(recall_score(1 - y, (1 - (p > thr)).astype(int))),
            "precision": float(precision_score(y, p > thr, zero_division=0)),
            "f1": float(f1_score(y, p > thr, zero_division=0)),
        }

    va = block(yva, prva)
    ho = block(yho, prho)
    lo_ci, hi_ci = ci_auc(prho, yho)
    va["auc_ci"] = None
    ho["auc_ci"] = [lo_ci, hi_ci]
    va["brier"] = brier_score(yva, prva)
    ho["brier"] = brier_score(yho, prho)
    va["ece"] = expected_calibration_error(yva, prva)
    ho["ece"] = expected_calibration_error(yho, prho)
    ho_patient_auc, ho_n_patients = patient_level_auc(yho, prho, hopids)
    ho["patient_level_auc"] = ho_patient_auc
    ho["patient_level_n"] = ho_n_patients
    print("VAL:", va, "\nHOLDOUT:", ho, flush=True)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"val": va, "holdout": ho}, f, indent=2)
    print("wrote metrics.json", flush=True)

    # ---- ROC figure ----
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
    for label, y, p in [("Validation", yva, prva), ("Unseen holdout", yho, prho)]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2.6, label=f"{label} AUC {roc_auc_score(y, p):.3f}")
        ax.fill_between(fpr, tpr, alpha=0.12)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    ax.set_xlabel("False positive rate (1 - specificity)", fontsize=12)
    ax.set_ylabel("True positive rate (sensitivity)", fontsize=12)
    ax.set_title("AttentionLSTM ensemble - ROC", fontsize=13, weight="bold")
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.annotate(f"holdout AUC 95% CI\n{ho['auc_ci'][0]:.3f} - {ho['auc_ci'][1]:.3f}",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "roc.png"))
    plt.close(fig)
    print("wrote roc.png", flush=True)

    # ---- metrics table figure ----
    rows = [
        ["Metric", "Validation", "Unseen holdout"],
        ["AUC", f"{va['auc']:.3f}", f"{ho['auc']:.3f}"],
        ["Accuracy", f"{va['accuracy']:.3f}", f"{ho['accuracy']:.3f}"],
        ["Sensitivity", f"{va['sensitivity']:.3f}", f"{ho['sensitivity']:.3f}"],
        ["Specificity", f"{va['specificity']:.3f}", f"{ho['specificity']:.3f}"],
        ["Precision", f"{va['precision']:.3f}", f"{ho['precision']:.3f}"],
        ["F1", f"{va['f1']:.3f}", f"{ho['f1']:.3f}"],
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    ax.axis("off")
    tab = ax.table(cellText=rows, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(13)
    tab.scale(1, 1.8)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("0.6")
        if r == 0:
            cell.set_facecolor("#1757AC")
            cell.set_text_props(color="white", weight="bold")
        elif c == 0:
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "metrics_table.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote metrics_table.png", flush=True)

    # ---- SHAP figure (mean-pooled KernelSHAP over a few holdout patients) ----
    try:
        import shap
        rng = np.random.default_rng(3)
        picks = np.argsort(prho)[::-1][:3]
        bg = np.nan_to_num(Xho[rng.choice(picks, 20)], nan=0.0)

        def pred_fn(x_flat):
            b = x_flat.shape[0]
            x = torch.tensor(x_flat.reshape(b, 90, 12), dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = torch.stack([m(x).cpu() for m in models]).mean(0)
                return torch.sigmoid(logits).numpy()

        explainer = shap.KernelExplainer(pred_fn, bg.reshape(20, -1))
        feat_imp = np.zeros(12)
        for pidx in picks:
            x = np.nan_to_num(normalize(Xho[[pidx]]), nan=0.0)
            sv = explainer.shap_values(x.reshape(1, -1), nsamples=60)
            vals = sv[0] if isinstance(sv, list) else sv
            feat_imp += np.abs(vals.reshape(90, 12)).mean(axis=0)
        feat_imp /= len(picks)
        order = np.argsort(feat_imp)
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
        ax.barh(range(12), feat_imp[order], color="#1757AC")
        ax.set_yticks(range(12), [FEATS[i] for i in order], fontsize=11)
        ax.set_xlabel("mean |SHAP value| (3 high-risk holdout patients)", fontsize=11)
        ax.set_title("Which features move the risk score?", fontsize=13, weight="bold")
        ax.grid(alpha=0.25, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "shap.png"))
        plt.close(fig)
        print("wrote shap.png", flush=True)
    except Exception as e:
        print(f"SHAP skipped: {e}", flush=True)


if __name__ == "__main__":
    main()