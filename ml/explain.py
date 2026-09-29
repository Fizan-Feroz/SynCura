"""SHAP-based explainability for the ICU prediction model.

Uses KernelSHAP to compute per-feature importance for individual patient predictions.
Based on: Liu et al. (2023) "Early prediction of MODS interventions using ML" and
xTimesNet-TSR-CoMTE framework (IEEE 2025).
"""
import shap
import torch
import numpy as np

from ml.dataset import SERVING_FEATURES


FEATURE_NAMES = SERVING_FEATURES


def compute_shap_explanation(model, patient_sequence, feature_names=None, n_background=20):
    """Compute SHAP values for a single patient's prediction.

    Args:
        model: Trained AttentionLSTMModel (eval mode).
        patient_sequence: numpy array of shape (window_size, n_features).
        feature_names: list of feature name strings.
        n_background: number of background samples for KernelSHAP.

    Returns:
        dict mapping feature names to mean absolute SHAP values.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    model.eval()
    window_size, n_features = patient_sequence.shape

    # Wrapper: takes flat 2D input, returns prediction probability
    # (model outputs logits; sigmoid applied here)
    def predict_fn(x_flat):
        batch = x_flat.shape[0]
        x = torch.tensor(
            x_flat.reshape(batch, window_size, n_features),
            dtype=torch.float32
        )
        with torch.no_grad():
            return torch.sigmoid(model(x)).numpy()

    # Background: mean-centered noise around the patient's data.
    # Must have the same (window, features) shape as the input so the
    # flattened background matches what predict_fn() reshapes.
    bg_mean = patient_sequence.mean(axis=0, keepdims=True)  # (1, n_features)
    noise = np.random.randn(n_background, window_size, n_features) * 0.1
    background = np.broadcast_to(bg_mean, (n_background, window_size, n_features)).copy() + noise
    bg_flat = background.reshape(n_background, -1)

    explainer = shap.KernelExplainer(predict_fn, bg_flat)
    patient_flat = patient_sequence.reshape(1, -1)

    shap_values = explainer.shap_values(patient_flat, nsamples=100)

    # Aggregate: mean absolute SHAP per feature (across all time steps)
    vals = shap_values[0] if isinstance(shap_values, list) else shap_values
    vals = vals.flatten()

    feature_importance = {}
    for i, name in enumerate(feature_names):
        # Every n_features-th value corresponds to this feature across timesteps
        feature_vals = vals[i::n_features]
        feature_importance[name] = float(np.abs(feature_vals).mean())

    return feature_importance


def compute_attention_interpretation(model, patient_sequence):
    """Extract attention weights to show which time steps mattered most.

    Returns:
        list of floats (one per timestep) representing attention weights.
    """
    model.eval()
    x_tensor = torch.from_numpy(patient_sequence.reshape(1, *patient_sequence.shape)).float()
    with torch.no_grad():
        weights = model.get_attention_weights(x_tensor)
    return weights.squeeze(0).numpy().tolist()
