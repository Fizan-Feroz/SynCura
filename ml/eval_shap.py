"""
SHAP explainability stub. Loads a trained model and prepares for waveform-level explanations.

Note: For large models and multivariate time series, prefer `shap.DeepExplainer` or model-appropriate explainer.
"""
import os
import numpy as np
import torch
import shap

from ml.train_lstm import LSTMModel


def load_model(path, input_size):
    model = LSTMModel(input_size)
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def explain_sample(model, X_sample):
    # This is a minimal example using KernelExplainer (slow). Replace with DeepExplainer if supported.
    def f(x):
        with torch.no_grad():
            t = torch.tensor(x.reshape(-1, X_sample.shape[1], X_sample.shape[2]), dtype=torch.float32)
            return torch.sigmoid(model(t)).numpy()  # model returns logits

    # Use the sample data itself as background (better than zeros)
    background = X_sample[:min(10, len(X_sample))]
    explainer = shap.KernelExplainer(f, background.reshape(background.shape[0], -1))
    vals = explainer.shap_values(X_sample.reshape(X_sample.shape[0], -1))
    return vals


if __name__ == '__main__':
    print('SHAP explainability script (stub)')
