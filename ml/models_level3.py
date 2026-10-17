# -*- coding: utf-8 -*-
"""Model architectures for Levels 1, 2, and 3 (Static Fusion, Multi-Scale, and LightGBM Hybrid)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


class AdditiveTemporalAttention(nn.Module):
    """Additive Bahdanau-style temporal attention over LSTM hidden states."""
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_size)
        energy = torch.tanh(self.W(lstm_out))
        scores = self.v(energy)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)  # (batch, hidden_size)
        return context, weights


class MultiScaleFusionAttentionLSTM(nn.Module):
    """Deep Neural Network fusing Acute 90-min Vitals with Static & 24h Trend Context."""
    def __init__(self, input_size=12, hidden_size=96, static_dim=8, trend_dim=36,
                 dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
            bidirectional=False
        )
        self.attention = AdditiveTemporalAttention(hidden_size)

        # Context branch (Static demographics + Longitudinal 24h trends)
        total_context_dim = static_dim + trend_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(total_context_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Multi-modal fusion classification head
        fusion_dim = hidden_size + 16  # 96 + 16 = 112
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq, x_context):
        # x_seq: (batch, 90, 12)
        # x_context: (batch, 44)
        lstm_out, _ = self.lstm(x_seq)
        h_temp, attn_weights = self.attention(lstm_out)  # (batch, 96)
        h_context = self.context_mlp(x_context)          # (batch, 16)

        fused = torch.cat([h_temp, h_context], dim=1)    # (batch, 112)
        logits = self.classifier(fused).squeeze(-1)       # (batch,)
        return logits, attn_weights


def predict_multiscale_batched(model, X_seq, X_ctx, batch_size=512, device=None):
    """Run batched forward passes to prevent CUDA OOM on large evaluation sets."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds = []
    with torch.no_grad():
        for b_start in range(0, len(X_seq), batch_size):
            b_seq = torch.tensor(X_seq[b_start:b_start + batch_size], dtype=torch.float32, device=device)
            b_ctx = torch.tensor(X_ctx[b_start:b_start + batch_size], dtype=torch.float32, device=device)
            logits, _ = model(b_seq, b_ctx)
            prob = torch.sigmoid(logits).cpu().numpy()
            preds.append(prob)
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def train_multiscale_neural(model, X_seq_tr, X_ctx_tr, y_tr,
                           X_seq_va, X_ctx_va, y_va,
                           epochs=20, lr=1e-4, batch_size=128, device=None,
                           patience=7):
    """Train the MultiScaleFusionAttentionLSTM with early stopping."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    n_pos = int((y_tr == 1).sum())
    pw = torch.tensor(int((y_tr == 0).sum()) / max(1, n_pos), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auc = 0.0
    best_state = None
    no_improve = 0

    N = len(X_seq_tr)
    indices = np.arange(N)

    for ep in range(epochs):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0

        for b_start in range(0, N, batch_size):
            b_idx = indices[b_start:b_start + batch_size]
            b_seq = torch.tensor(X_seq_tr[b_idx], dtype=torch.float32, device=device)
            b_ctx = torch.tensor(X_ctx_tr[b_idx], dtype=torch.float32, device=device)
            b_y = torch.tensor(y_tr[b_idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits, _ = model(b_seq, b_ctx)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validation eval using memory-safe batched inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        va_preds = predict_multiscale_batched(model, X_seq_va, X_ctx_va, batch_size=512, device=device)
        va_auc = roc_auc_score(y_va, va_preds)

        if va_auc > best_auc + 0.0005:
            best_auc = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"  Neural Epoch {ep+1:02d} | Loss: {total_loss/max(1, n_batches):.4f} | Val AUC: {va_auc:.4f}", flush=True)

        if no_improve >= patience:
            print(f"  Early stopping at epoch {ep+1} (best Val AUC: {best_auc:.4f})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_auc


def train_lightgbm_model(X_tab_tr, y_tr, X_tab_va, y_va):
    """Train an optimized LightGBM gradient boosted tree model on clinical tabular features."""
    print("  Training LightGBM on 116 clinical tabular features...", flush=True)
    n_pos = int((y_tr == 1).sum())
    scale_pos = int((y_tr == 0).sum()) / max(1, n_pos)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos,
        'verbose': -1,
        'random_state': 42,
    }

    trn_data = lgb.Dataset(X_tab_tr, label=y_tr)
    val_data = lgb.Dataset(X_tab_va, label=y_va, reference=trn_data)

    gbm = lgb.train(
        params,
        trn_data,
        num_boost_round=400,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=25, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    va_preds = gbm.predict(X_tab_va, num_iteration=gbm.best_iteration)
    va_auc = roc_auc_score(y_va, va_preds)
    print(f"  LightGBM best iteration: {gbm.best_iteration} | Val AUC: {va_auc:.4f}", flush=True)
    return gbm, va_preds, va_auc
