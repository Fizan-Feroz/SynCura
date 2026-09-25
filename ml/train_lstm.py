"""
Minimal LSTM training scaffold (PyTorch) with AttentionLSTMModel.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np


class SimpleLSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class LSTMModel(nn.Module):
    """Original plain LSTM model (kept for backward compatibility)."""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sig(out).squeeze(-1)


class AttentionLSTMModel(nn.Module):
    """LSTM with temporal attention mechanism for interpretable ICU risk prediction.

    Architecture follows DEWS [Choi et al., IEEE JBHI 2020] and ARLF [Li et al., IEEE Access 2025]:
    - Multi-layer LSTM with dropout
    - Additive attention over all time steps
    - Batch normalization for regularization
    - Dropout before final classification

    forward() returns raw logits (no sigmoid): train with BCEWithLogitsLoss
    and apply sigmoid only at inference/evaluation time.
    The attention weights provide per-timestep interpretability, showing which
    moments in the patient's trajectory most influenced the risk prediction.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        ctx = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Sequential(
            nn.Linear(ctx, ctx),
            nn.Tanh(),
            nn.Linear(ctx, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(ctx)
        self.fc = nn.Linear(ctx, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

        # Additive attention: learn which time steps matter
        attn_scores = self.attention(lstm_out)  # (batch, seq, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)

        # Regularization + classification (logits; sigmoid applied by caller)
        context = self.dropout(context)
        context = self.batch_norm(context)
        out = self.fc(context)
        return out.squeeze(-1)

    def get_attention_weights(self, x):
        """Return per-timestep attention weights for interpretability."""
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights.squeeze(-1)  # (batch, seq)


def train(
    X,
    y,
    epochs=3,
    batch_size=32,
    learning_rate=1e-3,
    device=None,
    progress_callback=None,
    pos_weight=None,
    model_class=None,
    model=None,
    optimizer=None,
    dropout=None,
    weight_decay=0.0,
    hidden_size=None,
    bidirectional=False,
):
    """Train LSTM on (X, y).

    - Uses CUDA automatically if available, unless `device` is provided.
    - Keeps DataLoader CPU-based and moves batches to device each step.
    - `progress_callback(epoch, metrics_dict)` is optional.
    - `pos_weight`: float ratio (neg/pos) for BCEWithLogitsLoss class weighting.
    - `model_class`: which model to create when `model` is not given (default: AttentionLSTMModel).
    - `model` / `optimizer`: pass an existing model (and its optimizer) to
      continue training it instead of starting from scratch. Required for
      correct per-epoch training loops with early stopping.
    - `dropout`: dropout rate for models that support it (ignored otherwise).
    - `hidden_size` / `bidirectional`: architecture overrides for supported models.
    - `weight_decay`: L2 regularization for Adam.
    - Returns `(model, optimizer)` so callers can keep training the same model.
    """
    # One tensor for the whole array: batches are slices, not per-sample
    # torch.tensor() copies (SimpleLSTMDataset is kept for old callers).
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pin = device == "cuda"
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)

    if model is None:
        if model_class is None:
            model_class = AttentionLSTMModel
        try:
            kwargs = {}
            if dropout is not None:
                kwargs["dropout"] = dropout
            if hidden_size is not None:
                kwargs["hidden_size"] = hidden_size
            # Only AttentionLSTMModel supports bidirectional; try, else retry plain
            try:
                model = model_class(input_size=X.shape[-1], bidirectional=bidirectional,
                                    **kwargs).to(device)
            except TypeError:
                model = model_class(input_size=X.shape[-1], **kwargs).to(device)
        except TypeError:
            # Model class does not support extra kwargs (e.g. legacy LSTMModel)
            model = model_class(input_size=X.shape[-1]).to(device)
    else:
        model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    opt = optimizer
    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), dtype=torch.float32, device=device)
    else:
        pw = None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    for e in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dl))
        metrics = {"train_loss": float(avg_loss), "device": str(device)}
        print(f"Epoch {e} loss: {avg_loss:.4f} ({device})")
        if progress_callback is not None:
            progress_callback(e + 1, metrics)

    return model, opt


if __name__ == '__main__':
    # quick smoke test with random data
    X = np.random.randn(200, 60, 6)
    y = (np.random.rand(200) > 0.8).astype(float)
    model, _ = train(X, y, epochs=2)
    torch.save(model.state_dict(), 'ml/lstm_baseline.pt')
    print('Saved ml/lstm_baseline.pt')
