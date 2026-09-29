import time
import sys

try:
    import torch
    from ml.train_lstm import LSTMModel
except Exception as e:
    print('ERROR: torch or training module not available:', e)
    sys.exit(2)

# Adjust to match model shape
batch_size = 32
seq_len = 60
features = 5
iters = 200

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = LSTMModel(input_size=features).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # LSTMModel returns logits

    X = torch.randn(batch_size, seq_len, features, device=device)
    y = torch.rand(batch_size, device=device).round()

    # warm-up
    for _ in range(10):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t0 = time.time()
    for i in range(iters):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t1 = time.time()
    avg = (t1 - t0) / iters
    print(f"avg time per train batch: {avg:.6f}s")
    if device.type == 'cuda':
        print('GPU detected; timings include GPU execution.')

if __name__ == '__main__':
    run_benchmark()
