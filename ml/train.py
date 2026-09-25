"""High-level training script for PhysioNet 2012 dataset.

Standard (reportable) run - full set-a, proximity labels, population norm.
Defaults match the serving contract (12 features, 90-min window, hidden 96):
  python -m ml.train --physionet <set-a> --outcomes <Outcomes-a.txt> \
    --epochs 25 --patience 7 --stride 15 --batch-size 128 --lr 0.0003

Smoke test only (not reportable): add --max-patients 100 --epochs 2

Artifacts always go to the run dir. `--deploy` additionally installs the model
as the single-model fallback (ml/models/lstm_baseline.pt + its own
lstm_baseline_scaler.json). It never writes ml/scaler.json, which the deployed
ensemble members depend on, and refuses configs the engine cannot serve.
"""
import argparse
import datetime
import os
import json
import shutil
import numpy as np
import logging
import warnings
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
import torch
from tqdm import tqdm

from ml.dataset import load_and_create_sequences
from ml.train_lstm import train as quick_train, AttentionLSTMModel

# Serving contract (backend/inference.py builds exactly this input).
SERVING_FEATURES = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
                    'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose']
SERVING_WINDOW = 90
SERVING_HIDDEN = 96
DEPLOY_MODEL_PATH = os.path.join('ml', 'models', 'lstm_baseline.pt')


def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def evaluate_model(model, X, y, batch_size=4096, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    probs_batches = []
    with torch.no_grad():
        for start in tqdm(range(0, len(X), batch_size), desc='Evaluating', leave=False):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            probs_batches.append(torch.sigmoid(logits).detach().cpu().numpy())
    probs = np.concatenate(probs_batches)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float('nan')
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    return {'auc': float(auc), 'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec)}


def serving_contract_errors(args):
    """Reasons the trained config cannot be served by the 12-feature engine ([] = servable)."""
    errors = []
    if list(args.vital_features) != SERVING_FEATURES:
        errors.append(f'vital_features must be {SERVING_FEATURES} in that order')
    if args.window != SERVING_WINDOW:
        errors.append(f'window must be {SERVING_WINDOW}')
    if args.hidden_size != SERVING_HIDDEN:
        errors.append(f'hidden_size must be {SERVING_HIDDEN}')
    if args.bidirectional:
        errors.append('bidirectional models are not servable')
    return errors


def deploy_artifacts(run_dir, model_path=DEPLOY_MODEL_PATH):
    """Install run_dir/model.pt + scaler.json as the single-model fallback.

    The scaler goes NEXT TO the model (<model>_scaler.json), never to
    ml/scaler.json, so ensemble members keep their normalization stats.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    shutil.copy2(os.path.join(run_dir, 'model.pt'), model_path)
    scaler_path = os.path.splitext(model_path)[0] + '_scaler.json'
    shutil.copy2(os.path.join(run_dir, 'scaler.json'), scaler_path)
    return model_path, scaler_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--physionet', required=True, help='Path to PhysioNet set-a/b/c directory')
    parser.add_argument('--outcomes', required=True, help='Path to Outcomes-a.txt/b.txt/c.txt file')
    parser.add_argument('--vital-features', nargs='+',
                        default=list(SERVING_FEATURES))
    parser.add_argument('--window', type=int, default=SERVING_WINDOW, help='Window size in minutes')
    parser.add_argument('--stride', type=int, default=15, help='Minutes between consecutive windows (decorrelates windows)')
    parser.add_argument('--label-mode', choices=['all', 'last', 'proximity'], default='proximity',
                        help="Windowing/labels: 'all' labels every window (noisy), 'last' keeps final window only, "
                             "'proximity' keeps windows ending in the last --horizon-hours of the stay (recommended)")
    parser.add_argument('--horizon-hours', type=float, default=12, help='Outcome horizon for proximity labeling')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate (default: model default)')
    parser.add_argument('--hidden-size', type=int, default=SERVING_HIDDEN, help='LSTM hidden size')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM (DEWS-style)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Adam L2 regularization')
    parser.add_argument('--max-patients', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=0.001, help='Min AUC improvement for early stopping')
    parser.add_argument('--run-dir', default=None, help='Directory to store run outputs (model, logs, metrics)')
    parser.add_argument('--deploy', action='store_true',
                        help='Install the model as ml/models/lstm_baseline.pt (+ its own scaler). '
                             'Refused unless the config matches the serving contract.')
    parser.add_argument('--no-deploy', action='store_true',
                        help='Deprecated no-op: training no longer deploys unless --deploy is given')
    args = parser.parse_args(argv)
    if args.deploy:
        errors = serving_contract_errors(args)
        if errors:
            parser.error('--deploy refused: ' + '; '.join(errors))

    # create a run directory if not provided
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = os.path.join('ml', 'training_runs', 'run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'training.log')
    logger = setup_logging(log_path)

    logger.info('Loading PhysioNet data...')
    X, y, patient_ids = load_and_create_sequences(
        physionet_dir=args.physionet,
        outcomes_file=args.outcomes,
        vital_features=args.vital_features,
        window_minutes=args.window,
        max_patients=args.max_patients,
        stride=args.stride,
        label_mode=args.label_mode,
        horizon_hours=args.horizon_hours,
    )
    X = X.astype(np.float32)  # float32 halves memory vs the float64 windows
    logger.info(f'Loaded X={X.shape} y={y.shape}, class distribution: {np.bincount(y.astype(int))}')

    # Patient-level train/val split to prevent data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=patient_ids))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    del X
    logger.info(f'Split: train={X_train.shape[0]} val={X_val.shape[0]} (patient-level)')

    # Population normalization (Issue 1): statistics from TRAIN ONLY, so absolute
    # severity is preserved. Per-patient z-scoring would erase it. NaNs (columns
    # entirely missing for a patient) are filled with the training mean.
    flat = X_train.reshape(-1, X_train.shape[-1])
    # Accumulate in float64 for accurate stats without a float64 copy of X.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN feature
        train_mean = np.nanmean(flat, axis=0, dtype=np.float64)
        train_std = np.nanstd(flat, axis=0, dtype=np.float64) + 1e-6
    # A feature entirely missing from training (all-NaN) gets neutral stats:
    # missing values filled with 0.0 then normalize to exactly 0.
    train_mean = np.where(np.isnan(train_mean), 0.0, train_mean)
    train_std = np.where(np.isnan(train_std), 1.0, train_std)
    logger.info(f'Population stats per feature {args.vital_features}:')
    for f, m, s in zip(args.vital_features, train_mean, train_std):
        logger.info(f'  {f}: mean={m:.3f} std={s:.3f}')
    mean32, std32 = train_mean.astype(np.float32), train_std.astype(np.float32)
    X_train = (np.where(np.isnan(X_train), mean32, X_train) - mean32) / std32
    X_val = (np.where(np.isnan(X_val), mean32, X_val) - mean32) / std32

    scaler = {'features': args.vital_features,
              'mean': [float(v) for v in train_mean],
              'std': [float(v) for v in train_std]}
    with open(os.path.join(run_dir, 'scaler.json'), 'w') as f:
        json.dump(scaler, f, indent=2)

    # Compute pos_weight for class imbalance
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = n_neg / max(1, n_pos)
    logger.info(f'Class imbalance: neg={n_neg} pos={n_pos} pos_weight={pos_weight:.2f}')
    if not (3.0 <= pos_weight <= 12.0):
        logger.warning(f'pos_weight={pos_weight:.2f} outside expected 3-12x range for PhysioNet 2012 '
                       f'(~14% mortality) - check label windowing for artifacts')

    # ---- Early stopping training loop ----
    logger.info('Training AttentionLSTM with early stopping (patience=%d)...', args.patience)
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    epoch_bar = tqdm(range(args.epochs), desc='Training', unit='epoch')
    model, opt = None, None
    epochs_trained = 0
    for epoch in epoch_bar:
        epochs_trained = epoch + 1
        epoch_bar.set_postfix({'best_auc': f'{best_auc:.4f}', 'patience': f'{patience_counter}/{args.patience}'})

        # Continue training the SAME model (weights + optimizer state persist).
        model, opt = quick_train(
            X_train, y_train,
            epochs=1,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            pos_weight=pos_weight,
            model_class=AttentionLSTMModel,
            model=model,
            optimizer=opt,
            dropout=args.dropout,
            hidden_size=args.hidden_size,
            bidirectional=args.bidirectional,
            weight_decay=args.weight_decay,
        )

        # Evaluate on validation set
        metrics = evaluate_model(model, X_val, y_val)
        logger.info(f'Epoch {epoch + 1} val metrics: {metrics}')

        # Check for improvement
        if metrics['auc'] - best_auc > args.min_delta:
            best_auc = metrics['auc']
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_model(model, os.path.join(run_dir, 'best_model.pt'))
            logger.info(f'  -> New best AUC: {best_auc:.4f} (saved)')
            epoch_bar.set_postfix({'best_auc': f'{best_auc:.4f}', 'status': 'saved!'})
        else:
            patience_counter += 1
            logger.info(f'  -> No improvement ({patience_counter}/{args.patience})')
            epoch_bar.set_postfix({'best_auc': f'{best_auc:.4f}', 'patience': f'{patience_counter}/{args.patience}'})

        if patience_counter >= args.patience:
            logger.info(f'Early stopping at epoch {epoch + 1}')
            tqdm.write(f'[Early Stopping] No improvement for {args.patience} epochs. Stopping.')
            break

    epoch_bar.close()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    final_metrics = evaluate_model(model, X_val, y_val)
    final_metrics['best_auc'] = best_auc
    final_metrics['epochs_trained'] = epochs_trained

    # Save final metrics
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Print final summary
    tqdm.write('\n' + '=' * 50)
    tqdm.write('TRAINING COMPLETE')
    tqdm.write('=' * 50)
    tqdm.write(f'AUC-ROC:       {final_metrics["auc"]:.4f}')
    tqdm.write(f'Accuracy:      {final_metrics["accuracy"]:.4f}')
    tqdm.write(f'Precision:     {final_metrics["precision"]:.4f}')
    tqdm.write(f'Recall:        {final_metrics["recall"]:.4f}')
    tqdm.write(f'Best AUC:      {final_metrics["best_auc"]:.4f}')
    tqdm.write(f'Epochs:        {final_metrics["epochs_trained"]}/{args.epochs}')
    tqdm.write(f'Model saved:   {os.path.join(run_dir, "best_model.pt")}')
    tqdm.write(f'Metrics saved: {metrics_path}')
    tqdm.write('=' * 50)

    logger.info(f'Final validation metrics: {final_metrics}')

    # Save final model
    model_path = os.path.join(run_dir, 'model.pt')
    save_model(model, model_path)
    logger.info('Saved final model to %s', model_path)

    if args.deploy:
        dm, ds = deploy_artifacts(run_dir)
        logger.info('Deployed single-model fallback: %s + %s', dm, ds)
    else:
        logger.info('Not deployed (pass --deploy, or add the run to ml/deployed_manifest.json)')
    return final_metrics


if __name__ == '__main__':
    main()
