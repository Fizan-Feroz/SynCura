"""PhysioNet 2012 dataset loader utilities.

Usage:
  from ml.dataset import load_and_create_sequences
  X, y = load_and_create_sequences(
      physionet_dir="C:/Users/fizan/Downloads/Techfusion/.../set-a",
      outcomes_file="C:/Users/fizan/Downloads/Techfusion/.../Outcomes-a.txt"
  )

PhysioNet format: CSV files per patient with columns [Time, Parameter, Value].
Expected parameters: HR, RespRate, Temp, NISysABP, NIDiasABP, NIMAP, etc.
"""
import os
import glob
import numpy as np
import pandas as pd

# PhysioNet 2012 uses different names for some vitals than bedside monitors.
# Aliases are resolved in order when the requested feature is absent:
#   'SpO2' (pulse-ox saturation) -> 'SaO2' (arterial saturation, gold standard,
#   same units %). The model slot stays named 'SpO2' so training, inference,
#   and the frontend remain in sync.
PARAMETER_ALIASES = {
    'SpO2': ['SaO2'],
}


def carry_forward(window, initial=None):
    """Forward-fill a (T, F) window, seeded with last-known values from before it.

    Serving-side mirror of the training windows in
    create_sequences_from_physionet (`df_raw.ffill()` over the whole stay, then
    slice): a gap takes the most recent earlier observation, including one
    measured before the window started (`initial`, NaN = never observed).
    No interpolation and no backward fill, so nothing leaks from the future.
    Values never observed stay NaN; callers fill them with the train mean.
    """
    W = np.array(window, dtype=np.float64, copy=True)
    last = (np.full(W.shape[1], np.nan) if initial is None
            else np.array(initial, dtype=np.float64, copy=True))
    for t in range(W.shape[0]):
        row = W[t]
        missing = np.isnan(row)
        row[missing] = last[missing]
        last = row
    return W


def load_physionet_file(file_path, patient_outcome=None):
    """Load a single PhysioNet patient file and pivot to wide format."""
    df = pd.read_csv(file_path)
    record_id = os.path.basename(file_path).replace('.txt', '')
    df_pivot = df[df['Parameter'] != 'RecordID'].pivot_table(index='Time', columns='Parameter', values='Value')

    def time_to_minutes(t):
        h, m = map(int, t.split(':'))
        return h * 60 + m

    df_pivot.index = df_pivot.index.map(time_to_minutes)
    df_pivot = df_pivot.sort_index()
    df_pivot.index.name = 'minutes'

    for col in df_pivot.columns:
        df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce')
    df_pivot = df_pivot.dropna(axis=1, how='all')

    return df_pivot, record_id, patient_outcome


def load_physionet_batch(physionet_dir, outcomes_file=None, max_patients=None):
    """Load all PhysioNet patient files from a directory."""
    outcomes = {}
    if outcomes_file and os.path.exists(outcomes_file):
        outcomes_df = pd.read_csv(outcomes_file)
        if {'RecordID', 'In-hospital_death'}.issubset(outcomes_df.columns):
            outcomes = dict(zip(
                outcomes_df['RecordID'].astype(str),
                outcomes_df['In-hospital_death'].astype(int)
            ))
        else:
            with open(outcomes_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        outcomes[parts[0]] = int(parts[1])

    patient_files = sorted(glob.glob(os.path.join(physionet_dir, '*.txt')))
    if max_patients:
        patient_files = patient_files[:max_patients]

    data = []
    skipped = 0
    for pf in patient_files:
        try:
            patient_id = os.path.basename(pf).replace('.txt', '')
            if patient_id not in outcomes:
                skipped += 1
                continue
            label = outcomes[patient_id]
            df_pivot, rec_id, _ = load_physionet_file(pf, label)
            data.append((df_pivot, label, patient_id))
        except Exception as e:
            print(f'Warning: failed to load {pf}: {e}')

    if skipped > 0:
        print(f'Warning: skipped {skipped} patients with no outcome record')
    return data


def create_sequences_from_physionet(data_list, vital_features=None, window_minutes=60, stride=1,
                                    label_mode='all', horizon_hours=12, gap_channels=False):
    """Convert PhysioNet data list into sequences X, y.

    No normalization is applied here on purpose: per-patient z-scoring erases
    absolute severity (the main mortality signal). Normalization uses
    population statistics computed on the training split (see ml/train.py).

    `stride` controls the step (in minutes) between consecutive windows.
    `label_mode` controls which windows are kept and how they are labeled:
      - 'all': every window gets the patient's whole-stay outcome (noisy for
        early windows, kept for backward compatibility).
      - 'last': only the final window per patient (clean labels, small data).
      - 'proximity' (recommended): only windows ending within the last
        `horizon_hours` of the stay, labeled by patient outcome. Early,
        likely-stable windows that would inject label noise are dropped.
        End-of-record is an approximation of outcome time (death/discharge
        may occur after the 48h record), but far less noisy than 'all'.
    Remaining NaNs (columns entirely missing for a patient) are left as NaN;
    callers fill them with the training-set mean before normalizing.

    Windows are CAUSAL: the stay is forward-filled once (past-only, vectorized),
    then windows are sliced from the carried frame — no future information can
    leak, while sparse labs keep their last-known value. Values never observed
    before the window end stay NaN (train-mean filled later). NOTE: checkpoints
    trained before this fix used whole-stay interpolation + backward-fill and
    must be retrained for a fair comparison.

    `gap_channels`: when True, append 12 time-since-last-observation channels
    (minutes since the feature was actually measured, 0 = observed now, capped
    at `window_minutes`). This preserves the irregular-sampling signal that
    interpolation/ffill would otherwise erase. Output is (N, window, 2*F).
    """
    if vital_features is None:
        vital_features = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP']

    X_all = []
    y_all = []
    patient_ids = []

    for df_pivot, label, patient_id in data_list:
        # Resolve aliases so requested features map to available PhysioNet params
        resolved = {}
        for f in vital_features:
            if f in df_pivot.columns:
                resolved[f] = df_pivot[f]
            else:
                for alias in PARAMETER_ALIASES.get(f, []):
                    if alias in df_pivot.columns:
                        resolved[f] = df_pivot[alias]
                        break
        available = [f for f in vital_features if f in resolved]
        if len(available) == 0:
            continue

        # Only use columns that exist; fill missing columns with NaN (not zero)
        minute_index = range(int(df_pivot.index.min()), int(df_pivot.index.max()) + 1)
        df_vitals = pd.DataFrame({f: resolved[f] for f in available}, index=df_pivot.index)
        df_raw = df_vitals.reindex(index=minute_index, columns=vital_features)

        seq_len = window_minutes
        if len(df_raw) < seq_len:
            continue

        if label_mode == 'last':
            starts = [len(df_raw)]
        elif label_mode == 'proximity':
            cutoff = max(seq_len, len(df_raw) - int(horizon_hours * 60))
            starts = range(cutoff, len(df_raw), stride)
        else:  # 'all'
            starts = range(seq_len, len(df_raw), stride)

        # CAUSAL carry-forward: ffill over the whole stay is past-only by
        # construction, so it can never leak future information. Slice windows
        # from the carried frame: sparse labs (BUN, creatinine, ...) keep
        # their last-known value instead of collapsing to the train mean.
        # Deliberately NO linear interpolation and NO backward-fill:
        #  - bfill pulls values from the future (the original leakage);
        #  - interpolate-then-slice-only-inside-the-window discards pre-window
        #    history, which starved sparse labs and collapsed AUC to ~0.69.
        # Values never observed before the window end stay NaN and are filled
        # with the train mean at normalization time (honest missingness).
        # Models trained before the causal fix still need retraining.
        df_carry = df_raw.ffill()
        for i in starts:
            window_raw = df_raw.iloc[i - seq_len:i]
            obs_win = window_raw.notna().values.astype(np.float32)
            window = df_carry.iloc[i - seq_len:i].values
            if gap_channels:
                gaps = np.zeros_like(obs_win, dtype=np.float32)
                for c in range(obs_win.shape[1]):
                    g = 0.0
                    for t in range(obs_win.shape[0]):
                        if obs_win[t, c] > 0.5:
                            g = 0.0
                        else:
                            g = min(g + 1.0, float(seq_len))
                        gaps[t, c] = g
                window = np.concatenate([window, gaps], axis=1)
            X_all.append(window)
            y_all.append(label)
            patient_ids.append(patient_id)

    if len(X_all) == 0:
        raise ValueError('No valid sequences created from data')

    X = np.stack(X_all)
    y = np.array(y_all)
    patient_ids = np.array(patient_ids)
    return X, y, patient_ids


def load_and_create_sequences(physionet_dir, outcomes_file=None, vital_features=None, window_minutes=60, max_patients=None, stride=1,
                                label_mode='all', horizon_hours=12, gap_channels=False):
    """All-in-one: load PhysioNet directory and create training sequences."""
    data_list = load_physionet_batch(physionet_dir, outcomes_file, max_patients)
    X, y, patient_ids = create_sequences_from_physionet(data_list, vital_features, window_minutes, stride,
                                                        label_mode, horizon_hours, gap_channels)
    return X, y, patient_ids


if __name__ == '__main__':
    print('ml.dataset loaded')
