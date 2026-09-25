"""Challenge 2019 (sepsis) -> SynCura 12-feature contract.

Usage:
    from ml.challenge2019_to_features import load_challenge2019_batch
    from ml.dataset import create_sequences_from_physionet

    data_list = load_challenge2019_batch(challenge2019_setA(), max_patients=500)
    X, y, patient_ids = create_sequences_from_physionet(
        data_list, vital_features=SERVING_FEATURES, window_minutes=90,
        stride=15, label_mode='proximity', horizon_hours=12,
    )

Deliberately reuses ml.dataset.create_sequences_from_physionet for windowing/
labeling rather than reimplementing it, so Challenge 2019 windows are built
by exactly the same causal-forward-fill, patient-level-safe logic as
PhysioNet 2012 windows. This loader's only job is to produce
(df_pivot, label, patient_id) tuples in the same shape
ml.dataset.load_physionet_batch produces.

*** READ THIS BEFORE POOLING WITH PHYSIONET 2012 ***
Challenge 2019's per-row `SepsisLabel` marks proximity to sepsis ONSET, not
in-hospital death. This loader collapses it to a single patient-level flag
(`max(SepsisLabel) over the stay`) so the output shape matches PhysioNet's
`In-hospital_death`, but the two labels mean different things clinically.
Concatenating this loader's (X, y) directly with PhysioNet 2012's (X, y)
into one training call trains the model on a blended, ill-defined target
("died OR had sepsis onset") unless you deliberately intend that. Options,
in order of how defensible they are:
  1. Train a separate sepsis-onset model on this data alone; don't pool.
  2. Multi-task: two output heads (mortality, sepsis), shared LSTM trunk.
  3. Use this purely as unsupervised/self-supervised pretraining for the
     LSTM+attention trunk, then fine-tune the classification head on
     PhysioNet 2012 alone.
Per ml/dataset_registry.py, 'challenge2019' is cleared for 'train' but NOT
'deploy_train' until one of the above is deliberately implemented and
someone signs off on it - don't wire this into ml/train.py's --deploy path
as-is.

*** GCS IS ABSENT FROM CHALLENGE 2019 ***
There is no Glasgow Coma Scale column in this dataset. The GCS column in
the output df_pivot is simply never populated (stays entirely NaN for
every Challenge 2019 patient). ml/train.py's existing normalization code
already handles an all-NaN training-split feature gracefully (mean/std
fall back to 0.0/1.0, so it normalizes to exactly 0 - see the "all-NaN
feature" branch in ml/train.py). That's a reasonable default, not a fix:
every Challenge 2019 window effectively tells the model "GCS is exactly at
the population average" whether or not that's true for that patient. If
GCS matters a lot for this label, consider adding a per-feature
"was this observed from a source that has this column" mask channel
(similar in spirit to the existing `gap_channels` option in ml.dataset)
rather than relying on silent imputation.

*** RESOLUTION MISMATCH ***
Challenge 2019 is hourly-resolution (one row per ICU hour); PhysioNet 2012
is effectively per-minute after reindexing. To produce (90, 12) windows
that are shape-compatible with the existing serving contract, this loader
upsamples each patient's hourly values onto a per-minute grid by
forward-filling each hourly value across the following 59 minutes. This
makes the output shape-compatible but NOT information-equivalent to a
real per-minute PhysioNet window - most of a 90-minute Challenge 2019
window is a repeated hourly value, not new information. Whether that's an
acceptable approximation for scale-up pretraining vs. a distorting
artifact is worth empirically checking (e.g., compare attention-weight
patterns on Challenge 2019 windows vs. PhysioNet windows) before trusting
results trained on this data.
"""
import os
import glob
import numpy as np
import pandas as pd

from ml.dataset_registry import require

# Challenge 2019 PSV column -> SynCura SERVING_FEATURES name.
# GCS deliberately has no mapping (see module docstring).
COLUMN_MAP = {
    'HR': 'HR',
    'Resp': 'RespRate',
    'Temp': 'Temp',
    'SBP': 'NISysABP',
    'DBP': 'NIDiasABP',
    'O2Sat': 'SpO2',
    'BUN': 'BUN',
    'Creatinine': 'Creatinine',
    'WBC': 'WBC',
    'Platelets': 'Platelets',
    'Glucose': 'Glucose',
}
SERVING_FEATURES_ORDER = [
    'HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
    'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose',
]


def _load_one_psv(path):
    """Load one Challenge-2019 patient .psv, return (df_pivot, label, patient_id).

    df_pivot: minute-indexed DataFrame with SERVING_FEATURES_ORDER columns
    (GCS all-NaN), matching ml.dataset.load_physionet_file's output shape.
    label: 1 if SepsisLabel is ever 1 during the stay, else 0.
    """
    df = pd.read_csv(path, sep='|')
    if 'SepsisLabel' not in df.columns:
        return None

    patient_id = os.path.splitext(os.path.basename(path))[0]
    label = int((df['SepsisLabel'] == 1).any())

    # ICULOS is the hour index (1-based per the Challenge-2019 spec).
    hours = df['ICULOS'].astype(int) if 'ICULOS' in df.columns else pd.Series(range(1, len(df) + 1))
    minutes = (hours - hours.min()) * 60

    present_cols = {src: dst for src, dst in COLUMN_MAP.items() if src in df.columns}
    if not present_cols:
        return None

    hourly = pd.DataFrame(
        {dst: df[src].values for src, dst in present_cols.items()},
        index=minutes.values,
    )
    hourly.index.name = 'minutes'
    hourly = hourly[~hourly.index.duplicated(keep='last')].sort_index()

    # Upsample hourly -> per-minute by forward-filling each hourly value
    # across the following 59 minutes (see "RESOLUTION MISMATCH" above).
    full_minute_index = range(int(hourly.index.min()), int(hourly.index.max()) + 1)
    per_minute = hourly.reindex(full_minute_index).ffill()

    # Add GCS as an all-NaN column and enforce the canonical column order.
    per_minute['GCS'] = np.nan
    per_minute = per_minute.reindex(columns=SERVING_FEATURES_ORDER)

    return per_minute, label, patient_id


def load_challenge2019_batch(setA_dir, max_patients=None):
    """Load Challenge-2019 training_setA .psv files into ml.dataset-compatible tuples."""
    require('challenge2019', 'train')  # raises with a clear reason if misused

    files = sorted(glob.glob(os.path.join(setA_dir, '*.psv')))
    if max_patients:
        files = files[:max_patients]

    data = []
    skipped = 0
    for fp in files:
        try:
            result = _load_one_psv(fp)
            if result is None:
                skipped += 1
                continue
            data.append(result)
        except Exception as e:
            print(f'Warning: failed to load {fp}: {e}')
            skipped += 1

    if skipped:
        print(f'Warning: skipped {skipped} of {len(files)} Challenge-2019 files')
    return data


if __name__ == '__main__':
    print('ml.challenge2019_to_features loaded')
