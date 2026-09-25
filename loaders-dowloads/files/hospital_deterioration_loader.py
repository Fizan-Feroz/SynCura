"""hospital-deterioration-dataset -> SynCura pipeline smoke test. NOT CLINICAL EVIDENCE.

Per DATA.md: "10,000 simulated admissions... Use: end-to-end pipeline
smoke tests and benchmarking harness development... Never report metrics
from this as clinical evidence." ml/dataset_registry.py enforces this by
only clearing 'hospital_deterioration' for 'smoke_test'.

*** FEATURE COVERAGE IS INCOMPLETE - READ BEFORE USING ***
Checked against data_dictionary.md's full column list (patients.csv,
vitals_timeseries.csv, labs_timeseries.csv): this dataset has NO column
for GCS, BUN, Platelets, or Glucose anywhere in its schema. That's 4 of
SynCura's 12 model input features that this dataset can never populate -
they'll be NaN for every row, every patient, always. A pipeline test that
"passes" on this dataset has only exercised 8 of 12 input channels. It also
has several columns SynCura doesn't use (lactate, crp_level, hemoglobin,
sepsis_risk_score) - dropped here, not mapped.

*** LABEL MISMATCH, SAME CAVEAT AS CHALLENGE 2019 ***
This dataset's target is `deterioration_next_12h` (any deterioration event
in the next 12h), not in-hospital mortality. Don't treat a model's
performance on this label as informative about mortality prediction
performance, even loosely - they're different clinical questions.
"""
import os
import numpy as np
import pandas as pd

from ml.dataset_registry import require

VITALS_COLUMN_MAP = {
    'heart_rate': 'HR',
    'respiratory_rate': 'RespRate',
    'temperature_c': 'Temp',
    'systolic_bp': 'NISysABP',
    'diastolic_bp': 'NIDiasABP',
    'spo2_pct': 'SpO2',
}
LABS_COLUMN_MAP = {
    'wbc_count': 'WBC',
    'creatinine': 'Creatinine',
    # No BUN, Platelets, or Glucose columns exist in this dataset - see
    # module docstring. Deliberately not mapped (would silently fabricate
    # a column that isn't there).
}
SERVING_FEATURES_ORDER = [
    'HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
    'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose',
]

_WARNED = False


def _warn_once():
    global _WARNED
    if not _WARNED:
        print(
            'WARNING: hospital-deterioration-dataset only covers 8 of 12 '
            'SynCura features (no GCS, BUN, Platelets, Glucose) and uses a '
            'different label (deterioration_next_12h) than PhysioNet 2012 '
            '(In-hospital_death). Pipeline smoke test only - not evidence. '
            'See ml/hospital_deterioration_loader.py docstring.'
        )
        _WARNED = True


def load_hourly_panel(root_dir, max_patients=None):
    """Load hospital_deterioration_hourly_panel.csv, map to the 12-feature contract.

    Returns a list of (df_pivot, label, patient_id) tuples in the same
    shape as ml.dataset.load_physionet_batch, so it CAN be fed to
    ml.dataset.create_sequences_from_physionet for a shape/pipeline smoke
    test - not because its output is comparable to real training data.
    """
    require('hospital_deterioration', 'smoke_test')
    _warn_once()

    panel_path = os.path.join(root_dir, 'hospital_deterioration_hourly_panel.csv')
    df = pd.read_csv(panel_path)

    patient_ids = df['patient_id'].unique()
    if max_patients:
        patient_ids = patient_ids[:max_patients]

    data = []
    for pid in patient_ids:
        sub = df[df['patient_id'] == pid].sort_values('hour_from_admission')
        if sub.empty:
            continue

        cols = {}
        for src, dst in {**VITALS_COLUMN_MAP, **LABS_COLUMN_MAP}.items():
            if src in sub.columns:
                cols[dst] = sub.set_index('hour_from_admission')[src]

        df_hourly = pd.DataFrame(cols)
        df_hourly.index = df_hourly.index * 60  # hours -> minutes, matches other loaders
        df_hourly.index.name = 'minutes'

        full_minute_index = range(int(df_hourly.index.min()), int(df_hourly.index.max()) + 1)
        df_pivot = df_hourly.reindex(full_minute_index).ffill()
        df_pivot['GCS'] = np.nan
        df_pivot['BUN'] = np.nan
        df_pivot['Platelets'] = np.nan
        df_pivot['Glucose'] = np.nan
        df_pivot = df_pivot.reindex(columns=SERVING_FEATURES_ORDER)

        # Patient-level label: any positive deterioration_next_12h in the stay.
        label = int((sub['deterioration_next_12h'] == 1).any()) if 'deterioration_next_12h' in sub.columns else 0

        data.append((df_pivot, label, str(pid)))

    return data


if __name__ == '__main__':
    print('ml.hospital_deterioration_loader loaded - smoke test only, see docstring')
