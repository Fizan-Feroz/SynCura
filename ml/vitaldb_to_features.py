"""VitalDB (perioperative & ICU) -> SynCura 12-feature contract.

Converts VitalDB clinical cases and laboratory panels stored in PROJ/data/vitaldb/
into (df_pivot, label, patient_id) tuples compatible with ml.dataset.create_sequences_from_physionet.

Features mapped from VitalDB:
- Labs: BUN ('bun'), Creatinine ('cr'), WBC ('wbc'), Platelets ('plt'), Glucose ('gluc')
- Label: 'death_inhosp' (in-hospital mortality: 0 or 1, matching PhysioNet In-hospital_death)
- Patient metadata: caseid, age, sex, icu_days
"""
import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.paths import vitaldb_root
from ml.dataset import SERVING_FEATURES

# Mapping from VitalDB lab codes to SynCura canonical SERVING_FEATURES
LAB_MAP = {
    'bun': 'BUN',
    'cr': 'Creatinine',
    'wbc': 'WBC',
    'plt': 'Platelets',
    'gluc': 'Glucose',
}


def load_vitaldb_batch(vitaldb_dir=None, icu_only=True, max_cases=None, min_labs=3):
    """Load VitalDB cases and lab panels from local PROJ/data/vitaldb/.

    Args:
        vitaldb_dir: path to vitaldb folder (defaults to PROJ/data/vitaldb)
        icu_only: if True, filter to patients admitted to ICU (icu_days > 0)
        max_cases: maximum number of cases to return
        min_labs: minimum number of lab records required for a case
    Returns:
        List of (df_pivot, label, patient_id) tuples matching ml.dataset
    """
    vdir = vitaldb_dir or vitaldb_root()
    cases_path = os.path.join(vdir, 'cases.csv')
    labs_path = os.path.join(vdir, 'labs.csv')

    if not os.path.exists(cases_path) or not os.path.exists(labs_path):
        raise FileNotFoundError(
            f"VitalDB files not found in {vdir}. Run 'python scripts/download_vitaldb.py' first."
        )

    print(f"Loading VitalDB cases from {cases_path}...")
    df_cases = pd.read_csv(cases_path)
    if icu_only and 'icu_days' in df_cases.columns:
        df_cases = df_cases[df_cases['icu_days'] > 0]

    # Filter to cases with known mortality label
    df_cases = df_cases.dropna(subset=['death_inhosp'])
    case_ids = set(df_cases['caseid'])

    print(f"Loading VitalDB labs from {labs_path}...")
    # Read labs filtered to the target features and case IDs
    df_labs = pd.read_csv(labs_path)
    df_labs = df_labs[df_labs['name'].isin(LAB_MAP.keys()) & df_labs['caseid'].isin(case_ids)]

    # Group labs by caseid
    grouped_labs = df_labs.groupby('caseid')
    outcomes_dict = dict(zip(df_cases['caseid'], df_cases['death_inhosp'].astype(int)))

    data = []
    skipped = 0

    valid_caseids = sorted(list(grouped_labs.groups.keys()))
    if max_cases:
        valid_caseids = valid_caseids[:max_cases]

    for cid in valid_caseids:
        clabs = grouped_labs.get_group(cid).copy()
        if len(clabs) < min_labs:
            skipped += 1
            continue

        label = outcomes_dict[cid]
        patient_id = f"vitaldb_{cid}"

        # clabs['dt'] is in seconds from casestart; convert to minutes
        clabs['minutes'] = (clabs['dt'] / 60.0).round().astype(int)
        # Shift so the earliest record starts at 0 minutes
        clabs['minutes'] = clabs['minutes'] - clabs['minutes'].min()
        clabs['feature'] = clabs['name'].map(LAB_MAP)

        # Convert result column to float
        clabs['result'] = pd.to_numeric(clabs['result'], errors='coerce')
        clabs = clabs.dropna(subset=['result'])
        if clabs.empty:
            skipped += 1
            continue

        # Pivot into minutes x features
        piv = clabs.pivot_table(index='minutes', columns='feature', values='result', aggfunc='mean')
        
        # Enforce canonical SERVING_FEATURES (vitals not present in lab file remain NaN for forward imputation)
        for feat in SERVING_FEATURES:
            if feat not in piv.columns:
                piv[feat] = np.nan
        piv = piv.reindex(columns=SERVING_FEATURES)

        # Reindex to full minute grid
        full_index = range(int(piv.index.min()), int(piv.index.max()) + 1)
        per_minute = piv.reindex(full_index).ffill()

        data.append((per_minute, label, patient_id))

    print(f"Loaded {len(data)} VitalDB patients ({skipped} skipped due to sparse observations).")
    return data


if __name__ == '__main__':
    batch = load_vitaldb_batch(max_cases=10)
    print(f"Sample loaded: {len(batch)} cases. First patient: {batch[0][2]}, label: {batch[0][1]}, shape: {batch[0][0].shape}")
