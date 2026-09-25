"""eICU-CRD Demo 2.0.1 -> SynCura 12-feature contract. EXTERNAL-VALIDATION CODE-PATH TEST ONLY.

Per DATA.md, this ~2,500-stay demo across 20 hospitals is for testing that
the external-validation code path *runs*, not for reporting a real
external-validation *result* - it's a demo subset, not the full eICU-CRD.
ml/dataset_registry.py clears 'eicu_demo' for 'external_eval' only.

Column names below are the eICU-CRD table/column names from the standard
schema (vitalPeriodic, lab, nurseCharting). I don't have your actual demo
files to confirm these against - spot-check the first loaded patient's
values against something plausible (e.g. HR in 40-180) before trusting
this at scale.

Tables used:
  vitalPeriodic: patientunitstayid, observationoffset (minutes from unit
    admission), heartrate, respiration, sao2, temperature,
    systemicsystolic, systemicdiastolic
  lab: patientunitstayid, labresultoffset, labname, labresult
    (labname values: 'BUN', 'creatinine', 'WBC x 1000', 'platelets x 1000',
    'glucose' - exact strings vary by site; inspect labname.unique() first)
  nurseCharting: patientunitstayid, nursingchartoffset,
    nursingchartcelltypevalname == 'GCS Total', nursingchartvalue
"""
import pandas as pd

from ml.dataset_registry import require

SERVING_FEATURES_ORDER = [
    'HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
    'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose',
]

VITAL_COLUMN_MAP = {
    'heartrate': 'HR',
    'respiration': 'RespRate',
    'temperature': 'Temp',
    'systemicsystolic': 'NISysABP',
    'systemicdiastolic': 'NIDiasABP',
    'sao2': 'SpO2',
}

# labname strings to check against your actual `lab.labname.unique()` -
# eICU labnames are inconsistent across sites, this is a starting guess.
LAB_NAME_MAP = {
    'BUN': 'BUN',
    'creatinine': 'Creatinine',
    'WBC x 1000': 'WBC',
    'platelets x 1000': 'Platelets',
    'glucose': 'Glucose',
}


def load_stay(vital_periodic_df, lab_df, nurse_charting_df, patientunitstayid):
    """Build a minute-indexed df_pivot for one eICU stay, in SERVING_FEATURES_ORDER.

    Pass each table already filtered/loaded for this stay (or the whole
    small demo table - it's a demo, not full eICU-CRD).
    """
    require('eicu_demo', 'external_eval')

    cols = {}

    vp = vital_periodic_df[vital_periodic_df['patientunitstayid'] == patientunitstayid]
    if not vp.empty:
        vp = vp.set_index('observationoffset')
        for src, dst in VITAL_COLUMN_MAP.items():
            if src in vp.columns:
                cols[dst] = vp[src].groupby(level=0).mean()

    lab = lab_df[lab_df['patientunitstayid'] == patientunitstayid]
    if not lab.empty:
        for src, dst in LAB_NAME_MAP.items():
            rows = lab[lab['labname'] == src]
            if not rows.empty:
                cols[dst] = rows.groupby('labresultoffset')['labresult'].mean()

    nc = nurse_charting_df[nurse_charting_df['patientunitstayid'] == patientunitstayid]
    if not nc.empty:
        gcs_rows = nc[nc['nursingchartcelltypevalname'] == 'GCS Total']
        if not gcs_rows.empty:
            cols['GCS'] = pd.to_numeric(
                gcs_rows.set_index('nursingchartoffset')['nursingchartvalue'], errors='coerce'
            ).groupby(level=0).mean()

    if not cols:
        return pd.DataFrame(columns=SERVING_FEATURES_ORDER)

    df_pivot = pd.DataFrame(cols)
    df_pivot = df_pivot.reindex(columns=SERVING_FEATURES_ORDER)
    df_pivot.index.name = 'minutes'
    return df_pivot.sort_index()


if __name__ == '__main__':
    print('ml.eicu_loader loaded - external-validation code-path test only')
