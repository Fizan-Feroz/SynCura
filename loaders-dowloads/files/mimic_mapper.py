"""MIMIC-III/IV demo -> SynCura 12-feature contract. ETL DEBUG TOOL ONLY.

Per DATA.md and ml/dataset_registry.py, the demo subsets (n=100 each) are
never cleared for 'train' - this module exists to build and debug the
MIMIC->12-feature mapping against small, fast-to-load data, not to produce
training examples. Deliberately, NOTHING in this file produces
(df_pivot, label, patient_id) tuples shaped for ml.dataset's windowing/
training code (contrast with ml/challenge2019_to_features.py, which does)
- that absence is the guard. If you later add a batch loader here for
convenience, call `dataset_registry.require('mimic4_demo', 'train')` at
its entry point first; it will raise (mimic4_demo has an empty
capabilities set), which is the point.

*** ITEM IDs ARE RESOLVED BY LABEL TEXT, NOT HARDCODED ***
MIMIC-III (CareVue/MetaVision) and MIMIC-IV use different numeric itemids
for the same concept, and I don't have your actual d_items.csv/
d_labitems.csv to look them up against. Hardcoding itemid numbers from
memory would be a real risk of silently pulling the wrong concept. Instead,
`resolve_itemids()` fuzzy-matches each feature against the LABEL column of
d_items/d_labitems using keyword patterns below. Print and manually check
`resolve_itemids()`'s output against your actual dictionaries before
trusting any pivoted data - text matching can grab the wrong item
(e.g. "Heart Rate" vs "Heart Rate Alarm - High").

*** GCS IS THE ONE CONFIRMED EXCEPTION ***
DATA.md already confirms MIMIC-IV's GCS itemids precisely: Eye/Motor/Verbal
= 220739/223900/223901, summed to total. Those are hardcoded below since
they're already verified, unlike everything else in this file.
"""
import pandas as pd

from ml.dataset_registry import require


def raise_if_used_for_training():
    """Call this from any future batch/training loader added to this file.

    Always raises (mimic4_demo has an empty capabilities set in
    ml/dataset_registry.py) - this is a deliberate hard stop, not a
    placeholder to fill in later.
    """
    require('mimic4_demo', 'train')

# Keyword patterns (case-insensitive substring match) for resolving each
# SynCura feature against d_items.LABEL / d_labitems.LABEL. First match
# wins; inspect resolve_itemids()'s output and refine these if wrong.
CHARTEVENTS_LABEL_PATTERNS = {
    'HR': ['heart rate'],
    'RespRate': ['respiratory rate'],
    'Temp': ['temperature f', 'temperature c'],  # check units per matched item
    'NISysABP': ['non invasive blood pressure systolic', 'nbp systolic', 'nbp [systolic]'],
    'NIDiasABP': ['non invasive blood pressure diastolic', 'nbp diastolic', 'nbp [diastolic]'],
    'SpO2': ['o2 saturation pulseoxymetry', 'spo2'],
}
LABEVENTS_LABEL_PATTERNS = {
    'BUN': ['urea nitrogen'],
    'Creatinine': ['creatinine'],
    'WBC': ['white blood cells'],
    'Platelets': ['platelet count'],
    'Glucose': ['glucose'],
}
# Confirmed by DATA.md - MIMIC-IV only. MIMIC-III's GCS itemids are
# different (older CareVue scheme) and are NOT filled in here - resolve
# them the same way as everything else via CHARTEVENTS_LABEL_PATTERNS
# with a 'GCS' entry (omitted above deliberately, since summing three
# components needs a different code path than a single-item lookup).
MIMIC_IV_GCS_ITEMIDS = {'eye': 220739, 'motor': 223900, 'verbal': 223901}


def resolve_itemids(d_items_df, d_labitems_df):
    """Return {feature: [itemid, ...]} by matching LABEL text. INSPECT THIS OUTPUT.

    d_items_df / d_labitems_df: the D_ITEMS.csv / D_LABITEMS.csv tables,
    loaded with pandas.read_csv, columns must include ITEMID and LABEL.

    This function only resolves itemids for later manual inspection - it
    does not load patient data and is safe to run freely (ETL debugging).
    """
    resolved = {}
    labels_lower = d_items_df['LABEL'].astype(str).str.lower()
    for feature, patterns in CHARTEVENTS_LABEL_PATTERNS.items():
        mask = pd.Series(False, index=d_items_df.index)
        for p in patterns:
            mask |= labels_lower.str.contains(p, na=False)
        resolved[feature] = d_items_df.loc[mask, 'ITEMID'].tolist()

    lab_labels_lower = d_labitems_df['LABEL'].astype(str).str.lower()
    for feature, patterns in LABEVENTS_LABEL_PATTERNS.items():
        mask = pd.Series(False, index=d_labitems_df.index)
        for p in patterns:
            mask |= lab_labels_lower.str.contains(p, na=False)
        resolved[feature] = d_labitems_df.loc[mask, 'ITEMID'].tolist()

    return resolved


def load_chartevents_for_admission(chartevents_df, hadm_id, itemid_map):
    """Pivot CHARTEVENTS rows for one admission into a minute-indexed df_pivot.

    chartevents_df: CHARTEVENTS.csv already filtered/loaded for speed - the
    full table is large even in the demo; filter by HADM_ID upstream if
    reading from disk rather than passing the whole table here.
    itemid_map: output of resolve_itemids() (chartevents half only).

    This produces a per-admission df_pivot for ETL debugging/inspection
    only. It intentionally has no `label` or `patient_id` bundling and no
    batch/loop wrapper - do not build one that feeds this into
    ml.dataset.create_sequences_from_physionet without first getting a
    real sign-off on training on MIMIC (see module docstring).
    """
    sub = chartevents_df[chartevents_df['HADM_ID'] == hadm_id].copy()
    sub['CHARTTIME'] = pd.to_datetime(sub['CHARTTIME'])
    t0 = sub['CHARTTIME'].min()
    sub['minutes'] = (sub['CHARTTIME'] - t0).dt.total_seconds() // 60

    cols = {}
    for feature, itemids in itemid_map.items():
        if not itemids:
            continue
        rows = sub[sub['ITEMID'].isin(itemids)]
        if rows.empty:
            continue
        series = rows.groupby('minutes')['VALUENUM'].mean()
        cols[feature] = series

    # GCS: sum the three confirmed MIMIC-IV component itemids, when present.
    gcs_ids = list(MIMIC_IV_GCS_ITEMIDS.values())
    gcs_rows = sub[sub['ITEMID'].isin(gcs_ids)]
    if not gcs_rows.empty:
        gcs_by_minute_component = gcs_rows.pivot_table(
            index='minutes', columns='ITEMID', values='VALUENUM', aggfunc='mean'
        )
        cols['GCS'] = gcs_by_minute_component.sum(axis=1, min_count=3)  # NaN unless all 3 present

    if not cols:
        return pd.DataFrame()
    df_pivot = pd.DataFrame(cols)
    df_pivot.index.name = 'minutes'
    return df_pivot.sort_index()


if __name__ == '__main__':
    print('ml.mimic_mapper loaded - ETL/debug tool only, never produces training data')
