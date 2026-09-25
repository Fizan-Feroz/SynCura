"""Portable dataset locations for SynCura.

All training/eval scripts resolve data through here instead of hardcoding
absolute paths, so the repo works on any machine:

    set SYNCURA_DATA_ROOT=D:\\ml-data        # optional override (PowerShell)
    python -m ml.sweep_xval                   # otherwise defaults to PROJ/data

Layout expected under the root (per DATA.md):
    <root>/predicting-mortality-of-icu-patients-the-physionetcomputing-in-cardiology-challenge-2012-1.0.0/
        predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/
            set-a/  set-b_full/set-b/  Outcomes-a.txt  Outcomes-b.txt
    <root>/challenge-2019-1.0.0/training/training_setA/   (via scripts/download_data.py)
    <root>/mimic-iii-clinical-database-demo-1.4/           (extracted from the .zip)
    <root>/mimic-iv-demo-2.2/
    <root>/eicu-crd-demo-2.0.1/
    <root>/hospital-deterioration-dataset/
    <root>/kaggle-icu-mortality/
    <root>/kaggle-icu-risk-score/
    <root>/kaggle-sepsis-mimic-style/

NEW dataset helpers below are unverified against the actual on-disk layout —
I don't have access to your local data/ folder, only DATA.md's directory map.
If a folder name below doesn't match what download_data.py actually produces,
fix the constant, not the caller.
"""
import os
import zipfile

PHYSIONET2012_DIRNAME = (
    'predicting-mortality-of-icu-patients-the-physionetcomputing-in-cardiology-challenge-2012-1.0.0'
)
PHYSIONET2012_INNER = (
    'predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0'
)


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_root():
    """Absolute path to the datasets folder (override with SYNCURA_DATA_ROOT)."""
    override = os.environ.get('SYNCURA_DATA_ROOT')
    if override:
        return os.path.abspath(override)
    return os.path.join(repo_root(), 'data')


def physionet2012_root():
    """Absolute path to the extracted PhysioNet 2012 challenge folder."""
    return os.path.join(data_root(), PHYSIONET2012_DIRNAME, PHYSIONET2012_INNER)


def challenge2019_setA():
    """Absolute path to the Challenge-2019 training_setA folder."""
    return os.path.join(data_root(), 'challenge-2019-1.0.0', 'training', 'training_setA')


def _extracted_or_zip(dirname, zipname):
    """Return the extracted folder if present, else extract the zip next to it.

    MIMIC-III demo and both MIMIC-IV full-database demo zips ship as .zip in
    DATA.md's directory map (unlike MIMIC-IV's *tables* demo, which is
    already extracted). This helper makes both cases transparent to callers.
    """
    extracted = os.path.join(data_root(), dirname)
    if os.path.isdir(extracted):
        return extracted
    zpath = os.path.join(data_root(), zipname)
    if os.path.isfile(zpath):
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(data_root())
        if os.path.isdir(extracted):
            return extracted
    raise FileNotFoundError(
        f'Neither {extracted} nor {zpath} exists. Run scripts/download_data.py '
        f'or check the folder/zip name against your actual data/ contents.'
    )


def mimic3_demo_root():
    """MIMIC-III Clinical Database Demo (n=100). ETL testbed only — never train."""
    return _extracted_or_zip(
        'mimic-iii-clinical-database-demo-1.4',
        'mimic-iii-clinical-database-demo-1.4.zip',
    )


def mimic4_demo_root():
    """MIMIC-IV Clinical Database Demo (n=100, tables). ETL testbed only — never train."""
    # DATA.md lists this as already-extracted (mimic-iv-demo-2.2/), distinct
    # from the mimic-iv-clinical-database-demo-2.2.zip full-DB export below.
    return os.path.join(data_root(), 'mimic-iv-demo-2.2')


def mimic4_full_demo_zip_root():
    """The separate mimic-iv-clinical-database-demo-2.2.zip full-DB export, if used."""
    return _extracted_or_zip(
        'mimic-iv-clinical-database-demo-2.2',
        'mimic-iv-clinical-database-demo-2.2.zip',
    )


def eicu_demo_root():
    """eICU-CRD Demo 2.0.1 (~2,500 stays, 20 hospitals). External-validation testbed only."""
    return os.path.join(data_root(), 'eicu-crd-demo-2.0.1')


def hospital_deterioration_root():
    """Simulated 10k-admission benchmark. Pipeline smoke tests only — never clinical evidence."""
    return os.path.join(data_root(), 'hospital-deterioration-dataset')


def kaggle_icu_mortality_root():
    """15k-row aggregate tabular CSV. Tabular-baseline playground only — not sequence data."""
    return os.path.join(data_root(), 'kaggle-icu-mortality')


def kaggle_icu_risk_score_root():
    """1k-row snapshot CSV with a precomputed risk_score. Sanity checks only."""
    return os.path.join(data_root(), 'kaggle-icu-risk-score')


def kaggle_sepsis_mimic_style_root():
    """5k-subject wide aggregate CSV. Feature playground only — not sequence data."""
    return os.path.join(data_root(), 'kaggle-sepsis-mimic-style')
