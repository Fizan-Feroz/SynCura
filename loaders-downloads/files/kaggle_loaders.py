"""Kaggle tabular snapshots. NOT SEQUENCE DATA - NOT USABLE BY THE LSTM.

Per DATA.md, all three of these are aggregate/snapshot tables (one row per
patient, not a time series), so they can't feed ml.dataset's windowing at
all - there's no time axis to window over. ml/dataset_registry.py clears
all three for 'tabular_baseline' only.

Their only legitimate use: a quick non-LSTM baseline (logistic regression
/ gradient boosting) on aggregate features, as a sanity check that "does a
much simpler model do obviously-reasonable things on this kind of data" -
not as a component of the AttentionLSTM pipeline, and not comparable
apples-to-apples with the LSTM's AUC (different data, different features,
different label definitions in some cases - check each one).
"""
import os
import pandas as pd

from ml.dataset_registry import require


def load_kaggle_icu_mortality(root_dir):
    """15k-row aggregate table (mean/std/max/min vitals + mortality_label, APACHE/SOFA)."""
    require('kaggle_icu_mortality', 'tabular_baseline')
    path = os.path.join(root_dir, 'kaggle-icu-mortality.csv')  # confirm actual filename
    return pd.read_csv(path)


def load_kaggle_icu_risk_score(root_dir):
    """1k-row snapshot table with a precomputed risk_score column."""
    require('kaggle_icu_risk_score', 'tabular_baseline')
    path = os.path.join(root_dir, 'kaggle-icu-risk-score.csv')  # confirm actual filename
    return pd.read_csv(path)


def load_kaggle_sepsis_mimic_style(root_dir):
    """5k-subject wide aggregate table (77 columns)."""
    require('kaggle_sepsis_mimic_style', 'tabular_baseline')
    path = os.path.join(root_dir, 'kaggle-sepsis-mimic-style.csv')  # confirm actual filename
    return pd.read_csv(path)


if __name__ == '__main__':
    print('ml.kaggle_loaders loaded - tabular baselines only, not for the LSTM')
