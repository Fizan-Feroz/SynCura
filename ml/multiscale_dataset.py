# -*- coding: utf-8 -*-
"""Multi-Scale Dataset Builder for SynCura (Level 1, 2, and 3).

Extracts 3 complementary feature representations per sliding window:
1. X_seq: (N, 90, 12) - Acute 90-minute causal sequence for AttentionLSTM
2. X_static: (N, 8) - Demographic & ICU admission covariates (Age, Gender, Height, Weight, 4x ICUType)
3. X_trend: (N, 36) - 24-hour longitudinal context (24h baseline mean, delta from baseline, slope)
4. X_tab: (N, 116) - Comprehensive tabular feature vector for Gradient Boosted Trees (LightGBM)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.dataset import SERVING_FEATURES, PARAMETER_ALIASES, load_physionet_batch

STATIC_FEATURE_NAMES = ['Age', 'Gender', 'Height', 'Weight', 'ICUType_1', 'ICUType_2', 'ICUType_3', 'ICUType_4']


def extract_patient_static(df_pivot):
    """Extract and normalize static demographic covariates from a patient pivot table."""
    def get_val(param, default=np.nan):
        if param in df_pivot.columns:
            s = df_pivot[param].dropna()
            if len(s) > 0:
                v = float(s.iloc[0])
                return v if v >= 0 else np.nan
        return default

    age = get_val('Age', np.nan)
    gender = get_val('Gender', np.nan)
    height = get_val('Height', np.nan)
    weight = get_val('Weight', np.nan)
    icutype = get_val('ICUType', np.nan)

    # One-hot encode ICUType (1: CCU, 2: CSRU, 3: MICU, 4: SICU)
    icu_oh = [1.0 if icutype == k else 0.0 for k in [1.0, 2.0, 3.0, 4.0]]
    if not any(icu_oh):
        icu_oh = [0.25, 0.25, 0.25, 0.25]  # Uniform prior if missing

    return np.array([age, gender, height, weight] + icu_oh, dtype=np.float32)


def create_multiscale_sequences(data_list, vital_features=None, window_minutes=90, stride=30,
                                horizon_hours=12.0):
    """Generate multi-scale representations (Seq, Static, Trend, Tabular) causally per window."""
    if vital_features is None:
        vital_features = list(SERVING_FEATURES)

    X_seq_list = []
    X_static_list = []
    X_trend_list = []
    X_tab_list = []
    y_list = []
    pids_list = []

    for df_pivot, label, patient_id in data_list:
        static_vec = extract_patient_static(df_pivot)

        # Resolve vitals and aliases
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

        minute_index = range(int(df_pivot.index.min()), int(df_pivot.index.max()) + 1)
        df_vitals = pd.DataFrame({f: resolved[f] for f in available}, index=df_pivot.index)
        df_raw = df_vitals.reindex(index=minute_index, columns=vital_features)

        seq_len = window_minutes
        if len(df_raw) < seq_len:
            continue

        # Proximity cutoff: last horizon_hours
        cutoff = max(seq_len, len(df_raw) - int(horizon_hours * 60))
        starts = list(range(cutoff, len(df_raw), stride))
        if not starts:
            continue

        df_carry = df_raw.ffill()

        for i in starts:
            # 1. Acute Window: [i - seq_len, i]
            window_raw = df_raw.iloc[i - seq_len:i]
            window_carried = df_carry.iloc[i - seq_len:i].values  # (90, 12)

            # 2. Longitudinal 24h Context: [max(0, i - 1440), i]
            long_start = max(0, i - 1440)
            long_carried = df_carry.iloc[long_start:i]

            # Compute Trend Features (36 dims: baseline mean, delta, slope per feature)
            trend_features = []
            for col_idx, col_name in enumerate(vital_features):
                base_series = long_carried[col_name].dropna()
                win_series = window_raw[col_name].dropna()

                b_mean = base_series.mean() if len(base_series) > 0 else np.nan
                w_mean = win_series.mean() if len(win_series) > 0 else b_mean
                delta = w_mean - b_mean if (not np.isnan(w_mean) and not np.isnan(b_mean)) else 0.0

                # Slope over 24h
                if len(base_series) >= 2:
                    slope = (base_series.iloc[-1] - base_series.iloc[0]) / max(1.0, len(base_series))
                else:
                    slope = 0.0

                trend_features.extend([b_mean, delta, slope])

            trend_vec = np.array(trend_features, dtype=np.float32)

            # 3. Tabular Summary Features for LightGBM (116 dims)
            # 6 stats per vital (mean, min, max, std, last, diff) = 72 stats
            acute_stats = []
            for col_idx in range(window_carried.shape[1]):
                col_vals = window_carried[:, col_idx]
                col_valid = col_vals[~np.isnan(col_vals)]
                if len(col_valid) > 0:
                    c_mean = np.mean(col_valid)
                    c_min = np.min(col_valid)
                    c_max = np.max(col_valid)
                    c_std = np.std(col_valid)
                    c_last = col_valid[-1]
                    c_diff = col_valid[-1] - col_valid[0]
                else:
                    c_mean = c_min = c_max = c_std = c_last = c_diff = np.nan
                acute_stats.extend([c_mean, c_min, c_max, c_std, c_last, c_diff])

            tab_vec = np.concatenate([static_vec, trend_vec, np.array(acute_stats, dtype=np.float32)])

            X_seq_list.append(window_carried)
            X_static_list.append(static_vec)
            X_trend_list.append(trend_vec)
            X_tab_list.append(tab_vec)
            y_list.append(label)
            pids_list.append(patient_id)

    return (
        np.array(X_seq_list, dtype=np.float32),
        np.array(X_static_list, dtype=np.float32),
        np.array(X_trend_list, dtype=np.float32),
        np.array(X_tab_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(pids_list)
    )
