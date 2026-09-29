"""Gap-model detection contracts for ml/eval_locked.py (pure functions only)."""
from ml.eval_locked import needs_gap_channels


def _scaler(n_stats, gap_flag=False):
    return {
        'features': ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
                     'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose'],
        'mean': [0.0] * n_stats,
        'std': [1.0] * n_stats,
        **({'gap_channels': True} if gap_flag else {}),
    }


def test_baseline_scaler_needs_no_gap():
    assert needs_gap_channels(_scaler(12)) is False


def test_gap_scaler_detected_by_stat_width():
    # Gap scalers keep 12 feature names but carry 24 stats — the old
    # len(features) > 12 check never fired on these.
    assert needs_gap_channels(_scaler(24)) is True


def test_gap_scaler_detected_by_flag():
    assert needs_gap_channels(_scaler(12, gap_flag=True)) is True


def test_non_dict_needs_no_gap():
    assert needs_gap_channels(None) is False
    assert needs_gap_channels({}) is False
