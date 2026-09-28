"""Dataset capability and Challenge 2019 mapping contracts.

Run from the repo root:
    python -m pytest backend/tests/test_dataset_policy.py -q
"""
import pytest

from ml.challenge2019_to_features import COLUMN_MAP
from ml.dataset import SERVING_FEATURES
from ml.dataset_registry import DatasetPolicyError, require


def test_challenge2019_mapping_covers_all_non_gcs_features():
    assert set(COLUMN_MAP.values()) == set(SERVING_FEATURES) - {'GCS'}


def test_challenge2019_is_trainable_but_not_deployable():
    assert require('challenge2019', 'train')['kind'] == 'real'
    with pytest.raises(DatasetPolicyError, match='not cleared for "deploy_train"'):
        require('challenge2019', 'deploy_train')


def test_demo_and_simulated_datasets_are_not_trainable():
    with pytest.raises(DatasetPolicyError, match='not cleared for "train"'):
        require('mimic4_demo', 'train')
    with pytest.raises(DatasetPolicyError, match='not cleared for "train"'):
        require('hospital_deterioration', 'train')


def test_unknown_dataset_is_rejected():
    with pytest.raises(DatasetPolicyError, match='Unknown dataset'):
        require('not-a-dataset', 'train')
