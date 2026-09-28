"""Single source of truth for what each dataset is allowed to be used for.

This exists because DATA.md's rules ("never train on the MIMIC demos",
"never report hospital-deterioration-dataset as clinical evidence") were
previously only enforced by people reading the docs. This module makes
them enforceable in code: anything that trains or deploys a model can
call `require(dataset, 'train')` / `require(dataset, 'deploy')` and get a
hard error instead of relying on nobody making a mistake.

Keep this in sync with DATA.md by hand — there's no automatic check that
they agree, so if you change one, change the other.
"""

REAL = 'real'
SIMULATED = 'simulated'
CREDENTIALED_MISSING = 'credentialed_missing'

# capabilities: which operations each dataset is cleared for.
#   'train'        - may contribute training examples to a model
#   'deploy_train'  - may contribute to a model that gets --deploy'd
#                    (subset of 'train'; PhysioNet 2012 only, per the
#                    serving contract in ml/train.py)
#   'eval'          - may be used to report a held-out metric
#   'external_eval' - may be used as an external-validation code-path test
#                    (not a real external-validation *result*, just a test
#                    that the code path runs against a different source)
#   'smoke_test'    - pipeline/benchmark-harness testing only, results are
#                    never reportable as evidence of anything clinical
#   'tabular_baseline' - not sequence data, unusable by the LSTM; only
#                    useful for a non-LSTM tabular sanity-check model

DATASETS = {
    'physionet2012': {
        'kind': REAL,
        'capabilities': {'train', 'deploy_train', 'eval'},
        'note': 'The only dataset currently wired into ml/dataset.py + ml/train.py.',
    },
    'challenge2019': {
        'kind': REAL,
        'capabilities': {'train', 'eval'},
        'note': (
            'Real data, but SepsisLabel (sepsis onset) is not the same target '
            'as In-hospital_death. Do not pool its labels with PhysioNet 2012 '
            'windows into one y vector without deliberately handling the '
            'label mismatch (see ml/challenge2019_to_features.py docstring). '
            'Not cleared for deploy_train until that is resolved.'
        ),
    },
    'mimic3_demo': {
        'kind': REAL,
        'capabilities': set(),  # deliberately empty: ETL testbed only
        'note': 'n=100. Build/debug the MIMIC->12-feature mapper here. Never train.',
    },
    'mimic4_demo': {
        'kind': REAL,
        'capabilities': set(),
        'note': 'n=100. Same as mimic3_demo. Never train.',
    },
    'eicu_demo': {
        'kind': REAL,
        'capabilities': {'external_eval'},
        'note': (
            '~2,500 stays, 20 hospitals, but a demo subset - a good score '
            'here is evidence the external-validation code path works, not '
            'evidence of real external validation. Full eICU-CRD is what '
            'would give an actual external-validation result.'
        ),
    },
    'hospital_deterioration': {
        'kind': SIMULATED,
        'capabilities': {'smoke_test'},
        'note': (
            'Simulated, not real patients. Its 12-feature coverage is also '
            'incomplete: no GCS, BUN, Platelets, or Glucose exist anywhere '
            'in its schema, so those 4 of 12 model input channels are '
            'never exercised by this dataset. Never report metrics from '
            'this as clinical evidence, per DATA.md.'
        ),
    },
    'kaggle_icu_mortality': {
        'kind': REAL,
        'capabilities': {'tabular_baseline'},
        'note': 'Aggregate rows, not time series. Unusable by the LSTM.',
    },
    'kaggle_icu_risk_score': {
        'kind': REAL,
        'capabilities': {'tabular_baseline'},
        'note': '1k rows, precomputed risk_score. Sanity checks only.',
    },
    'kaggle_sepsis_mimic_style': {
        'kind': REAL,
        'capabilities': {'tabular_baseline'},
        'note': 'Aggregate rows, not time series. Unusable by the LSTM.',
    },
    'mimic4_ed_demo': {
        'kind': REAL,
        'capabilities': set(),
        'note': 'Skipped entirely per DATA.md: wrong care setting, no ICU mortality label.',
    },
    'mimic4_full': {
        'kind': CREDENTIALED_MISSING,
        'capabilities': set(),
        'note': 'Requires PhysioNet login + CITI + signed DUA. Never redistribute via team Drive.',
    },
    'eicu_crd_full': {
        'kind': CREDENTIALED_MISSING,
        'capabilities': set(),
        'note': 'Same access requirements as mimic4_full.',
    },
    'hirid': {
        'kind': CREDENTIALED_MISSING,
        'capabilities': set(),
        'note': 'Same access requirements as mimic4_full.',
    },
}


class DatasetPolicyError(RuntimeError):
    pass


def require(dataset_name, capability):
    """Raise if `dataset_name` isn't cleared for `capability`. Returns the entry if it is."""
    entry = DATASETS.get(dataset_name)
    if entry is None:
        raise DatasetPolicyError(
            f'Unknown dataset "{dataset_name}" - add it to ml/dataset_registry.py first.'
        )
    if capability not in entry['capabilities']:
        raise DatasetPolicyError(
            f'"{dataset_name}" is not cleared for "{capability}". '
            f'Reason: {entry["note"]}'
        )
    return entry


def is_deploy_eligible(dataset_name):
    return 'deploy_train' in DATASETS.get(dataset_name, {}).get('capabilities', set())
