"""Backend + ML contract tests for SynCura.

Run from the repo root:
    .\\.venv\\Scripts\\python.exe -m pytest backend/tests/ -q

These tests guard the defects fixed in the in-depth audit:
- serving contract (manifest, per-member scalers, feature order)
- causal windowing (no future leakage into early windows)
- SHAP background shape
- API health/validation/metrics schema
- training isolation (no global artifact overwrite)
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_manifest_contract():
    path = os.path.join(REPO, 'ml', 'deployed_manifest.json')
    assert os.path.exists(path), 'deployed manifest missing'
    m = json.load(open(path))
    assert len(m['members']) == 3
    assert m['arch']['hidden_size'] == 96
    assert len(m['features']) == 12
    assert m['window_minutes'] == 90
    for member in m['members']:
        assert os.path.exists(os.path.join(REPO, member['checkpoint'])), member
        assert os.path.exists(os.path.join(REPO, member['scaler'])), member
        s = json.load(open(os.path.join(REPO, member['scaler'])))
        assert len(s['mean']) == 12 and len(s['std']) == 12


def test_ensemble_best_matches_manifest():
    best = json.load(open(os.path.join(REPO, 'ml', 'ensemble_best.json')))
    assert len(best['arch']) == len(best['members']) == 3
    assert len(best.get('scalers', [])) == 3


def test_causal_windowing_no_future_leak():
    """A spike at the end of a stay must not appear in an early window."""
    from ml.dataset import create_sequences_from_physionet
    minutes = list(range(200))
    df = pd.DataFrame({
        'HR': [70.0] * 199 + [200.0],
        'RespRate': [16.0] * 200,
    }, index=pd.Index(minutes, name='minutes'))
    X, y, pids = create_sequences_from_physionet(
        [(df, 1, 'p1')],
        vital_features=['HR', 'RespRate'],
        window_minutes=90, stride=90,
        label_mode='all',
    )
    # First window covers minutes [0, 90): must not contain the 200.0 spike.
    assert X[0][:, 0].max() < 100.0, 'future spike leaked into early window'


def test_shap_background_shape():
    from ml.train_lstm import AttentionLSTMModel
    from ml.explain import compute_shap_explanation
    import torch
    torch.manual_seed(0)
    model = AttentionLSTMModel(input_size=12, hidden_size=8)
    x = np.random.randn(90, 12).astype(np.float32)
    out = compute_shap_explanation(model, x, None, n_background=3)
    assert isinstance(out, dict) and len(out) == 12


def test_api_health_and_validation():
    from fastapi.testclient import TestClient
    import backend.app as app_module
    client = TestClient(app_module.app)
    r = client.get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['model_loaded'] is True
    assert body['ensemble_members'] == 3
    # Missing patient_id must be a 422, not a 500
    r = client.post('/ingest', json={'timestamp': 1.0, 'HR': 80})
    assert r.status_code == 422


def test_version_endpoint_tracks_website_and_model():
    from fastapi.testclient import TestClient
    import backend.app as app_module
    client = TestClient(app_module.app)
    r = client.get('/version')
    assert r.status_code == 200
    body = r.json()
    package = json.load(open(os.path.join(REPO, 'frontend', 'package.json')))
    manifest = json.load(open(os.path.join(REPO, 'ml', 'deployed_manifest.json')))
    assert body['service'] == 'syncura-backend'
    assert body['website_version'] == package['version']
    assert body['model_id'] == manifest['model_id']
    assert 'git_commit' in body


def test_admin_status_reports_throughput():
    from fastapi.testclient import TestClient
    import backend.app as app_module
    client = TestClient(app_module.app)
    r = client.get('/admin/status')
    assert r.status_code == 200
    rt = r.json()['runtime']
    for key in ('ingest_count', 'last_ingest_time', 'ingest_per_min_1m',
                'ingest_per_min_5m', 'ingest_per_min_avg'):
        assert key in rt, f'missing throughput key: {key}'
    assert isinstance(rt['ingest_count'], int) and rt['ingest_count'] >= 0
    assert rt['ingest_per_min_1m'] >= 0 and rt['ingest_per_min_5m'] >= 0


def test_ingest_roundtrip():
    from fastapi.testclient import TestClient
    import backend.app as app_module
    client = TestClient(app_module.app)
    pid = 'test-patient-contract'
    r = client.post('/ingest', json={'patient_id': pid, 'timestamp': 1.0,
                                     'HR': 80, 'SpO2': 98, 'RespRate': 16,
                                     'Temp': 37.0, 'NISysABP': 120, 'NIDiasABP': 80})
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert r.json()['risk_score'] is not None
        r2 = client.get(f'/patient/{pid}')
        assert r2.status_code == 200


def test_metrics_schema():
    from fastapi.testclient import TestClient
    import backend.app as app_module
    client = TestClient(app_module.app)
    r = client.get('/metrics')
    assert r.status_code == 200
    body = r.json()
    assert 'auc' in body, 'metrics must expose flat auc key for the dashboard'


def test_training_artifacts_are_job_scoped():
    import backend.training as t
    src = open(os.path.join(REPO, 'backend', 'training.py')).read()
    # The only default model path must be job-scoped; the global serving
    # artifacts must never be written by the training worker.
    assert "config.get('model_output', f'ml/models/{job.job_id}.pt')" in src
    assert "open('ml/scaler.json', 'w')" not in src
    mgr = t.TrainingManager()
    j1 = mgr.create_job({'epochs': 1})
    j2 = mgr.create_job({'epochs': 1})
    assert j1.job_id != j2.job_id, 'job IDs must be unique'
    assert j1.metrics['train_loss'] is None, 'metrics must be scalars, not lists'
