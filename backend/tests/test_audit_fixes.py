"""Regression tests for the serving/training audit fixes.

Run from the repo root:
    python -m pytest backend/tests/ -q

Each section guards one defect:
  1. carry_forward: serving mirrors training's causal ffill windows
  2. serving preprocessing: missing features / short history no longer map to raw 0
  3. minute binning: the 90-step window is 90 MINUTES, not 90 readings, and
     values measured before the window are carried in (sparse labs)
  4. payload aliases: lowercase device keys (ESP32 "hr"/"spo2") are scored
  5. ml/train.py: defaults match serving; deploy is opt-in and never touches ml/scaler.json
  6. single-model fallback: previously unreachable
  7. backend/training.py: all-NaN feature, created_at, active_job reset
  8. db: one row per patient in the top list; index + WAL
  9. explain endpoint: SHAP sees exactly the scored input
 10. CORS, alert cooldown thread safety, stale-patient eviction, batching
"""
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import types

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
            'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose']
HEALTHY_VITALS = {'HR': 75, 'RespRate': 16, 'Temp': 36.8, 'NISysABP': 120, 'NIDiasABP': 70, 'SpO2': 98}
SICK = {'HR': 135, 'RespRate': 32, 'Temp': 39.5, 'NISysABP': 82, 'NIDiasABP': 45, 'SpO2': 86, 'GCS': 8}


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    # Model/scaler/manifest paths are repo-relative.
    monkeypatch.chdir(REPO)


@pytest.fixture(scope='module')
def engine():
    os.chdir(REPO)
    from backend.inference import RiskScoreEngine
    return RiskScoreEngine()


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    import backend.db as db
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'vitals.db'))
    db.init_db()
    return db


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _denormalize(engine, X):
    mean, std = engine.member_scalers[0]
    return X * std + mean


def _write_physionet(root, n_patients=10, minutes=180, missing=()):
    """Tiny PhysioNet-2012-format dataset: one CSV per patient + Outcomes file."""
    set_dir = root / 'set-a'
    set_dir.mkdir()
    rng = np.random.default_rng(0)
    outcomes = ['RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death']
    base = {'HR': 85, 'RespRate': 19, 'Temp': 37, 'NISysABP': 118, 'NIDiasABP': 58, 'SaO2': 96,
            'GCS': 12, 'BUN': 25, 'Creatinine': 1.3, 'WBC': 12, 'Platelets': 190, 'Glucose': 130}
    for p in range(n_patients):
        rid = 140000 + p
        label = p % 2
        lines = ['Time,Parameter,Value', f'00:00,RecordID,{rid}']
        for m in range(0, minutes, 5):
            for param, v in base.items():
                if param in missing:
                    continue
                shift = 15 * label if param == 'HR' else 0
                lines.append(f'{m // 60:02d}:{m % 60:02d},{param},{v + shift + rng.normal():.2f}')
        (set_dir / f'{rid}.txt').write_text('\n'.join(lines))
        outcomes.append(f'{rid},10,5,8,-1,{label}')
    out = root / 'Outcomes-a.txt'
    out.write_text('\n'.join(outcomes))
    return str(set_dir), str(out)


# ------------------------------------------------------------- 1. carry_forward

def test_carry_forward_matches_pandas_ffill():
    from ml.dataset import carry_forward
    rng = np.random.default_rng(1)
    for _ in range(200):
        W = rng.normal(size=(90, 12))
        W[rng.random((90, 12)) < rng.random()] = np.nan
        initial = rng.normal(size=12)
        initial[rng.random(12) < 0.5] = np.nan
        ref = pd.DataFrame(np.vstack([initial, W])).ffill().values[1:]
        out = carry_forward(W, initial)
        assert np.array_equal(np.isnan(out), np.isnan(ref))
        np.testing.assert_array_equal(np.nan_to_num(out), np.nan_to_num(ref))


def test_carry_forward_is_causal_step_fill():
    from ml.dataset import carry_forward
    W = np.array([[np.nan], [10.0], [np.nan], [40.0], [np.nan]])
    out = carry_forward(W)[:, 0]
    assert np.isnan(out[0]), 'no backward fill: nothing observed yet'
    np.testing.assert_array_equal(out[1:], [10, 10, 40, 40])
    np.testing.assert_array_equal(carry_forward(W, initial=[5.0])[:, 0], [5, 10, 10, 40, 40])


def test_carry_forward_does_not_mutate_inputs():
    from ml.dataset import carry_forward
    W = np.array([[1.0, np.nan], [np.nan, np.nan]])
    initial = np.array([np.nan, 3.0])
    out = carry_forward(W, initial)
    assert np.isnan(W[1, 0]) and np.isnan(initial[0])
    np.testing.assert_array_equal(out, [[1, 3], [1, 3]])


def test_training_windows_carry_pre_window_values():
    """Training windows keep a lab measured before the window (causal ffill)."""
    from ml.dataset import create_sequences_from_physionet, carry_forward
    idx = pd.Index(range(300), name='minutes')
    bun = np.full(300, np.nan)
    bun[[10, 250]] = [20, 40]
    df = pd.DataFrame({'HR': 80.0, 'BUN': bun}, index=idx)
    X, _, _ = create_sequences_from_physionet([(df, 0, 'p')], ['HR', 'BUN'],
                                              window_minutes=90, label_mode='last')
    assert X[0][0, 1] == 20, 'BUN from minute 10 must carry into the 210-299 window'
    raw = df.values[-90:]
    np.testing.assert_array_equal(X[0], carry_forward(raw, initial=[80.0, 20.0]))


# ------------------------------------------------ 2. serving preprocessing (bug 1)

def test_missing_features_normalize_to_zero(engine):
    engine.add_vital('pp-missing', {'timestamp': 0, 'HR': 120})
    X = engine.get_model_input('pp-missing')
    assert X.shape == (90, 12)
    others = [i for i, f in enumerate(FEATURES) if f != 'HR']
    assert np.all(X[:, others] == 0), 'unobserved features must be the train mean (z=0)'
    assert np.all(X[:-1, 0] == 0), 'minutes before the first reading are unknown (z=0)'
    np.testing.assert_allclose(_denormalize(engine, X)[-1, 0], 120, rtol=1e-5)


def test_short_history_is_not_zero_padded(engine):
    engine.add_vital('pp-short', dict(HEALTHY_VITALS, timestamp=0))
    X = engine.get_model_input('pp-short')
    assert np.abs(X).max() < 5, 'raw-zero padding produced extreme z-scores'
    # Before the first reading nothing is known: train mean (z=0), like training.
    assert np.all(X[:-1] == 0)
    np.testing.assert_allclose(_denormalize(engine, X)[-1, :6],
                               [HEALTHY_VITALS[f] for f in FEATURES[:6]], rtol=1e-5)


def test_healthy_vitals_only_score_is_stable(engine):
    # Regression: this climbed 11 -> 58 as missing labs (raw 0) filled the buffer.
    # Now it settles as real history replaces "unknown" minutes, then holds.
    scores = [engine.add_vital('pp-healthy', dict(HEALTHY_VITALS, timestamp=60 * i)) for i in range(180)]
    assert max(scores) < 45
    assert max(scores[90:]) - min(scores[90:]) == 0, 'full window of identical vitals must be flat'
    assert scores[-1] <= scores[0], 'a healthy history must not raise risk'


def test_sick_scores_higher_than_healthy(engine):
    healthy = [engine.add_vital('pp-h2', dict(HEALTHY_VITALS, timestamp=60 * i)) for i in range(90)][-1]
    sick = [engine.add_vital('pp-sick', dict(SICK, timestamp=60 * i)) for i in range(90)][-1]
    assert sick > healthy + 15


def test_serving_input_matches_training_window(engine):
    """Same raw readings -> identical model input in training and serving."""
    from ml.dataset import create_sequences_from_physionet
    rng = np.random.default_rng(3)
    # Stay spans 300 minutes, so the scored window must carry values in from
    # minutes 0-209 exactly like training's whole-stay ffill.
    minutes = sorted(rng.choice(np.arange(1, 300), size=60, replace=False).tolist() + [0])
    rows = {}
    for m in minutes:
        rows[m] = {f: float(rng.normal(50, 10)) for f in FEATURES if rng.random() < 0.3 and f != 'Glucose'}
    df = pd.DataFrame.from_dict(rows, orient='index').reindex(columns=FEATURES)
    df.index.name = 'minutes'
    df = df.dropna(axis=1, how='all')

    X_train, _, _ = create_sequences_from_physionet([(df, 0, 'p')], FEATURES,
                                                    window_minutes=90, label_mode='last')
    mean, std = engine.member_scalers[0]
    expected = ((np.where(np.isnan(X_train[0]), mean, X_train[0]) - mean) / std).astype(np.float32)

    order = list(minutes)
    for m in order:
        engine.add_vital('pp-parity', dict(rows[m], timestamp=m * 60))
    np.testing.assert_allclose(engine.get_model_input('pp-parity'), expected, atol=1e-5)
    # Same result when readings arrive out of order.
    rng.shuffle(order)
    for m in order:
        engine.add_vital('pp-parity-shuffled', dict(rows[m], timestamp=m * 60))
    np.testing.assert_allclose(engine.get_model_input('pp-parity-shuffled'), expected, atol=1e-5)


def test_score_is_logit_average_of_members(engine):
    import math
    import torch
    for i in range(5):
        score = engine.add_vital('pp-avg', dict(SICK, timestamp=60 * i))
    with engine.lock:
        window = engine._window_locked('pp-avg')
    with torch.no_grad():
        logits = [m(torch.from_numpy(X[None])).item() for m, X in engine._member_inputs(window)]
    assert len(logits) == 3
    assert score == round(100 / (1 + math.exp(-sum(logits) / 3)))


# ------------------------------------------------------ 3. minute binning (bug 3)

def test_same_minute_readings_are_averaged(engine):
    engine.add_vital('mb-avg', {'timestamp': 600, 'HR': 60})
    engine.add_vital('mb-avg', {'timestamp': 630, 'HR': 80})
    X = _denormalize(engine, engine.get_model_input('mb-avg'))
    np.testing.assert_allclose(X[-1, 0], 70, rtol=1e-5)


def test_window_spans_minutes_not_readings(engine):
    # 200 readings 10 s apart = 33 minutes, well under the 90-minute window.
    for i in range(200):
        engine.add_vital('mb-span', dict(HEALTHY_VITALS, timestamp=10 * i))
    assert len(engine.vital_buffer['mb-span']) == 34
    # Two hours later the old minutes fall out of the window entirely.
    engine.add_vital('mb-span', dict(HEALTHY_VITALS, timestamp=10 * 199 + 7200))
    assert len(engine.vital_buffer['mb-span']) == 1


def test_gaps_carry_last_value_not_interpolated(engine):
    engine.add_vital('mb-step', {'timestamp': 0, 'HR': 60})
    engine.add_vital('mb-step', {'timestamp': 89 * 60, 'HR': 150})
    hr = _denormalize(engine, engine.get_model_input('mb-step'))[:, 0]
    np.testing.assert_allclose(hr, [60] * 89 + [150], rtol=1e-5)


def test_lab_measured_before_window_is_carried(engine):
    # Regression: a lab measured 3 h ago fell back to the train mean.
    engine.add_vital('mb-lab', {'timestamp': 0, 'BUN': 80, 'HR': 90})
    for i in range(1, 181):
        engine.add_vital('mb-lab', {'timestamp': i * 60, 'HR': 90})
    X = _denormalize(engine, engine.get_model_input('mb-lab'))
    bun = FEATURES.index('BUN')
    np.testing.assert_allclose(X[:, bun], 80, rtol=1e-5)
    assert 0 not in engine.vital_buffer['mb-lab'], 'minute 0 must have left the window'


def test_late_reading_older_than_window_only_updates_carry(engine):
    engine.add_vital('mb-late', {'timestamp': 200 * 60, 'HR': 90})
    engine.add_vital('mb-late', {'timestamp': 10 * 60, 'Glucose': 300})   # 190 min late
    engine.add_vital('mb-late', {'timestamp': 5 * 60, 'Glucose': 100})    # even older: ignored
    assert set(engine.vital_buffer['mb-late']) == {200}
    X = _denormalize(engine, engine.get_model_input('mb-late'))
    np.testing.assert_allclose(X[:, FEATURES.index('Glucose')], 300, rtol=1e-5)


@pytest.mark.parametrize('ts', [None, 'not-a-number', float('nan'), float('inf')])
def test_bad_timestamp_falls_back_to_now(engine, ts):
    score = engine.add_vital('mb-badts', dict(HEALTHY_VITALS, timestamp=ts))
    assert isinstance(score, int) and 0 <= score <= 100


def test_reading_without_model_features_is_not_buffered(engine):
    assert engine.add_vital('mb-empty', {'timestamp': 0, 'EtCO2': 35}) == 0
    assert engine.vital_buffer['mb-empty'] == {}


# -------------------------------------------------------- 4. payload aliases (bug 2)

def test_canonicalize_vital_maps_aliases():
    from backend.inference import canonicalize_vital
    out = canonicalize_vital({'patient_id': 'P1', 'timestamp': 1, 'hr': 72, 'spo2': 98,
                              'rr': 18, 'temp': 37, 'systolic': 120, 'diastolic': 80, 'etco2': 35})
    assert out == {'patient_id': 'P1', 'timestamp': 1, 'HR': 72, 'SpO2': 98, 'RespRate': 18,
                   'Temp': 37, 'NISysABP': 120, 'NIDiasABP': 80, 'EtCO2': 35}


def test_canonicalize_vital_prefers_canonical_key():
    from backend.inference import canonicalize_vital
    assert canonicalize_vital({'HR': 90, 'hr': 50})['HR'] == 90
    assert canonicalize_vital({'hr': 50, 'HR': 90})['HR'] == 90
    assert canonicalize_vital({'HR': None, 'hr': 50})['HR'] == 50


def test_lowercase_payload_is_scored(engine):
    # Regression: ESP32 keys were ignored, buffer stayed empty, score always 0.
    engine.add_vital('al-lower', {'timestamp': 0, 'hr': 140, 'spo2': 80})
    X = _denormalize(engine, engine.get_model_input('al-lower'))
    np.testing.assert_allclose(X[-1, [0, 5]], [140, 80], rtol=1e-5)


def test_mqtt_lowercase_payload_is_scored_and_stored(tmp_db):
    import backend.mqtt_subscriber as sub
    msg = types.SimpleNamespace(payload=json.dumps(
        {'patient_id': 'mqtt-lower', 'timestamp': 60, 'hr': 140, 'spo2': 80}).encode())
    sub.on_message(None, None, msg)
    rows = tmp_db.get_latest_vitals('mqtt-lower')
    assert len(rows) == 1
    ts, hr, spo2 = rows[0][:3]
    risk = rows[0][8]
    assert (ts, hr, spo2) == (60, 140, 80)
    assert risk is not None and risk > 0


def test_mqtt_invalid_payloads_are_skipped(tmp_db):
    import backend.mqtt_subscriber as sub
    for payload in [b'not json', b'[1,2]', json.dumps({'hr': 70}).encode()]:
        sub.on_message(None, None, types.SimpleNamespace(payload=payload))
    with sqlite3.connect(tmp_db.DB_PATH) as conn:
        assert conn.execute('SELECT COUNT(*) FROM vitals').fetchone()[0] == 0


def test_firmware_payload_uses_model_keys_and_real_time():
    src = open(os.path.join(REPO, 'firmware', 'esp32_max30105', 'publish_example.ino')).read()
    # Full C string literal, honouring \" escapes.
    fmt = re.search(r'snprintf\(payload,[^"]*"((?:[^"\\]|\\.)*)"', src, re.S).group(1)
    keys = re.findall(r'\\"(\w+)\\":', fmt)
    assert keys == ['patient_id', 'timestamp', 'HR', 'SpO2']
    assert set(keys[2:]) <= set(FEATURES)
    assert 'time(nullptr)' in src and '1234567' not in src


# ------------------------------------------------------ 5. ml/train.py (bug 4)

def test_train_defaults_match_serving_contract(tmp_path):
    import ml.train as train
    import backend.inference as inf
    assert train.SERVING_FEATURES == inf.FEATURES
    set_dir, outcomes = _write_physionet(tmp_path)
    protected = [os.path.join(REPO, 'ml', 'scaler.json'),
                 os.path.join(REPO, 'ml', 'models', 'lstm_baseline.pt')]
    before = [_sha(p) for p in protected]
    run_dir = tmp_path / 'run'
    metrics = train.main(['--physionet', set_dir, '--outcomes', outcomes, '--epochs', '1',
                          '--run-dir', str(run_dir)])
    assert [_sha(p) for p in protected] == before, 'training without --deploy must not touch serving files'
    scaler = json.load(open(run_dir / 'scaler.json'))
    assert scaler['features'] == inf.FEATURES
    assert metrics['epochs_trained'] == 1
    import torch
    state = torch.load(run_dir / 'model.pt', weights_only=True)
    assert tuple(state['lstm.weight_ih_l0'].shape) == (4 * 96, 12)


def test_train_deploy_refuses_unservable_config(tmp_path):
    import ml.train as train
    with pytest.raises(SystemExit):
        train.main(['--physionet', 'x', '--outcomes', 'y', '--window', '60', '--deploy'])
    with pytest.raises(SystemExit):
        train.main(['--physionet', 'x', '--outcomes', 'y', '--hidden-size', '64', '--deploy'])


def test_serving_contract_errors():
    import argparse
    import ml.train as train
    ok = argparse.Namespace(vital_features=list(train.SERVING_FEATURES), window=90,
                            hidden_size=96, bidirectional=False)
    assert train.serving_contract_errors(ok) == []
    bad = argparse.Namespace(vital_features=['HR'], window=60, hidden_size=64, bidirectional=True)
    assert len(train.serving_contract_errors(bad)) == 4


def test_deploy_artifacts_writes_sibling_scaler(tmp_path):
    import ml.train as train
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'model.pt').write_bytes(b'model')
    (run / 'scaler.json').write_text('{"mean": [1]}')
    dest = tmp_path / 'models' / 'lstm_baseline.pt'
    model_path, scaler_path = train.deploy_artifacts(str(run), str(dest))
    assert open(model_path, 'rb').read() == b'model'
    assert scaler_path == str(tmp_path / 'models' / 'lstm_baseline_scaler.json')
    assert json.load(open(scaler_path)) == {'mean': [1]}


# ------------------------------------------- 6. single-model fallback (bug 5)

def test_single_model_fallback_loads(tmp_path):
    from backend.inference import RiskScoreEngine
    model = tmp_path / 'lstm_baseline.pt'
    shutil.copy(os.path.join(REPO, 'ml', 'models', 'lstm_baseline.pt'), model)
    e = RiskScoreEngine(model_path=str(model), manifest_path=str(tmp_path / 'none.json'))
    assert e.model is not None and e.models == []
    assert e._train_mean is not None, 'shared ml/scaler.json should be the fallback scaler'
    score = e.add_vital('fb', dict(HEALTHY_VITALS, timestamp=0))
    assert isinstance(score, int) and 0 <= score <= 100


def test_single_model_prefers_sibling_scaler(tmp_path):
    from backend.inference import RiskScoreEngine
    model = tmp_path / 'lstm_baseline.pt'
    shutil.copy(os.path.join(REPO, 'ml', 'models', 'lstm_baseline.pt'), model)
    (tmp_path / 'lstm_baseline_scaler.json').write_text(json.dumps({'mean': [7.0] * 12, 'std': [2.0] * 12}))
    e = RiskScoreEngine(model_path=str(model), manifest_path=str(tmp_path / 'none.json'))
    np.testing.assert_allclose(e._train_mean, 7.0)


def test_missing_model_returns_none_and_degrades(tmp_path):
    from backend.inference import RiskScoreEngine
    e = RiskScoreEngine(model_path=str(tmp_path / 'nope.pt'), manifest_path=str(tmp_path / 'none.json'))
    assert e.model is None
    assert e.add_vital('none', dict(HEALTHY_VITALS, timestamp=0)) is None
    assert e.degraded is True


def test_manifest_ensemble_still_preferred(engine):
    assert len(engine.models) == 3 and engine.model is engine.models[0]
    assert engine.window_size == 90


# ---------------------------------------------- 7. backend/training.py (bug 6)

def test_population_stats_all_nan_feature_is_neutral():
    from backend.training import population_stats, normalize
    X = np.random.default_rng(0).normal(size=(4, 5, 3)).astype(np.float32)
    X[:, :, 2] = np.nan
    mean, std = population_stats(X)
    assert mean[2] == 0.0 and std[2] == 1.0
    Xn = normalize(X, mean, std)
    assert Xn.dtype == np.float32 and not np.isnan(Xn).any()
    assert np.all(Xn[:, :, 2] == 0)


def test_training_job_with_missing_feature_completes(tmp_path, monkeypatch):
    import backend.training as t
    monkeypatch.setattr(t, 'JOBS_STORE', str(tmp_path / 'jobs.json'))
    set_dir, outcomes = _write_physionet(tmp_path, missing=('Glucose',))
    protected = os.path.join(REPO, 'ml', 'scaler.json')
    before = _sha(protected)
    mgr = t.TrainingManager()
    job = mgr.create_job({
        'physionet_path': set_dir, 'outcomes_path': outcomes, 'epochs': 1,
        'vital_features': FEATURES, 'window': 90, 'hidden_size': 96, 'max_patients': 10,
        'model_output': str(tmp_path / 'job.pt'),
    })
    mgr.active_job = job.job_id
    mgr._train_worker(job)
    assert job.status == 'completed', job.error_message
    assert np.isfinite(job.metrics['train_loss'])
    assert mgr.active_job is None
    assert os.path.exists(tmp_path / 'job.pt') and os.path.exists(tmp_path / 'job_scaler.json')
    assert _sha(protected) == before


def test_job_created_at_is_stable_and_persisted(tmp_path, monkeypatch):
    import backend.training as t
    monkeypatch.setattr(t, 'JOBS_STORE', str(tmp_path / 'jobs.json'))
    mgr = t.TrainingManager()
    job = mgr.create_job({'epochs': 1})
    first = job.to_dict()['created_at']
    assert job.to_dict()['created_at'] == first
    reloaded = t.TrainingManager().get_job(job.job_id)
    assert reloaded.to_dict()['created_at'] == first


def test_only_one_training_job_at_a_time(tmp_path, monkeypatch):
    import backend.training as t
    monkeypatch.setattr(t, 'JOBS_STORE', str(tmp_path / 'jobs.json'))
    mgr = t.TrainingManager()
    monkeypatch.setattr(mgr, '_train_worker', lambda job: None)
    j1, j2 = mgr.create_job({'epochs': 1}), mgr.create_job({'epochs': 1})
    assert mgr.start_training(j1.job_id) is True
    assert mgr.start_training(j2.job_id) is False


# ------------------------------------------------------------- 8. db (bug 7)

def test_top_patients_one_row_per_patient_on_timestamp_tie(tmp_db):
    for risk in (10, 20, 30):
        tmp_db.insert_vital({'patient_id': 'P1', 'timestamp': 1234567, 'HR': 70, 'risk_score': risk})
    tmp_db.insert_vital({'patient_id': 'P2', 'timestamp': 5, 'HR': 70, 'risk_score': 50})
    rows = tmp_db.get_top_patients(6)
    assert [r[0] for r in rows] == ['P2', 'P1']
    assert rows[1][1] == 30, 'tie must resolve to the most recently inserted row'


def test_top_patients_uses_latest_reading_and_limit(tmp_db):
    tmp_db.insert_vital({'patient_id': 'A', 'timestamp': 2, 'risk_score': 5})
    tmp_db.insert_vital({'patient_id': 'A', 'timestamp': 1, 'risk_score': 99})
    for i in range(8):
        tmp_db.insert_vital({'patient_id': f'B{i}', 'timestamp': 1, 'risk_score': 10 + i})
    rows = tmp_db.get_top_patients(6)
    assert len(rows) == 6
    assert 'A' not in [r[0] for r in rows], 'A latest risk is 5, older 99 must be ignored'


def test_init_db_creates_index_and_wal(tmp_db):
    with sqlite3.connect(tmp_db.DB_PATH) as conn:
        idx = [r[1] for r in conn.execute("PRAGMA index_list('vitals')")]
        mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
    assert 'idx_vitals_patient_ts' in idx
    assert mode.lower() == 'wal'


def test_insert_vital_accepts_both_key_styles(tmp_db):
    tmp_db.insert_vital({'patient_id': 'K', 'timestamp': 1, 'hr': 70, 'SpO2': 97, 'GCS': 15})
    row = tmp_db.get_latest_vitals('K')[0]
    assert (row[1], row[2], row[9]) == (70, 97, 15)


# ---------------------------------------------------- 9. explain endpoint (bug 8)

def test_explain_uses_scored_input(monkeypatch, tmp_db):
    from fastapi.testclient import TestClient
    import backend.app as app_module
    import ml.explain
    captured = {}

    def fake_shap(model, X, feature_names, n_background=10):
        captured['X'] = X
        return {f: 0.0 for f in feature_names}

    monkeypatch.setattr(ml.explain, 'compute_shap_explanation', fake_shap)
    client = TestClient(app_module.app)
    for i in range(3):
        assert client.post('/ingest', json=dict(SICK, patient_id='ex-1', timestamp=60 * i)).status_code == 200
    body = client.get('/patient/ex-1/explain').json()
    assert len(body['attention_weights']) == 90
    assert set(body['feature_importance']) == set(FEATURES)
    np.testing.assert_array_equal(captured['X'], app_module.inference_engine.get_model_input('ex-1'))


def test_explain_unknown_patient():
    from fastapi.testclient import TestClient
    import backend.app as app_module
    body = TestClient(app_module.app).get('/patient/never-seen/explain').json()
    assert 'error' in body


def test_explain_real_shap_runs(tmp_db):
    from fastapi.testclient import TestClient
    import backend.app as app_module
    client = TestClient(app_module.app)
    client.post('/ingest', json=dict(SICK, patient_id='ex-real', timestamp=0))
    body = client.get('/patient/ex-real/explain').json()
    imp = body['feature_importance']
    assert 'error' not in imp and len(imp) == 12


# ------------------------------------------------------------- 10. misc fixes

def test_cors_wildcard_does_not_allow_credentials():
    from fastapi.testclient import TestClient
    import backend.app as app_module
    r = TestClient(app_module.app).get('/health', headers={'Origin': 'http://evil.example'})
    assert r.headers.get('access-control-allow-origin') == '*'
    assert r.headers.get('access-control-allow-credentials') is None


def test_alert_cooldown_is_thread_safe(monkeypatch):
    import backend.app as app_module
    monkeypatch.setattr(app_module, '_last_alert_sent', {})
    results = []
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        results.append(app_module._should_send_alert('race', 'k'))

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results.count(True) == 1


def test_stale_patients_are_evicted():
    import backend.inference as inf
    now = [0.0]
    e = inf.RiskScoreEngine(clock=lambda: now[0])
    e.add_vital('old', dict(HEALTHY_VITALS, timestamp=0))
    now[0] = inf.PATIENT_TTL_SECONDS + inf._EVICT_INTERVAL_SECONDS + 1
    e.add_vital('new', dict(HEALTHY_VITALS, timestamp=0))
    for store in (e.vital_buffer, e.risk_scores, e.last_seen):
        assert 'old' not in store and 'new' in store
    assert 'old' not in e.carry


def test_train_batches_from_single_tensor():
    from ml.train_lstm import train
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 20, 12))  # float64 input must still work
    y = (rng.random(64) > 0.7).astype(float)
    losses = []
    model, opt = train(X, y, epochs=2, batch_size=16, hidden_size=8, device='cpu',
                       progress_callback=lambda e, m: losses.append(m['train_loss']))
    assert len(losses) == 2 and all(np.isfinite(losses))
    model2, _ = train(X, y, epochs=1, batch_size=16, device='cpu', model=model, optimizer=opt)
    assert model2 is model, 'continuing training must reuse the same model'
