"""Server-authoritative simulation engine contracts.

Run from the repo root:
    .\\.venv\\Scripts\\python.exe -m pytest backend/tests/test_simulation.py -q
"""
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_engine(seed=7):
    from backend.simulation import SimulationEngine
    engine = SimulationEngine(seed=seed)
    engine.start()  # steps advance only when unpaused
    return engine


def test_seed_beds_match_frontend_seed_json():
    """Backend seed beds must stay identical to frontend/src/mimicDemoPatients.json."""
    from backend.simulation import SEED_PATIENTS
    with open(os.path.join(REPO, 'frontend', 'src', 'mimicDemoPatients.json')) as f:
        frontend = json.load(f)
    assert len(SEED_PATIENTS) == len(frontend) == 12
    for back, front in zip(SEED_PATIENTS, frontend):
        assert back['patient_id'] == front['patient_id']
        assert back['bed'] == front['bed']
        assert back['risk'] == front['risk']
        assert back['vitals'] == front['vitals']
        assert back['waveform'] == front['waveform']


def _timeless(snap):
    """Snapshot copy with wall-clock timestamps stripped (they legitimately differ)."""
    snap = json.loads(json.dumps(snap))
    for bed in snap['beds']:
        for point in bed['history']:
            point.pop('t', None)
    return snap


def test_same_seed_same_sequence():
    a, b = make_engine(seed=42), make_engine(seed=42)
    for _ in range(5):
        assert _timeless(a.step()) == _timeless(b.step())


def test_steps_advance_tick_and_stay_in_range():
    engine = make_engine(seed=1)
    first = engine.step()
    assert first['tick'] == 1
    assert len(first['beds']) == 12
    for bed in first['beds']:
        assert 8 <= bed['risk'] <= 99
        assert 45 <= bed['vitals']['HR'] <= 170
        assert 75 <= bed['vitals']['SpO2'] <= 100
        assert 10 <= bed['vitals']['Resp'] <= 42
        assert 34.5 <= bed['vitals']['Temp'] <= 41
        assert len(bed['waveform']) <= 12
        assert len(bed['history']) <= 60


def test_paused_engine_does_not_advance():
    from backend.simulation import SimulationEngine
    engine = SimulationEngine(seed=3)  # starts paused
    before = engine.snapshot()
    assert engine.step() == before


def test_control_transitions():
    engine = make_engine(seed=5)
    assert engine.control('pause')['paused'] is True
    assert engine.control('start')['paused'] is False
    assert engine.control('set_scenario', 'septic')['scenario'] == 'septic'
    with pytest.raises(ValueError):
        engine.control('set_scenario', 'not-a-scenario')
    with pytest.raises(ValueError):
        engine.control('explode')
    reset = engine.control('reset')
    assert reset['tick'] == 0 and reset['paused'] is True and reset['scenario'] == 'baseline'


def test_snapshot_schema_matches_frontend_shape():
    engine = make_engine(seed=9)
    snap = engine.step()
    assert snap['source'] == 'backend'
    for key in ('scenario', 'scenario_label', 'paused', 'tick', 'seed', 'beds'):
        assert key in snap
    bed = snap['beds'][0]
    for key in ('patient_id', 'bed', 'status', 'risk', 'trend', 'lead',
                'trajectory', 'waveform', 'vitals', 'history'):
        assert key in bed, key
    for key in ('HR', 'SpO2', 'Resp', 'Temp'):
        assert key in bed['vitals'], key


def test_simulation_endpoints():
    from fastapi.testclient import TestClient
    import backend.app as app_module
    client = TestClient(app_module.app)
    r = client.get('/simulation/state')
    assert r.status_code == 200
    assert len(r.json()['beds']) == 12
    r = client.post('/simulation/control', json={'action': 'pause'})
    assert r.status_code == 200 and r.json()['paused'] is True
    r = client.post('/simulation/control', json={'action': 'set_scenario', 'scenario': 'cardiac'})
    assert r.status_code == 200 and r.json()['scenario'] == 'cardiac'
    r = client.post('/simulation/control', json={'action': 'explode'})
    assert r.status_code == 400
    # Leave the shared engine running for other tests / manual use.
    client.post('/simulation/control', json={'action': 'start'})
