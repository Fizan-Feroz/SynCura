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
    for key in ('tick_ms_last', 'tick_ms_avg', 'tick_ms_max'):
        snap.pop(key, None)
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
    septic = engine.control('set_scenario', 'septic')
    assert septic['scenario'] == 'septic'
    assert septic['simulated'] is True
    assert septic['scenario_tick'] == 0
    with pytest.raises(ValueError):
        engine.control('set_scenario', 'not-a-scenario')
    with pytest.raises(ValueError):
        engine.control('explode')
    reset = engine.control('reset')
    assert reset['tick'] == 0 and reset['paused'] is True and reset['scenario'] == 'baseline'
    assert reset['simulated'] is False and reset['scenario_tick'] == 0


def test_baseline_has_no_nudge():
    engine = make_engine(seed=11)
    snap = engine.step()
    assert snap['simulated'] is False
    assert snap['scenario_tick'] == 1


def test_septic_nudge_lifts_displayed_risk():
    base = make_engine(seed=11)
    sept = make_engine(seed=11)
    sept.control('set_scenario', 'septic')
    for _ in range(6):
        base.step()
        displayed = sept.step()
    base_risks = [b['risk'] for b in base.snapshot()['beds']]
    sept_risks = [b['risk'] for b in displayed['beds']]
    assert sum(sept_risks) > sum(base_risks), 'septic overlay must visibly lift risk'
    assert displayed['simulated'] is True
    assert all(8 <= r <= 99 for r in sept_risks)


def test_recovery_nudge_lowers_displayed_risk():
    base = make_engine(seed=11)
    rec = make_engine(seed=11)
    rec.control('set_scenario', 'recovery')
    for _ in range(6):
        base.step()
        displayed = rec.step()
    base_risks = [b['risk'] for b in base.snapshot()['beds']]
    rec_risks = [b['risk'] for b in displayed['beds']]
    assert sum(rec_risks) < sum(base_risks), 'recovery overlay must visibly lower risk'


def test_snapshot_schema_matches_frontend_shape():
    engine = make_engine(seed=9)
    snap = engine.step()
    assert snap['source'] == 'backend'
    for key in ('scenario', 'scenario_label', 'simulated', 'scenario_tick', 'paused', 'tick', 'seed', 'beds',
                'tick_ms_last', 'tick_ms_avg', 'tick_ms_max'):
        assert key in snap
    bed = snap['beds'][0]
    for key in ('patient_id', 'bed', 'status', 'risk', 'trend', 'lead',
                'trajectory', 'waveform', 'vitals', 'history'):
        assert key in bed, key
    for key in ('HR', 'SpO2', 'Resp', 'Temp'):
        assert key in bed['vitals'], key


def test_tick_timing_recorded():
    engine = make_engine(seed=4)
    engine.step()
    engine.step()
    stats = engine.tick_stats()
    assert stats['tick'] == 2
    assert stats['tick_ms_last'] >= 0
    assert stats['tick_ms_max'] >= stats['tick_ms_avg'] >= 0


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


def test_respiratory_run_does_not_pin_all_beds_at_rails():
    """Regression: DRIFT_GAIN 2.0 slammed every bed into identical clamp
    rails (HR 170 / SpO2 75 / RR 42) within ~10 ticks, collapsing all
    differentiation. After 40 respiratory ticks beds must still differ."""
    engine = make_engine(seed=11)
    engine.control('set_scenario', 'respiratory')
    for _ in range(40):
        snap = engine.step()
    risks = [b['risk'] for b in snap['beds']]
    vitals = [(b['vitals']['HR'], b['vitals']['SpO2'], b['vitals']['Resp']) for b in snap['beds']]
    assert len(set(risks)) > 1, f'all 12 risks identical: {risks[0]}'
    assert len(set(vitals)) > 1, 'all 12 vital triples identical (rail-pinned)'
    # After 40 deterioration ticks most beds SHOULD look bad (that is the
    # what-if working); the bug was total collapse. Bound it below totality
    # with margin: at most 9 of 12 exactly rail-pinned at once.
    pinned = sum(1 for hr, spo2, rr in vitals if (hr, spo2, rr) == (170, 75, 42))
    assert pinned <= 9, f'{pinned}/12 beds pinned at rails simultaneously'


def test_seed_beds_carry_labs_for_full_scorer_payload():
    """Beds must carry all 12 model features so the scorer never sees a
    half-empty window (the other half of the identical-48s bug)."""
    from backend.simulation import LAB_ANCHORS
    engine = make_engine(seed=11)
    for bed in engine.snapshot()['beds']:
        for lab in LAB_ANCHORS:
            assert lab in bed['vitals'], lab
    engine.control('set_scenario', 'respiratory')
    for _ in range(3):
        engine.step()
    for bed in engine.snapshot()['beds']:
        for lab in LAB_ANCHORS:
            assert lab in bed['vitals'], lab


def test_scores_endpoint_excludes_sim_patients():
    """GET /scores must never surface sim-xxx engine keys as patients."""
    from fastapi.testclient import TestClient
    import backend.app as app_module
    client = TestClient(app_module.app)
    engine = app_module.sim_engine
    engine.control('start')
    for _ in range(3):
        engine.step()
    # Non-vacuous: the engine really holds sim scores right now...
    from backend import inference as inference_module
    live_scores = inference_module.get_engine().risk_scores
    assert any(str(pid).startswith('sim-') for pid in live_scores), 'test setup: no sim scores present'
    # ...yet the public endpoint must not surface them.
    body = client.get('/scores').json()
    rows = body.get('scores', body) if isinstance(body, dict) else body
    pids = [p[0] if isinstance(p, (list, tuple)) else p.get('patient_id') for p in rows]
    assert not any(str(p).startswith('sim-') for p in pids), pids
    engine.control('pause')
