"""Server-authoritative ICU simulation engine.

One global scenario engine advances every bed on a fixed tick so that every
connected browser sees identical state. The update math mirrors
frontend/src/simulationContext.jsx (same scenarios, trajectories, clamps and
rounding); only the randomness source differs (seeded ``random.Random`` here
versus ``Math.random`` in the browser), so sequences are deterministic per
seed on the backend.

Risk shown is the real model score (via the inference engine) when a scorer
is attached, falling back to the heuristic drift when the model is missing.

Restart behaviour (by decision): reseed on boot. No persistence — Render's
disk is ephemeral.

Divergences from the old browser engine (deliberate, do not "fix" without
deciding): non-baseline scenarios run slightly hotter drift (DRIFT_GAIN,
kept modest plus per-bed jitter so beds differentiate instead of pinning
at the clamp rails); displayed risk is the model score when a scorer is
attached (frontend demo never calls a model); backend waveform keeps 12
points while the old frontend kept 6. Labs/GCS ride along anchored near
population-normal values so the scorer always sees a full 12-feature
vector; the frontend demo engine mirrors this (see updateDemoBed).
"""
import logging
import math
import random
import threading
import time

logger = logging.getLogger('syncura.simulation')

TICK_SECONDS = 5
DEFAULT_SEED = 20260928
HISTORY_LIMIT = 60
WAVEFORM_LIMIT = 12

# Non-baseline scenarios are simulated what-if overlays: drift runs slightly
# hotter so the change is visible within a few ticks, and a capped risk nudge
# is added on top of the model score. Baseline stays pure model output (the
# "real" view). The snapshot flags simulated=true whenever a nudge is active.
# Gain is deliberately modest (1.2, was 2.0): hotter gains slammed every bed
# into the clamp rails within ~10 ticks, collapsing all differentiation.
# Per-bed drift jitter (see _seed_bed) keeps beds distinct on top of that.
DRIFT_GAIN = 1.2
SCENARIO_NUDGE_PER_TICK = {
    "baseline": 0.0,
    "respiratory": 1.5,
    "septic": 2.0,
    "cardiac": 2.0,
    "recovery": -2.0,
}
NUDGE_CAP = 20
NUDGE_RAMP_TICKS = 10

# Slow-moving features the 4-channel sim does not drive. Anchored near
# population-normal values with tiny per-tick noise so the model scorer
# receives a full, realistic 12-feature vector instead of 8 NaN channels
# (which collapsed every bed to the same middling score). No scenario drift
# is applied to labs: in reality they move on hour/day timescales, far
# slower than this tick. GCS stays integer-valued.
LAB_ANCHORS = {
    "GCS": 15,
    "BUN": 18.0,
    "Creatinine": 1.0,
    "WBC": 8.0,
    "Platelets": 250.0,
    "Glucose": 110.0,
}
LAB_NOISE = {
    "GCS": 0.0,
    "BUN": 0.4,
    "Creatinine": 0.03,
    "WBC": 0.15,
    "Platelets": 4.0,
    "Glucose": 1.5,
}
LAB_BOUNDS = {
    "GCS": (3, 15),
    "BUN": (5, 80),
    "Creatinine": (0.4, 6.0),
    "WBC": (1.0, 30.0),
    "Platelets": (20.0, 600.0),
    "Glucose": (60.0, 300.0),
}

# Seed beds — must stay in sync with frontend/src/mimicDemoPatients.json.
# backend/tests/test_simulation.py asserts parity.
SEED_PATIENTS = [
    {"patient_id": "33281088", "bed": "ICU-01", "risk": 30,
     "vitals": {"HR": 74, "SpO2": 96, "Resp": 18, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [24, 26, 27, 28, 29, 30]},
    {"patient_id": "32359580", "bed": "ICU-02", "risk": 40,
     "vitals": {"HR": 106, "SpO2": 96, "Resp": 23, "Temp": 34.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [34, 36, 37, 38, 39, 40]},
    {"patient_id": "31316840", "bed": "ICU-03", "risk": 30,
     "vitals": {"HR": 74, "SpO2": 96, "Resp": 19, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [24, 26, 27, 28, 29, 30]},
    {"patient_id": "38197705", "bed": "ICU-04", "risk": 30,
     "vitals": {"HR": 100, "SpO2": 96, "Resp": 24, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [24, 26, 27, 28, 29, 30]},
    {"patient_id": "38383343", "bed": "ICU-05", "risk": 30,
     "vitals": {"HR": 101, "SpO2": 96, "Resp": 19, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [24, 26, 27, 28, 29, 30]},
    {"patient_id": "32128372", "bed": "ICU-06", "risk": 45,
     "vitals": {"HR": 117, "SpO2": 97, "Resp": 16, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [39, 41, 42, 43, 44, 45]},
    {"patient_id": "37293400", "bed": "ICU-07", "risk": 30,
     "vitals": {"HR": 91, "SpO2": 96, "Resp": 20, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [24, 26, 27, 28, 29, 30]},
    {"patient_id": "30932571", "bed": "ICU-08", "risk": 45,
     "vitals": {"HR": 118, "SpO2": 96, "Resp": 18, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [39, 41, 42, 43, 44, 45]},
    {"patient_id": "30955999", "bed": "ICU-09", "risk": 45,
     "vitals": {"HR": 111, "SpO2": 96, "Resp": 26, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [39, 41, 42, 43, 44, 45]},
    {"patient_id": "32391858", "bed": "ICU-10", "risk": 30,
     "vitals": {"HR": 89, "SpO2": 96, "Resp": 17, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [24, 26, 27, 28, 29, 30]},
    {"patient_id": "34531557", "bed": "ICU-11", "risk": 30,
     "vitals": {"HR": 80, "SpO2": 96, "Resp": 18, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [24, 26, 27, 28, 29, 30]},
    {"patient_id": "39635619", "bed": "ICU-12", "risk": 30,
     "vitals": {"HR": 93, "SpO2": 96, "Resp": 19, "Temp": 36.8, "GCS": 15, "BUN": 18.0, "Creatinine": 1.0, "WBC": 8.0, "Platelets": 250.0, "Glucose": 110.0},
     "waveform": [24, 26, 27, 28, 29, 30]},
]

SCENARIOS = {
    "baseline": {"label": "Baseline Mix", "lead": "Mixed instability",
                 "profile": {"hr": 0, "spo2": 0, "resp": 0, "temp": 0, "riskDrift": 0, "volatility": 1.5}},
    "respiratory": {"label": "Respiratory Decline", "lead": "Respiratory decline",
                    "profile": {"hr": 4, "spo2": -3, "resp": 5, "temp": 0.3, "riskDrift": 4, "volatility": 2.5}},
    "septic": {"label": "Septic Shock", "lead": "Sepsis escalation",
               "profile": {"hr": 8, "spo2": -2, "resp": 4, "temp": 0.8, "riskDrift": 6, "volatility": 3}},
    "cardiac": {"label": "Cardiac Stress", "lead": "Arrhythmic stress",
                "profile": {"hr": 11, "spo2": -1, "resp": 2, "temp": 0.2, "riskDrift": 5, "volatility": 4}},
    "recovery": {"label": "Recovery Trend", "lead": "Clinical recovery",
                 "profile": {"hr": -5, "spo2": 2, "resp": -3, "temp": -0.4, "riskDrift": -5, "volatility": 1.2}},
}

TRAJECTORY_PROFILES = {
    "severe": {"hr": 7, "spo2": -3, "resp": 5, "temp": 0.4, "riskDrift": 7, "volatility": 2.6,
               "lead": "Multi-organ deterioration"},
    "recovery": {"hr": -4, "spo2": 2, "resp": -3, "temp": -0.3, "riskDrift": -6, "volatility": 1.4,
                 "lead": "Clinical improvement"},
    "volatile": {"hr": 3, "spo2": -1, "resp": 2, "temp": 0.1, "riskDrift": 1, "volatility": 4.2,
                 "lead": "Unstable oscillations"},
    "stable": {"hr": 0, "spo2": 0, "resp": 0, "temp": 0, "riskDrift": 0, "volatility": 1.2,
               "lead": "Stable monitoring"},
}


def _clamp(value, low, high):
    return max(low, min(high, value))


def _round_half_up(value):
    # Match JS Math.round for the non-negative values used here.
    return int(math.floor(value + 0.5))


def _round_1(value):
    return math.floor(value * 10 + 0.5) / 10


def _random_centered(rng, scale):
    return (rng.random() * 2 - 1) * scale


def _assign_trajectory(rng):
    roll = rng.random()
    if roll < 0.27:
        return "severe"
    if roll < 0.52:
        return "recovery"
    if roll < 0.78:
        return "stable"
    return "volatile"


def _score_contributions(vitals):
    return (
        (vitals["HR"] - 85) * 0.24
        + (92 - vitals["SpO2"]) * 1.7
        + (vitals["Resp"] - 18) * 0.6
        + (vitals["Temp"] - 37) * 4.5
    )


def _status_for_risk(risk):
    if risk >= 85:
        return "Critical"
    if risk >= 70:
        return "High"
    if risk >= 45:
        return "Watch"
    return "Stable"


class SimulationEngine:
    """Single global scenario engine. Thread-safe; call step() to advance."""

    def __init__(self, seed=DEFAULT_SEED, tick_seconds=TICK_SECONDS):
        self.seed = seed
        self.tick_seconds = tick_seconds
        self.lock = threading.Lock()
        self.scorer = None
        self._thread = None
        self._running = False
        self.reset()

    # ------------------------------------------------------------ lifecycle

    def reset(self, seed=None):
        """Reseed beds deterministically and pause. Restart behaviour."""
        with self.lock:
            if seed is not None:
                self.seed = seed
            self.rng = random.Random(self.seed)
            self.scenario = "baseline"
            self.paused = True
            self.tick = 0
            self.scenario_tick = 0
            self.tick_ms_last = 0.0
            self.tick_ms_avg = 0.0
            self.tick_ms_max = 0.0
            self.beds = [self._seed_bed(dict(p)) for p in SEED_PATIENTS]

    def _seed_bed(self, patient):
        trajectory = _assign_trajectory(self.rng)
        profile = TRAJECTORY_PROFILES[trajectory]
        if trajectory == "severe":
            risk = patient["risk"] + 18
        elif trajectory == "recovery":
            risk = patient["risk"] - 10
        else:
            risk = patient["risk"]
        risk = _round_half_up(_clamp(risk, 8, 95))
        # Per-bed drift jitter so identical scenarios/trajectories still
        # produce distinct beds instead of converging to the same rails.
        drift_mult = 0.85 + self.rng.random() * 0.3
        vitals = dict(patient["vitals"])
        for lab, anchor in LAB_ANCHORS.items():
            vitals.setdefault(lab, anchor)
        now = time.time()
        return {
            "patient_id": patient["patient_id"],
            "bed": patient["bed"],
            "status": _status_for_risk(risk),
            "risk": risk,
            "trend": "+0",
            "lead": profile["lead"],
            "trajectory": trajectory,
            "drift_mult": drift_mult,
            "waveform": list(patient["waveform"][:-1]) + [risk],
            "vitals": vitals,
            "history": [{"t": now, **vitals, "risk": risk}],
        }

    def start(self):
        with self.lock:
            self.paused = False

    def pause(self):
        with self.lock:
            self.paused = True

    def set_scenario(self, scenario):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        with self.lock:
            self.scenario = scenario
            self.scenario_tick = 0

    def control(self, action, scenario=None):
        if action == "start":
            self.start()
        elif action == "pause":
            self.pause()
        elif action == "reset":
            self.reset()
        elif action == "set_scenario":
            if not scenario:
                raise ValueError("set_scenario requires a scenario")
            self.set_scenario(scenario)
        else:
            raise ValueError(f"Unknown action: {action}")
        return self.snapshot()

    # ------------------------------------------------------------------ tick

    def set_scorer(self, scorer):
        """Attach the model scorer thread-safely (called once at boot).

        The scorer takes [(patient_id, vital_dict)] and returns aligned
        risks, so one tick costs a handful of batched forwards.
        """
        with self.lock:
            self.scorer = scorer

    def step(self):
        """Advance every bed one tick. Returns the new snapshot.

        Model scoring runs OUTSIDE the lock (it can take far longer than a
        tick) so snapshot()/control() never block behind torch inference.
        All RNG draws happen in the prepare phase, preserving determinism.
        Step duration (ms) is recorded for the admin panel.
        """
        t0 = time.perf_counter()
        with self.lock:
            if self.paused:
                return self._snapshot_locked()
            scenario = SCENARIOS[self.scenario]
            profile, scenario_lead = scenario["profile"], scenario["lead"]
            self.tick += 1
            self.scenario_tick += 1
            gain = DRIFT_GAIN if self.scenario != "baseline" else 1.0
            nudge = _clamp(SCENARIO_NUDGE_PER_TICK[self.scenario]
                           * min(self.scenario_tick, NUDGE_RAMP_TICKS),
                           -NUDGE_CAP, NUDGE_CAP)
            now = time.time()
            pending = [self._prepare_bed(bed, profile, gain, now) for bed in self.beds]
            scorer = self.scorer
        scored = self._score_pending(scorer, pending, nudge)
        with self.lock:
            for bed, prep, risk in zip(self.beds, pending, scored):
                self._commit_bed(bed, prep, risk, scenario_lead)
            dt_ms = (time.perf_counter() - t0) * 1000
            self.tick_ms_last = dt_ms
            self.tick_ms_max = max(self.tick_ms_max, dt_ms)
            # Rolling mean without keeping history.
            self.tick_ms_avg += (dt_ms - self.tick_ms_avg) / self.tick
            return self._snapshot_locked()

    def _score_pending(self, scorer, pending, nudge):
        """Resolve displayed risk for every prepared bed (lock-free)."""
        if scorer is None:
            scored = [None] * len(pending)
        else:
            try:
                items = [(f"sim-{prep['bed_id']}", prep["vital_dict"]) for prep in pending]
                scored = scorer(items)
                if list(scored or []) and len(scored) != len(pending):
                    raise ValueError(f"scorer returned {len(scored)} risks for {len(pending)} beds")
            except Exception:
                logger.exception("Simulation batch scorer failed; using heuristics")
                scored = [None] * len(pending)
        risks = []
        for prep, s in zip(pending, scored):
            base = prep["heuristic"] if s is None else int(s)
            risks.append(_round_half_up(_clamp(base + nudge, 8, 99)))
        return risks

    def _prepare_bed(self, bed, profile, gain, now):
        """Compute next vitals + heuristic risk. Caller must hold the lock."""
        traj = TRAJECTORY_PROFILES.get(bed.get("trajectory") or "stable",
                                       TRAJECTORY_PROFILES["stable"])
        randomizer = max(0.8, (profile["volatility"] + traj["volatility"]) / 2)
        jitter = bed.get("drift_mult", 1.0)
        combined = {
            "hr": (profile["hr"] + traj["hr"]) * gain * jitter,
            "spo2": (profile["spo2"] + traj["spo2"]) * gain * jitter,
            "resp": (profile["resp"] + traj["resp"]) * gain * jitter,
            "temp": (profile["temp"] + traj["temp"]) * gain * jitter,
            "riskDrift": (profile["riskDrift"] + traj["riskDrift"]) * gain * jitter,
        }
        v = bed["vitals"]
        next_vitals = {
            "HR": _round_half_up(_clamp(
                v["HR"] + combined["hr"] * 0.35 + _random_centered(self.rng, randomizer), 45, 170)),
            "SpO2": _round_half_up(_clamp(
                v["SpO2"] + combined["spo2"] * 0.25 + _random_centered(self.rng, randomizer * 0.35),
                75, 100)),
            "Resp": _round_half_up(_clamp(
                v["Resp"] + combined["resp"] * 0.25 + _random_centered(self.rng, randomizer * 0.4),
                10, 42)),
            "Temp": _round_1(_clamp(
                v["Temp"] + combined["temp"] * 0.12 + _random_centered(self.rng, randomizer * 0.03),
                34.5, 41)),
        }
        for lab, anchor in LAB_ANCHORS.items():
            lo, hi = LAB_BOUNDS[lab]
            prev = v.get(lab, anchor)
            if not isinstance(prev, (int, float)) or prev != prev:  # missing/NaN
                prev = anchor
            val = prev + _random_centered(self.rng, LAB_NOISE[lab])
            next_vitals[lab] = int(_round_half_up(_clamp(val, lo, hi))) if lab == "GCS" else round(_clamp(val, lo, hi), 1)
        # Physiology-tracking heuristic (fallback when no scorer): risk is
        # pulled toward the vitals-implied level each tick instead of
        # accumulating open-loop drift, which otherwise marches every bed to
        # the 99 clamp and collapses all differentiation. The small drift
        # term keeps scenarios visibly distinct; noise stays small so it
        # cannot drown the signal.
        implied = _clamp(30 + _score_contributions(next_vitals) * 0.35, 8, 99)
        heuristic = _round_half_up(_clamp(
            bed["risk"] + (implied - bed["risk"]) * 0.3
            + combined["riskDrift"] * 0.05
            + _random_centered(self.rng, randomizer * 0.5), 8, 99))
        return {"bed_id": bed["patient_id"], "vitals": next_vitals,
                "heuristic": heuristic, "now": now,
                "traj_lead": traj["lead"], "trajectory": bed.get("trajectory"),
                "vital_dict": {
                    "patient_id": f"sim-{bed['patient_id']}",
                    "timestamp": now,
                    "HR": next_vitals["HR"],
                    "SpO2": next_vitals["SpO2"],
                    "RespRate": next_vitals["Resp"],
                    "Temp": next_vitals["Temp"],
                    # Full 12-feature vector: labs/GCS ride along so the
                    # model never scores a half-empty (8-NaN) window, which
                    # collapsed every bed to the same middling score.
                    "GCS": next_vitals["GCS"],
                    "BUN": next_vitals["BUN"],
                    "Creatinine": next_vitals["Creatinine"],
                    "WBC": next_vitals["WBC"],
                    "Platelets": next_vitals["Platelets"],
                    "Glucose": next_vitals["Glucose"],
                }}

    def _commit_bed(self, bed, prep, risk, scenario_lead):
        """Write prepared results into bed state. Caller must hold the lock."""
        change = risk - bed["risk"]
        if risk >= 85:
            lead = "Multi-organ deterioration"
        elif risk < 45:
            lead = "Baseline recovery"
        elif prep["trajectory"] == "recovery":
            lead = prep["traj_lead"]
        else:
            lead = scenario_lead
        bed["vitals"] = prep["vitals"]
        bed["risk"] = risk
        bed["trend"] = f"{'+' if change >= 0 else ''}{change}"
        bed["status"] = _status_for_risk(risk)
        bed["lead"] = lead
        bed["waveform"] = (bed["waveform"] + [risk])[-WAVEFORM_LIMIT:]
        bed["history"] = (bed["history"] + [{"t": prep["now"], **prep["vitals"], "risk": risk}])[-HISTORY_LIMIT:]

    # ------------------------------------------------------------------ read

    def snapshot(self):
        with self.lock:
            return self._snapshot_locked()

    def tick_stats(self):
        """Tick timing counters, read under lock for the admin panel."""
        with self.lock:
            return {
                "tick": self.tick,
                "tick_ms_last": round(self.tick_ms_last, 1),
                "tick_ms_avg": round(self.tick_ms_avg, 1),
                "tick_ms_max": round(self.tick_ms_max, 1),
            }

    def _snapshot_locked(self):
        scenario = SCENARIOS[self.scenario]
        return {
            "source": "backend",
            "scenario": self.scenario,
            "scenario_label": scenario["label"],
            "simulated": self.scenario != "baseline",
            "scenario_tick": self.scenario_tick,
            "paused": self.paused,
            "tick": self.tick,
            "seed": self.seed,
            "tick_ms_last": round(self.tick_ms_last, 1),
            "tick_ms_avg": round(self.tick_ms_avg, 1),
            "tick_ms_max": round(self.tick_ms_max, 1),
            "beds": [dict(b, vitals=dict(b["vitals"]),
                          waveform=list(b["waveform"]),
                          history=[dict(h) for h in b["history"]]) for b in self.beds],
        }

    # ------------------------------------------------------------- background

    def start_background(self):
        with self.lock:
            if self._thread is not None:
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        while self._running:
            time.sleep(self.tick_seconds)
            try:
                self.step()
            except Exception:
                logger.exception("Simulation tick failed")


_engine = None
_engine_lock = threading.Lock()


def get_sim_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SimulationEngine()
                _engine.start_background()
    return _engine
