"""
Real-time inference module: loads trained AttentionLSTM and generates risk scores.

Serving builds the model input EXACTLY like training (ml/dataset.py):
readings are binned onto a 1-minute grid (mean per minute), the last
`window_size` MINUTES form the window, and gaps are forward-filled from the
most recent earlier observation, INCLUDING one older than the window (sparse
labs such as BUN are measured hours apart). Features never observed get the
training mean (normalizes to 0), then population z-scoring with the member's
scaler.
"""
import os
import glob
import json
import math
import time
import threading
import warnings

import numpy as np
import torch

from ml.dataset import carry_forward


# Determine model path: allow override via MODEL_PATH env var, check common paths,
# or pick the latest saved model from ml/training_runs/*/model.pt
DEFAULT_MODEL_PATH = os.getenv('MODEL_PATH', 'ml/models/lstm_baseline.pt')
if not os.path.exists(DEFAULT_MODEL_PATH):
    alt = 'ml/lstm_baseline.pt'
    if os.path.exists(alt):
        DEFAULT_MODEL_PATH = alt
    else:
        runs = sorted(glob.glob('ml/training_runs/*/model.pt'), key=os.path.getmtime, reverse=True)
        if runs:
            DEFAULT_MODEL_PATH = runs[0]

# Feature names used by the model (must match training)
FEATURES = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2',
            'GCS', 'BUN', 'Creatinine', 'WBC', 'Platelets', 'Glucose']

# Payload keys accepted from devices / older clients, mapped to model names.
# The ESP32 sketch historically sent lowercase keys ("hr", "spo2") which the
# engine silently ignored, so every hardware reading scored 0.
FEATURE_ALIASES = {
    'hr': 'HR', 'heart_rate': 'HR',
    'rr': 'RespRate', 'resp': 'RespRate', 'resprate': 'RespRate', 'respiratory_rate': 'RespRate',
    'temp': 'Temp', 'temperature': 'Temp',
    'systolic': 'NISysABP', 'sbp': 'NISysABP', 'nisysabp': 'NISysABP',
    'diastolic': 'NIDiasABP', 'dbp': 'NIDiasABP', 'nidiasabp': 'NIDiasABP',
    'spo2': 'SpO2', 'sao2': 'SpO2',
    'gcs': 'GCS', 'bun': 'BUN', 'creatinine': 'Creatinine', 'wbc': 'WBC',
    'platelets': 'Platelets', 'glucose': 'Glucose', 'etco2': 'EtCO2',
}

MANIFEST_PATH = os.path.join('ml', 'deployed_manifest.json')
DEFAULT_SCALER_PATH = os.path.join('ml', 'scaler.json')

# Patients with no reading for this long are dropped from memory.
PATIENT_TTL_SECONDS = float(os.getenv('PATIENT_TTL_SECONDS', str(6 * 3600)))
_EVICT_INTERVAL_SECONDS = 60.0


def canonicalize_vital(vital_dict):
    """Return a copy of `vital_dict` with alias keys mapped to model feature names.

    A canonical key that is already present always wins over an alias.
    """
    out = {k: v for k, v in vital_dict.items() if FEATURE_ALIASES.get(str(k).lower()) in (None, k)}
    for k, v in vital_dict.items():
        canon = FEATURE_ALIASES.get(str(k).lower())
        if canon and canon != k and out.get(canon) is None:
            out[canon] = v
    return out


def _to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return np.nan
    return f if math.isfinite(f) else np.nan


def _load_json_scaler(path):
    with open(path) as f:
        scaler = json.load(f)
    mean = np.array(scaler['mean'], dtype=np.float32)
    std = np.array(scaler['std'], dtype=np.float32) + 1e-6
    return mean, std


def _load_state(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


class RiskScoreEngine:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, window_size=90,
                 manifest_path=None, clock=time.monotonic):
        """Load model and initialize per-patient minute buffers."""
        self.model_path = model_path
        self.window_size = window_size
        self.manifest_path = manifest_path or MANIFEST_PATH
        self.model = None
        self.models = []  # ensemble members (logit-averaged); empty => use self.model
        self.member_scalers = []  # (mean, std) per member; falls back to shared stats
        # per patient: {minute: array(2, F) of [sums, counts]} for the last window
        self.vital_buffer = {}
        # per patient: (values[F], minutes[F]) last known value per feature from
        # minutes that already left the window (NaN / -inf = never observed)
        self.carry = {}
        self.risk_scores = {}   # per patient: latest risk score
        self.last_seen = {}     # per patient: clock() of last reading
        self.lock = threading.Lock()
        self.load_error = None  # set when model/scaler loading fails (surfaced via /health)
        self.degraded = False  # True when serving fallback scores instead of model scores
        self._clock = clock
        self._last_evict = clock()

        # Training-set statistics for normalization (set after training or loaded)
        self._train_mean = None
        self._train_std = None

        self._load_model()
        self._load_scaler()

    # ------------------------------------------------------------------ loading

    def _load_scaler(self, scaler_path=None):
        """Load population normalization stats saved during training.

        A single model's own `<model>_scaler.json` is preferred over the shared
        ml/scaler.json so a retrained baseline never needs to touch the file the
        ensemble members depend on.
        """
        if scaler_path is None:
            sibling = os.path.splitext(self.model_path)[0] + '_scaler.json'
            scaler_path = sibling if os.path.exists(sibling) else DEFAULT_SCALER_PATH
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path) as f:
                    scaler = json.load(f)
                self.set_normalization_stats(scaler['mean'], scaler['std'])
                print(f'[Inference] Loaded scaler stats from {scaler_path}')
            except Exception as e:
                print(f'[Inference] Warning: failed to load scaler: {e}')

    def _load_model(self):
        """Load the manifest ensemble, else an ensemble dir, else a single model."""
        from ml.train_lstm import AttentionLSTMModel

        # Preferred: versioned manifest (ml/deployed_manifest.json) with per-member
        # checkpoints, scalers, and architecture.
        manifest_members = self._manifest_members()
        if manifest_members:
            try:
                arch = self._manifest_arch()
                for member in manifest_members:
                    m = AttentionLSTMModel(
                        input_size=int(arch.get('input_size', len(FEATURES))),
                        hidden_size=arch.get('hidden_size', 96),
                        num_layers=arch.get('num_layers', 2),
                        dropout=arch.get('dropout', 0.3),
                        bidirectional=arch.get('bidirectional', False),
                    )
                    m.load_state_dict(_load_state(member['checkpoint']))
                    m.eval()
                    self.models.append(m)
                    try:
                        self.member_scalers.append(_load_json_scaler(member['scaler']))
                    except Exception as e:
                        print(f"[Inference] Warning: scaler load failed for {member['id']}: {e}")
                        self.member_scalers.append((None, None))
                self.model = self.models[0]  # primary (attention weights source)
                print(f'[Inference] Loaded manifest ensemble of {len(self.models)} models')
                return
            except Exception as e:
                self.load_error = f'manifest ensemble load failed: {e}'
                print(f'[Inference] Warning: {self.load_error}; falling back')
                self.models, self.member_scalers, self.model = [], [], None

        # Ensemble dir: ml/models/ensemble/*.pt, logits averaged.
        ens_dir = os.path.join(os.path.dirname(self.model_path), 'ensemble')
        ens_paths = sorted(glob.glob(os.path.join(ens_dir, '*.pt'))) if os.path.isdir(ens_dir) else []
        if ens_paths:
            try:
                for p in ens_paths:
                    m = AttentionLSTMModel(input_size=len(FEATURES), hidden_size=96)
                    m.load_state_dict(_load_state(p))
                    m.eval()
                    self.models.append(m)
                    self.member_scalers.append((None, None))
                self.model = self.models[0]
                print(f'[Inference] Loaded ensemble of {len(self.models)} models from {ens_dir}')
                return
            except Exception as e:
                self.load_error = f'ensemble load failed: {e}'
                print(f'[Inference] Warning: {self.load_error}; falling back to single model')
                self.models, self.member_scalers, self.model = [], [], None

        # Single model (previously unreachable dead code after a return).
        if os.path.exists(self.model_path):
            try:
                m = AttentionLSTMModel(input_size=len(FEATURES), hidden_size=96)
                m.load_state_dict(_load_state(self.model_path))
                m.eval()
                self.model = m
                print(f'[Inference] Loaded AttentionLSTM from {self.model_path} ({len(FEATURES)} features)')
            except Exception as e:
                self.load_error = f'single model load failed: {e}'
                print(f'[Inference] Warning: {self.load_error}')
                self.model = None
        else:
            print(f'[Inference] Model not found at {self.model_path}; scores unavailable')
            self.model = None

    def _manifest(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f'[Inference] Warning: failed to read manifest: {e}')
        return None

    def _manifest_members(self):
        manifest = self._manifest()
        if manifest and isinstance(manifest.get('members'), list) and manifest['members']:
            arch = manifest.get('arch', {})
            need = int(arch.get('input_size', len(FEATURES)))
            if need != len(FEATURES):
                # Fail fast with a clear message (e.g. a 24-dim gap model
                # cannot be served by the 12-feature engine yet; see
                # ml/RESULTS_PLAN.md "Serving work required").
                print(f'[Inference] Warning: manifest needs input_size={need} but engine '
                      f'builds {len(FEATURES)}-dim vectors; using directory scan')
                return None
            members = [m for m in manifest['members']
                       if os.path.exists(m.get('checkpoint', ''))]
            if len(members) == len(manifest['members']):
                self.window_size = int(manifest.get('window_minutes', self.window_size))
                return members
            print('[Inference] Warning: manifest checkpoints missing; using directory scan')
        return None

    def _manifest_arch(self):
        manifest = self._manifest()
        if manifest and isinstance(manifest.get('arch'), dict):
            return manifest['arch']
        return {}

    def set_normalization_stats(self, mean, std):
        """Set training-set normalization statistics."""
        self._train_mean = np.array(mean, dtype=np.float32)
        self._train_std = np.array(std, dtype=np.float32) + 1e-6

    # ---------------------------------------------------------------- buffering

    def _evict_stale_locked(self, now):
        if now - self._last_evict < _EVICT_INTERVAL_SECONDS:
            return
        self._last_evict = now
        stale = [pid for pid, seen in self.last_seen.items() if now - seen > PATIENT_TTL_SECONDS]
        for pid in stale:
            self.vital_buffer.pop(pid, None)
            self.carry.pop(pid, None)
            self.risk_scores.pop(pid, None)
            self.last_seen.pop(pid, None)

    def _carry_locked(self, patient_id):
        F = len(FEATURES)
        return self.carry.setdefault(
            patient_id, (np.full(F, np.nan), np.full(F, -np.inf)))

    @staticmethod
    def _update_carry(carry, minute, values, observed):
        """Keep the newest pre-window value per feature (out-of-order safe)."""
        vals, mins = carry
        newer = observed & (minute >= mins)
        vals[newer] = values[newer]
        mins[newer] = minute

    def _window_locked(self, patient_id):
        """(raw window, carried-in values) for the latest window; NaN = unobserved.

        The raw window is the minute grid ending at the patient's latest
        minute; the carried-in values are the last observations from before it.
        """
        bins = self.vital_buffer.get(patient_id)
        if not bins:
            return None
        end = max(bins)
        start = end - self.window_size + 1
        W = np.full((self.window_size, len(FEATURES)), np.nan, dtype=np.float64)
        for minute, (sums, counts) in bins.items():
            if minute >= start:
                observed = counts > 0
                W[minute - start, observed] = sums[observed] / counts[observed]
        initial = self.carry[patient_id][0].copy() if patient_id in self.carry else None
        return W, initial

    def add_vital(self, patient_id, vital_dict):
        """Add a vital measurement and compute risk score.

        Returns an int 0-100, or None when no model is loaded / inference
        fails (callers must surface this instead of inventing a score).
        """
        vital_dict = canonicalize_vital(vital_dict)
        vec = np.array([_to_float(vital_dict.get(f)) for f in FEATURES], dtype=np.float64)
        observed = ~np.isnan(vec)
        ts = _to_float(vital_dict.get('timestamp'))
        if np.isnan(ts):
            ts = time.time()
        minute = int(ts // 60)

        with self.lock:
            now = self._clock()
            self._evict_stale_locked(now)
            self.last_seen[patient_id] = now
            bins = self.vital_buffer.setdefault(patient_id, {})
            if observed.any():
                cutoff = max([minute, *bins]) - self.window_size + 1
                if minute < cutoff:
                    # Late reading older than the window: only its carry value matters.
                    self._update_carry(self._carry_locked(patient_id), minute, vec, observed)
                else:
                    entry = bins.get(minute)
                    if entry is None:
                        entry = bins[minute] = np.zeros((2, len(FEATURES)), dtype=np.float64)
                    entry[0, observed] += vec[observed]
                    entry[1, observed] += 1
                # Minutes leaving the window hand their values to the carry state.
                for m in sorted(m for m in bins if m < cutoff):
                    sums, counts = bins.pop(m)
                    seen = counts > 0
                    means = np.divide(sums, counts, out=np.full(len(FEATURES), np.nan), where=seen)
                    self._update_carry(self._carry_locked(patient_id), m, means, seen)
            window = self._window_locked(patient_id)

        if window is None:
            return 0
        # Model inference runs OUTSIDE the lock so one slow patient cannot
        # block ingestion for every other patient.
        risk_score = self._score_window(window)
        if risk_score is not None:
            with self.lock:
                self.risk_scores[patient_id] = risk_score
        return risk_score

    # ------------------------------------------------------------ preprocessing

    def _prepare(self, window, mean=None, std=None):
        """(raw window, carried-in values) -> normalized float32 model input (window_size, F)."""
        raw, initial = window
        X = carry_forward(raw, initial)
        if mean is None or std is None:
            mean, std = self._train_mean, self._train_std
        if mean is None or std is None:
            # No population stats at all: per-window z-score, unobserved -> 0.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                mu = np.nanmean(X, axis=0)
                sd = np.nanstd(X, axis=0) + 1e-6
            return np.nan_to_num((X - mu) / sd).astype(np.float32)
        X = np.where(np.isnan(X), mean, X)
        return ((X - mean) / std).astype(np.float32)

    def _member_inputs(self, window):
        if self.models:
            scalers = self.member_scalers or [(None, None)] * len(self.models)
            return [(m, self._prepare(window, mean, std)) for m, (mean, std) in zip(self.models, scalers)]
        return [(self.model, self._prepare(window))]

    def _score_window(self, window):
        """Score a raw window. Returns int 0-100, or None on failure."""
        if self.model is None and not self.models:
            self.degraded = True
            return None
        try:
            with torch.inference_mode():
                logits = [m(torch.from_numpy(X[None])).item() for m, X in self._member_inputs(window)]
            prob = 1.0 / (1.0 + math.exp(-sum(logits) / len(logits)))
            return max(0, min(100, round(prob * 100)))
        except Exception as e:
            print(f'[Inference] Error computing score: {e}')
            return None

    def get_model_input(self, patient_id):
        """Normalized window the PRIMARY model scores (for SHAP / attention), or None."""
        with self.lock:
            window = self._window_locked(patient_id)
        if window is None or self.model is None:
            return None
        mean, std = self.member_scalers[0] if self.member_scalers else (None, None)
        return self._prepare(window, mean, std)

    def get_attention_weights(self, patient_id):
        """Get attention weights for a patient's current window (for explainability)."""
        X = self.get_model_input(patient_id)
        if X is None:
            return None
        try:
            with torch.inference_mode():
                weights = self.model.get_attention_weights(torch.from_numpy(X[None]))
            return weights.squeeze(0).numpy().tolist()
        except Exception as e:
            print(f'[Inference] Error getting attention weights for {patient_id}: {e}')
            return None

    def get_risk_score(self, patient_id):
        """Retrieve latest risk score for a patient."""
        with self.lock:
            return self.risk_scores.get(patient_id, 0)

    def get_all_scores(self):
        """Get all patient risk scores sorted by risk (descending)."""
        with self.lock:
            sorted_scores = sorted(
                self.risk_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_scores[:6]  # top 6 patients


# Global engine instance (thread-safe lazy init)
_engine = None
_engine_lock = threading.Lock()


def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RiskScoreEngine()
    return _engine
