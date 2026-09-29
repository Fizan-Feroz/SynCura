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
import logging
import math
import time
import threading

import numpy as np
import torch

from ml.dataset import SERVING_FEATURES, carry_forward

logger = logging.getLogger('syncura.inference')


def _env_float(name, default, minimum=None):
    """Read a float env var defensively: bad values fall back to the default."""
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning('Invalid %s; using default %s', name, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning('Invalid %s=%s; clamping to %s', name, value, minimum)
        return minimum
    return value


# Determine model path: allow override via MODEL_PATH env var, check common paths,
# or pick the latest saved model from ml/training_runs/*/model.pt
DEFAULT_MODEL_PATH = os.getenv('MODEL_PATH', 'ml/models/lstm_baseline.pt')
if not os.path.exists(DEFAULT_MODEL_PATH):
    alt = 'ml/lstm_baseline.pt'
    if os.path.exists(alt):
        DEFAULT_MODEL_PATH = alt
    else:
        try:
            runs = sorted(glob.glob('ml/training_runs/*/model.pt'), key=os.path.getmtime, reverse=True)
        except OSError as e:
            logger.warning('Could not scan training runs for a fallback model: %s', e)
            runs = []
        if runs:
            DEFAULT_MODEL_PATH = runs[0]

# Local alias for the canonical serving contract. Do not retype the list here.
FEATURES = SERVING_FEATURES

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
PATIENT_TTL_SECONDS = _env_float('PATIENT_TTL_SECONDS', 6 * 3600, minimum=60.0)
_EVICT_INTERVAL_SECONDS = 60.0


def canonicalize_vital(vital_dict):
    """Return a copy of `vital_dict` with alias keys mapped to model feature names.

    A canonical key that is already present always wins over an alias — unless
    it holds NaN, in which case the alias value fills the gap instead of
    blocking it.
    """
    out = {k: v for k, v in vital_dict.items() if FEATURE_ALIASES.get(str(k).lower()) in (None, k)}
    for k, v in vital_dict.items():
        canon = FEATURE_ALIASES.get(str(k).lower())
        if not canon or canon == k:
            continue
        current = out.get(canon)
        if current is None or (isinstance(current, float) and math.isnan(current)):
            out[canon] = v
    return out


def _to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return np.nan
    return f if math.isfinite(f) else np.nan


def _load_json_scaler(path, member=None):
    with open(path) as f:
        scaler = json.load(f)
    features = list(scaler.get('features', []))
    if features != list(SERVING_FEATURES):
        where = f' for member {member}' if member else ''
        raise ValueError(
            f'Scaler {path}{where} has features {features}, '
            f'but the serving contract requires {list(SERVING_FEATURES)}'
        )
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
                logger.info('[Inference] Loaded scaler stats from %s', scaler_path)
            except Exception as e:
                self.load_error = f'failed to load scaler {scaler_path}: {e}'
                self.degraded = True
                logger.warning('[Inference] %s; scores will be refused, not silently renormalized',
                               self.load_error)

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
                        self.member_scalers.append(_load_json_scaler(member['scaler'], member['id']))
                    except Exception as e:
                        self.load_error = f'scaler load failed for {member["id"]}: {e}'
                        self.degraded = True
                        logger.warning('[Inference] %s; member falls back to shared stats',
                                       self.load_error)
                        self.member_scalers.append((None, None))
                self.model = self.models[0]  # primary (attention weights source)
                logger.info('[Inference] Loaded manifest ensemble of %d models', len(self.models))
                return
            except Exception as e:
                self.load_error = f'manifest ensemble load failed: {e}'
                logger.warning('[Inference] %s; falling back', self.load_error)
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
                logger.info('[Inference] Loaded ensemble of %d models from %s', len(self.models), ens_dir)
                return
            except Exception as e:
                self.load_error = f'ensemble load failed: {e}'
                logger.warning('[Inference] %s; falling back to single model', self.load_error)
                self.models, self.member_scalers, self.model = [], [], None

        # Single model fallback.
        if os.path.exists(self.model_path):
            try:
                m = AttentionLSTMModel(input_size=len(FEATURES), hidden_size=96)
                m.load_state_dict(_load_state(self.model_path))
                m.eval()
                self.model = m
                logger.info('[Inference] Loaded AttentionLSTM from %s (%d features)',
                            self.model_path, len(FEATURES))
            except Exception as e:
                self.load_error = f'single model load failed: {e}'
                logger.warning('[Inference] %s', self.load_error)
                self.model = None
        else:
            logger.warning('[Inference] Model not found at %s; scores unavailable', self.model_path)
            self.model = None

    def _manifest(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning('[Inference] Failed to read manifest: %s', e)
        return None

    def _manifest_members(self):
        manifest = self._manifest()
        if manifest and isinstance(manifest.get('members'), list) and manifest['members']:
            arch = manifest.get('arch', {})
            need = int(arch.get('input_size', len(FEATURES)))
            if need != len(FEATURES):
                # Fail fast with a clear message (e.g. a 24-dim gap model
                # cannot be served by the 12-feature engine yet; see
                # ml/RESULTS_PLAN.md "Serving work required"). Do NOT fall
                # back to a directory scan here: silently serving stale or
                # wrong checkpoints is worse than refusing with load_error.
                self.load_error = (
                    f'manifest needs input_size={need} but engine builds '
                    f'{len(FEATURES)}-dim vectors')
                self.degraded = True
                logger.warning('[Inference] %s', self.load_error)
                return None
            members = [m for m in manifest['members']
                       if os.path.exists(m.get('checkpoint', ''))]
            if len(members) == len(manifest['members']):
                self.window_size = int(manifest.get('window_minutes', self.window_size))
                try:
                    for member in members:
                        _load_json_scaler(member['scaler'], member['id'])
                except Exception as e:
                    self.load_error = f'manifest scaler incompatible: {e}'
                    self.degraded = True
                    logger.warning('[Inference] %s', self.load_error)
                    return None
                return members
            logger.warning('[Inference] Manifest checkpoints missing; using directory scan')
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

    def build_window(self, patient_id, vital_dict):
        """Buffer one reading and return its raw (window, carried-in) tuple.

        Returns None when the patient has no buffered minutes yet. Locking
        lives inside; scoring is separate so callers can batch it.
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
            return self._window_locked(patient_id)

    def add_vital(self, patient_id, vital_dict):
        """Add a vital measurement and compute risk score.

        Returns an int 0-100, or None when no model is loaded / inference
        fails (callers must surface this instead of inventing a score).
        """
        window = self.build_window(patient_id, vital_dict)
        if window is None:
            return None
        # Model inference runs OUTSIDE the lock so one slow patient cannot
        # block ingestion for every other patient.
        risk_score = self._score_window(window)
        if risk_score is not None:
            with self.lock:
                self.risk_scores[patient_id] = risk_score
        return risk_score

    # ------------------------------------------------------------ preprocessing

    def _prepare(self, window, mean=None, std=None):
        """(raw window, carried-in values) -> normalized float32 model input (window_size, F).

        Raises ValueError when no population stats exist at all: silently
        falling back to per-window z-scoring would score with a different
        normalization than training used.
        """
        raw, initial = window
        X = carry_forward(raw, initial)
        if mean is None or std is None:
            mean, std = self._train_mean, self._train_std
        if mean is None or std is None:
            raise ValueError('no population normalization stats available')
        X = np.where(np.isnan(X), mean, X)
        return ((X - mean) / std).astype(np.float32)

    def _member_inputs(self, window):
        if self.models:
            scalers = self.member_scalers or [(None, None)] * len(self.models)
            return [(m, self._prepare(window, mean, std)) for m, (mean, std) in zip(self.models, scalers)]
        return [(self.model, self._prepare(window))]

    def _score_window(self, window):
        """Score a raw window. Returns int 0-100, or None on failure."""
        risks = self.score_windows([window])
        return risks[0]

    def score_windows(self, windows):
        """Score a batch of raw windows with one forward pass per model.

        Returns a list of int 0-100 / None aligned with the input, using
        exactly the same per-window math as the old single-window path
        (logit average across members, then sigmoid). Batching cuts per-tick
        torch overhead from O(beds x members) forwards to O(members).
        """
        models = self.models if self.models else ([self.model] if self.model is not None else [])
        if not models:
            with self.lock:
                self.degraded = True
            return [None] * len(windows)
        if not windows:
            return []
        scalers = self.member_scalers if self.models else [(None, None)]
        per_model_logits = []
        for model, (mean, std) in zip(models, scalers or [(None, None)] * len(models)):
            try:
                batch = np.stack([self._prepare(w, mean, std) for w in windows])
            except ValueError:
                per_model_logits.append(None)
                continue
            try:
                with torch.inference_mode():
                    out = model(torch.from_numpy(batch)).detach().cpu().numpy().ravel()
                per_model_logits.append(out)
            except Exception:
                logger.exception('[Inference] Error computing batch scores')
                per_model_logits.append(None)
        risks = []
        for i in range(len(windows)):
            vals = [arr[i] for arr in per_model_logits if arr is not None]
            if not vals:
                risks.append(None)
                continue
            prob = 1.0 / (1.0 + math.exp(-sum(vals) / len(vals)))
            risks.append(max(0, min(100, round(prob * 100))))
        return risks

    def score_patient_batch(self, items):
        """Buffer readings for [(patient_id, vital_dict)] and score them batched.

        Returns risks aligned with items. Used by the simulation tick so all
        beds share a handful of forward passes instead of one per bed.
        """
        windows = [self.build_window(pid, vd) for pid, vd in items]
        scored_idx = [i for i, w in enumerate(windows) if w is not None]
        risks = self.score_windows([windows[i] for i in scored_idx])
        out = [None] * len(items)
        for i, risk in zip(scored_idx, risks):
            out[i] = risk
            if risk is not None:
                with self.lock:
                    self.risk_scores[items[i][0]] = risk
        return out

    def get_model_input(self, patient_id):
        """Normalized window the PRIMARY model scores (for SHAP / attention), or None."""
        with self.lock:
            window = self._window_locked(patient_id)
        if window is None or self.model is None:
            return None
        mean, std = self.member_scalers[0] if self.member_scalers else (None, None)
        try:
            return self._prepare(window, mean, std)
        except ValueError as e:
            logger.warning('[Inference] Cannot build model input for %s: %s', patient_id, e)
            return None

    def get_attention_weights(self, patient_id):
        """Get attention weights for a patient's current window (for explainability)."""
        X = self.get_model_input(patient_id)
        if X is None:
            return None
        try:
            with torch.inference_mode():
                weights = self.model.get_attention_weights(torch.from_numpy(X[None]))
            return weights.squeeze(0).numpy().tolist()
        except Exception:
            logger.exception('[Inference] Error getting attention weights for %s', patient_id)
            return None

    def get_risk_score(self, patient_id):
        """Retrieve latest risk score for a patient."""
        with self.lock:
            return self.risk_scores.get(patient_id, 0)

    def get_all_scores(self, include_sim=False):
        """Get all patient risk scores sorted by risk (descending).

        Simulation beds (``sim-`` prefixed keys written by the shared
        scenario engine) are excluded by default so demo traffic can never
        surface in real-patient listings like GET /scores.
        """
        with self.lock:
            items = self.risk_scores.items()
            if not include_sim:
                items = [(pid, score) for pid, score in items
                         if not str(pid).startswith("sim-")]
            sorted_scores = sorted(items, key=lambda x: x[1], reverse=True)
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
