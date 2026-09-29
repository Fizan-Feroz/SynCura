from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import json
import logging
import glob
import os
import platform
import threading
import queue
import requests
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Discord Webhook Configuration - MUST be defined before use
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

try:
    from backend.dashboard import DASHBOARD_HTML
    from backend.db import init_db, insert_vital, get_latest_vitals, get_top_patients, purge_old_vitals
    from backend.inference import get_engine, FEATURES
    from backend.simulation import get_sim_engine
    from backend.training import training_manager
except ImportError:
    from dashboard import DASHBOARD_HTML
    from db import init_db, insert_vital, get_latest_vitals, get_top_patients, purge_old_vitals
    from inference import get_engine, FEATURES
    from simulation import get_sim_engine
    from training import training_manager

app = FastAPI()
init_db()
try:
    _purged = purge_old_vitals()
    if _purged:
        logging.getLogger("syncura.alerts").info("Purged %d old vitals rows at startup", _purged)
except Exception as e:
    logging.getLogger("syncura.alerts").warning("Startup vitals purge failed: %s", e)
logger = logging.getLogger("syncura.alerts")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(_handler)

# Log Discord webhook configuration at startup
if DISCORD_WEBHOOK_URL:
    logger.info("Discord webhook configured (URL length: %d chars)", len(DISCORD_WEBHOOK_URL))
else:
    logger.warning("Discord webhook NOT configured. Set DISCORD_WEBHOOK_URL in .env for alerts")

# Comma-separated allow-list, e.g. "http://localhost:5173". Browsers reject a
# wildcard origin combined with credentials, so credentials are only enabled
# for an explicit allow-list.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_engine = get_engine()
# Shared scenario engine: every browser sees the same beds. Ticks score real
# model risk via the inference engine; patient ids are sim-namespaced.
sim_engine = get_sim_engine()
sim_engine.set_scorer(inference_engine.score_patient_batch)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START_TIME = time.time()
INGEST_COUNT = 0
LAST_INGEST_TIME = None
# Timestamps of recent ingests (capped) for throughput rates. Guarded by _INGEST_LOCK.
_INGEST_TIMES = deque(maxlen=1200)
_INGEST_LOCK = threading.Lock()
def _env_int(name, default, minimum=None, maximum=None):
    """Read an int env var defensively: bad values fall back to the default."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("Invalid %s; using default %s", name, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("Invalid %s=%s; clamping to %s", name, value, minimum)
        return minimum
    if maximum is not None and value > maximum:
        logger.warning("Invalid %s=%s; clamping to %s", name, value, maximum)
        return maximum
    return value


ALERT_COOLDOWN_SECONDS = _env_int("ALERT_COOLDOWN_SECONDS", 120, minimum=1)
# Minimum risk required to send risk-based Discord alerts (0-100)
MIN_DISCORD_RISK = _env_int("MIN_DISCORD_RISK", 90, minimum=0, maximum=100)

# Internal cache for last-sent timestamps (ingest runs on a threadpool)
_last_alert_sent = {}
_alert_lock = threading.Lock()

# Key used when applying a per-patient cooldown (aggregate alerts)
_PATIENT_COOLDOWN_KEY = "__patient_alert__"

# Bounded pool for Discord webhook posts (see _dispatch_live_alerts).
_ALERT_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="discord-alert")


def _read_json_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def website_version():
    package = _read_json_file(os.path.join(REPO_ROOT, "frontend", "package.json"))
    version = package.get("version")
    return version if isinstance(version, str) and version else "unknown"


def deployed_model_info():
    manifest = _read_json_file(os.path.join(REPO_ROOT, "ml", "deployed_manifest.json"))
    return {
        "model_id": manifest.get("model_id", "unknown"),
        "val_auc": manifest.get("val_auc"),
    }


def uptime_seconds():
    return time.time() - START_TIME


def uptime_human(seconds=None):
    total = int(seconds if seconds is not None else uptime_seconds())
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def runtime_throughput():
    """Throughput snapshot for the admin panel: totals plus 1-min, 5-min, and
    since-boot ingest rates. Entries older than 5 minutes are pruned on read."""
    now = time.time()
    with _INGEST_LOCK:
        while _INGEST_TIMES and now - _INGEST_TIMES[0] > 300:
            _INGEST_TIMES.popleft()
        recent = list(_INGEST_TIMES)
        total = INGEST_COUNT
        last = LAST_INGEST_TIME
    per_min_1m = sum(1 for t in recent if now - t <= 60)
    per_min_5m = round(len(recent) / 5, 2)
    uptime_min = max(uptime_seconds() / 60, 1 / 60)
    return {
        "ingest_count": total,
        "last_ingest_time": last,
        "ingest_per_min_1m": per_min_1m,
        "ingest_per_min_5m": per_min_5m,
        "ingest_per_min_avg": round(total / uptime_min, 2),
    }


def send_discord_alert(message: str):
    """Send alert to Discord via webhook (synchronous)."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=5
        )
        response.raise_for_status()
        logger.info("Alert sent to Discord: %s", message[:50])
    except Exception as exc:
        logger.warning("Failed to send Discord alert: %s", exc)


def _should_send_alert(patient_id: str, alert_key: str) -> bool:
    """Return True if an alert identified by (patient_id, alert_key) may be sent.

    The `alert_key` can be a specific alert text or a special patient-level key
    (see `_PATIENT_COOLDOWN_KEY`) to apply a cooldown for all alerts for a
    patient (aggregation).
    """
    now = time.time()
    cache_key = (patient_id, alert_key)
    with _alert_lock:
        # Prune expired entries so the cache cannot grow without bound.
        expired = [k for k, ts in _last_alert_sent.items() if now - ts >= ALERT_COOLDOWN_SECONDS]
        for k in expired:
            del _last_alert_sent[k]
        last_sent = _last_alert_sent.get(cache_key, 0)
        if (now - last_sent) < ALERT_COOLDOWN_SECONDS:
            return False
        _last_alert_sent[cache_key] = now
        return True


def _should_send_patient_alert(patient_id: str) -> bool:
    """Helper: apply cooldown at the patient level (aggregate alerts)."""
    return _should_send_alert(patient_id, _PATIENT_COOLDOWN_KEY)


def _build_live_alerts(vital: dict, risk_score: float):
    patient_id = vital.get("patient_id", "unknown")
    alerts = []
    if risk_score >= 90:
        alerts.append(
            ("critical", f"Patient {patient_id} at {risk_score:.1f}% risk - immediate bedside review needed.")
        )
    spo2 = vital.get("SpO2")
    rr = vital.get("RespRate")
    temp = vital.get("Temp")
    if spo2 is not None and spo2 <= 88:
        alerts.append(("warning", f"Patient {patient_id} has low SpO2 ({spo2}%)."))
    if rr is not None and rr >= 30:
        alerts.append(("warning", f"Patient {patient_id} respiratory rate elevated ({rr}/min)."))
    if temp is not None and temp >= 39:
        alerts.append(("info", f"Patient {patient_id} temperature trend suggests infection ({temp} C)."))
    return alerts


def _dispatch_discord_live_alerts(vital: dict, risk_score: float):
    """Dispatch alerts to Discord via webhook in a background thread."""
    if not DISCORD_WEBHOOK_URL:
        return

    patient_id = vital.get("patient_id", "unknown")

    alerts = _build_live_alerts(vital, risk_score)
    if not alerts:
        return

    # Optionally filter by minimum risk for purely risk-based notifications
    if risk_score < MIN_DISCORD_RISK:
        # allow SpO2/RespRate/Temp warnings to still be notified
        non_risk_alerts = [a for a in alerts if a[0] != 'critical']
        if not non_risk_alerts:
            return
        alerts = non_risk_alerts

    # Aggregate alert texts and compute highest severity
    levels = {'info': 0, 'warning': 1, 'critical': 2}
    highest = 'info'
    texts = []
    for level, text in alerts:
        texts.append(text)
        if levels.get(level, 0) > levels.get(highest, 0):
            highest = level

    # Apply patient-level cooldown to ALL severities (including critical).
    # The first critical alert for a patient is always delivered because the
    # cooldown key is empty; repeats are suppressed for ALERT_COOLDOWN_SECONDS.
    # This prevents a crashing patient from flooding the channel on every ingest.
    if not _should_send_patient_alert(patient_id):
        logger.debug("Suppressed alerts for %s due to cooldown", patient_id)
        return

    message = f"[SynCura {highest.upper()}] Patient {patient_id}: " + "; ".join(texts)
    # Bounded worker pool (not a thread per alert) so a flood of ingests
    # cannot exhaust threads; the webhook POST has its own 5s timeout.
    _ALERT_POOL.submit(send_discord_alert, message)


class VitalRecord(BaseModel):
    patient_id: str
    timestamp: float
    HR: Optional[float] = None
    RespRate: Optional[float] = None
    Temp: Optional[float] = None
    NISysABP: Optional[float] = None
    NIDiasABP: Optional[float] = None
    SpO2: Optional[float] = None
    EtCO2: Optional[float] = None
    GCS: Optional[float] = None
    BUN: Optional[float] = None
    Creatinine: Optional[float] = None
    WBC: Optional[float] = None
    Platelets: Optional[float] = None
    Glucose: Optional[float] = None


class TrainingConfig(BaseModel):
    physionet_path: str
    outcomes_path: str
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    max_patients: int = 100
    # Deployment contract: 12 features, 90-min window, hidden 96. Custom
    # configs train into isolated job artifacts and NEVER overwrite serving.
    vital_features: list = ["HR", "RespRate", "Temp", "NISysABP", "NIDiasABP", "SpO2",
                            "GCS", "BUN", "Creatinine", "WBC", "Platelets", "Glucose"]
    window: int = 90
    hidden_size: int = 96
    stride: int = 15
    label_mode: str = "proximity"
    horizon_hours: float = 12.0


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    """Human-readable status page. Renders the same data as /admin/status."""
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/health")
def health():
    model_loaded = inference_engine.model is not None or bool(inference_engine.models)
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "ensemble_members": len(inference_engine.models),
        "degraded": inference_engine.degraded,
        "load_error": inference_engine.load_error,
        "uptime_seconds": round(uptime_seconds(), 1),
        "uptime_human": uptime_human(),
        "server_time": time.time(),
    }


@app.get("/version")
def version():
    """Version-tracking endpoint for website and deployment checks."""
    model = deployed_model_info()
    return {
        "service": "syncura-backend",
        "website_version": website_version(),
        "model_id": model["model_id"],
        "model_val_auc": model["val_auc"],
        "git_commit": os.getenv("RENDER_GIT_COMMIT", "unknown"),
        "render_service": os.getenv("RENDER_SERVICE_NAME", "unknown"),
        "render_external_url": os.getenv("RENDER_EXTERNAL_URL", ""),
        "python_version": platform.python_version(),
        "uptime_seconds": round(uptime_seconds(), 1),
        "uptime_human": uptime_human(),
    }


@app.get("/admin/status")
def admin_status():
    """Machine-readable service status for the /admin panel.

    Aggregates uptime, hosting (Render), model, and runtime counters so the
    frontend can render uptime/downtime, latency, Vercel hosting, and Render
    detail from live probes without inventing numbers.
    """
    model_loaded = inference_engine.model is not None or bool(inference_engine.models)
    model = deployed_model_info()
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "uptime_seconds": round(uptime_seconds(), 1),
        "uptime_human": uptime_human(),
        "server_time": time.time(),
        "started_at": START_TIME,
        "backend": {
            "service": "syncura-backend",
            "website_version": website_version(),
            "git_commit": os.getenv("RENDER_GIT_COMMIT", "unknown"),
            "render_service": os.getenv("RENDER_SERVICE_NAME", "unknown"),
            "render_external_url": os.getenv("RENDER_EXTERNAL_URL", ""),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cors_origins": CORS_ORIGINS,
        },
        "model": {
            "model_id": model["model_id"],
            "model_val_auc": model["val_auc"],
            "ensemble_members": len(inference_engine.models),
            "degraded": inference_engine.degraded,
            "load_error": inference_engine.load_error,
        },
        "simulation": {
            "scenario": sim_engine.snapshot().get("scenario"),
            **sim_engine.tick_stats(),
        },
        "runtime": runtime_throughput(),
    }


@app.post("/ingest")
def ingest_vital(vital: VitalRecord):
    """Ingest a vital sign reading, compute risk score, and store."""
    vital_dict = vital.model_dump(exclude_none=True)

    # Compute risk score via inference engine (None = model missing/failed)
    try:
        risk_score = inference_engine.add_vital(vital_dict['patient_id'], vital_dict)
    except Exception as e:
        logger.exception("Ingest scoring failed")
        raise HTTPException(status_code=503, detail=f"Scoring failed: {e}")
    if risk_score is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable: inference failed or no model loaded. "
                   f"load_error={inference_engine.load_error}",
        )
    vital_dict['risk_score'] = risk_score

    # Store to database
    try:
        insert_vital(vital_dict)
    except Exception as e:
        logger.exception("Ingest store failed")
        raise HTTPException(status_code=500, detail=f"Store failed: {e}")
    global INGEST_COUNT, LAST_INGEST_TIME
    with _INGEST_LOCK:
        now = time.time()
        INGEST_COUNT += 1
        LAST_INGEST_TIME = now
        _INGEST_TIMES.append(now)
    _dispatch_discord_live_alerts(vital_dict, risk_score)

    return {"patient_id": vital_dict['patient_id'], "risk_score": risk_score, "stored": True}


class DemoAlert(BaseModel):
    patient_id: str
    level: str  # info | warning | critical
    text: str
    bed: Optional[str] = None
    source: Optional[str] = "demo"  # demo | live-simulated


@app.post("/alerts/demo")
def post_demo_alert(alert: DemoAlert):
    """Forward a simulation alert to Discord.

    Demo/local browsers must never hold the webhook secret, so they POST
    here and the backend applies the same per-patient cooldown as live
    alerts. Messages are always tagged [Demo ...] so a simulated alert can
    never be mistaken for a real bedside event. Replay sources must NOT
    use this endpoint (their alerts already flow through /ingest).
    """
    level = (alert.level or "").lower()
    if level not in ("info", "warning", "critical"):
        raise HTTPException(status_code=422, detail="level must be info|warning|critical")
    text = (alert.text or "").strip()[:500]
    if not text:
        raise HTTPException(status_code=422, detail="text must be non-empty")
    source = (alert.source or "demo").strip().lower() or "demo"
    if source not in ("demo", "live-simulated"):
        raise HTTPException(status_code=422, detail="source must be demo|live-simulated")
    if not DISCORD_WEBHOOK_URL:
        return {"sent": False, "reason": "webhook-not-configured"}
    if not _should_send_patient_alert(f"demo:{alert.patient_id}"):
        return {"sent": False, "reason": "cooldown"}
    bed = f" {alert.bed}" if alert.bed else ""
    message = f"[Demo {level.upper()}] Patient {alert.patient_id}{bed} ({source}): {text}"
    _ALERT_POOL.submit(send_discord_alert, message)
    return {"sent": True}


@app.get("/patients")
def get_patients():
    """Return top 6 patients by current risk score."""
    top_patients = get_top_patients(6)
    return {
        "patients": [
            {"patient_id": pid, "risk": risk, "timestamp": ts}
            for pid, risk, ts in top_patients
        ]
    }


@app.get("/patient/{patient_id}")
def get_patient(patient_id: str):
    """Get patient details and recent vitals."""
    vitals = get_latest_vitals(patient_id, limit=20)
    risk = inference_engine.get_risk_score(patient_id)
    def _val(row, idx):
        try:
            return row[idx]
        except IndexError:
            return None

    return {
        "patient_id": patient_id,
        "risk": risk,
        "vitals": [
            {
                "timestamp": v[0],
                "hr": v[1],
                "spo2": v[2],
                "rr": v[3],
                "systolic": v[4],
                "diastolic": v[5],
                "temp": v[6],
                "etco2": v[7],
                "risk_score": v[8],
                "gcs": _val(v, 9),
                "bun": _val(v, 10),
                "creatinine": _val(v, 11),
                "wbc": _val(v, 12),
                "platelets": _val(v, 13),
                "glucose": _val(v, 14),
            }
            for v in vitals
        ]
    }


@app.get("/scores")
def get_scores():
    """Get all live risk scores."""
    return inference_engine.get_all_scores()


class SimControl(BaseModel):
    action: str  # start | pause | reset | set_scenario
    scenario: Optional[str] = None


@app.get("/simulation/state")
def simulation_state():
    """Shared scenario state: identical beds for every connected browser."""
    return sim_engine.snapshot()


@app.post("/simulation/control")
def simulation_control(cmd: SimControl):
    """Control the shared simulation (open to all visitors by decision)."""
    try:
        return sim_engine.control(cmd.action, cmd.scenario)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/replay/sample")
def replay_sample():
    """One-click replay seed: snapshot the shared engine's current beds into
    the ingest store so the Replay view has something to show without any
    local setup. Honestly labeled — these are engine beds, not retrospective
    patient rows. Real PhysioNet replay still goes through backend/replay.py."""
    snap = sim_engine.snapshot()
    now = time.time()
    stored = 0
    for bed in snap.get("beds", []):
        patient_id = bed.get("patient_id")
        if not patient_id:
            continue
        vitals = bed.get("vitals", {}) or {}
        try:
            insert_vital({
                "patient_id": patient_id,
                "timestamp": now,
                "HR": vitals.get("HR"),
                "SpO2": vitals.get("SpO2"),
                "RespRate": vitals.get("Resp"),
                "Temp": vitals.get("Temp"),
                "risk_score": bed.get("risk", 0),
            })
        except Exception as e:
            logger.warning("replay/sample store failed for %s: %s", patient_id, e)
            continue
        stored += 1
    return {"stored": stored, "source": "simulation-engine-snapshot"}


@app.get("/metrics")
def get_metrics():
    """Return deployed-model metrics with a stable schema for the dashboard.

    Canonical source is ml/metrics.json (deployed ensemble). Keys are mapped
    to the flat schema the frontend expects: auc/accuracy/precision/recall.
    """
    metrics_path = os.path.join('ml', 'metrics.json')
    if not os.path.exists(metrics_path):
        # Try run directories
        runs = sorted(glob.glob('ml/training_runs/run_*/metrics.json'), key=os.path.getmtime, reverse=True)
        if runs:
            metrics_path = runs[0]
        else:
            return {"error": "No trained model metrics found"}
    try:
        with open(metrics_path) as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("Failed to read metrics file %s: %s", metrics_path, e)
        return {"error": f"Metrics file unreadable: {metrics_path}"}
    if not isinstance(raw, dict):
        return {"error": f"Metrics file malformed: {metrics_path}"}
    mapped = dict(raw)
    # Prefer the ONE-TIME locked evaluation when present (v5 Step 5: the UI
    # reports the honest number, not the selection-influenced holdout).
    mapped["auc"] = raw.get("locked_auc", raw.get("fresh_holdout_auc", raw.get("val_auc", raw.get("auc"))))
    mapped["accuracy"] = raw.get("locked_accuracy", raw.get("fresh_holdout_accuracy", raw.get("val_accuracy", raw.get("accuracy"))))
    mapped["recall"] = raw.get("locked_recall", raw.get("fresh_holdout_recall", raw.get("val_recall", raw.get("recall"))))
    mapped["precision"] = raw.get("locked_precision", raw.get("fresh_holdout_precision", raw.get("val_precision", raw.get("precision"))))
    mapped["auc_source"] = ("locked-once" if raw.get("locked_auc") is not None
                            else "working-holdout" if raw.get("fresh_holdout_auc") is not None else "validation")
    return mapped


@app.get("/patient/{patient_id}/explain")
def explain_patient(patient_id: str):
    """Return SHAP feature importance and attention weights for a patient.

    SHAP runs on exactly the normalized window the primary model scores, so
    the explanation cannot disagree with the preprocessing behind the score.
    """
    if inference_engine.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    X = inference_engine.get_model_input(patient_id)
    if X is None:
        raise HTTPException(status_code=404, detail=f"No data for patient {patient_id}")

    attention = inference_engine.get_attention_weights(patient_id)

    # Compute SHAP (may be slow, so keep nsamples small)
    try:
        from ml.explain import compute_shap_explanation
        importance = compute_shap_explanation(inference_engine.model, X, FEATURES, n_background=10)
    except Exception as e:
        importance = {"error": str(e)}

    return {
        "patient_id": patient_id,
        "feature_importance": importance,
        "attention_weights": attention,
        "features_used": FEATURES,
    }


# ============ TRAINING ENDPOINTS ============

def _require_training_token(request: Request):
    """Gate expensive training starts when TRAINING_API_TOKEN is set.

    Unset (dev default) keeps the endpoint open so local use and CI are
    unaffected. Note this is obscurity, not access control: real protection
    for a public deployment is firewall/VPN, not a header check.
    """
    token = os.getenv("TRAINING_API_TOKEN", "").strip()
    if token and request.headers.get("X-Training-Token", "") != token:
        raise HTTPException(status_code=401, detail="Training API token required")


@app.post("/training/start")
def start_training(config: TrainingConfig, request: Request):
    """Start a new training job with the provided configuration."""
    _require_training_token(request)
    config_dict = config.model_dump()
    for key in ("physionet_path", "physionet_dir", "outcomes_path", "outcomes_file"):
        path = config_dict.get(key)
        if path and not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"Dataset path does not exist: {key}={path}")
    # Enforce the deployment contract: only the 12-feature / 90-min / h96
    # config may be promoted to serving; anything else trains in isolation.
    if (sorted(config_dict.get("vital_features", [])) != sorted(FEATURES)
            or int(config_dict.get("window", 90)) != 90
            or int(config_dict.get("hidden_size", 96)) != 96):
        config_dict["_promotable"] = False
    else:
        config_dict["_promotable"] = True
    job = training_manager.create_job(config_dict)
    if not training_manager.start_training(job.job_id):
        raise HTTPException(
            status_code=409,
            detail="Another training job is already running. Only one job at a time is supported.",
        )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "config": config_dict
    }


@app.get("/training/jobs")
def list_training_jobs():
    """Get all training jobs with their current status."""
    # get_all_jobs() already returns serialized dicts.
    return {"jobs": training_manager.get_all_jobs()}


@app.get("/training/{job_id}")
def get_training_job(job_id: str):
    """Get details of a specific training job."""
    job = training_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_dict()


@app.get("/training/{job_id}/progress")
def get_training_progress(job_id: str):
    """Get next progress update from training job (streaming)."""
    job = training_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    def progress_generator():
        while True:
            try:
                progress_update = job.progress_queue.get(timeout=5)
                yield json.dumps(progress_update) + "\n"
            except queue.Empty:
                current_job = training_manager.get_job(job_id)
                if current_job:
                    yield json.dumps({
                        "status": current_job.status,
                        "current_epoch": current_job.current_epoch,
                        "total_epochs": current_job.total_epochs,
                        "metrics": current_job.metrics
                    }) + "\n"
                if current_job and current_job.status in ["completed", "failed"]:
                    break

    return StreamingResponse(progress_generator(), media_type="application/x-ndjson")
