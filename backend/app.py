from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import logging
import os
import threading
import queue
import requests
import time
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Discord Webhook Configuration - MUST be defined before use
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

try:
    from backend.db import init_db, insert_vital, get_latest_vitals, get_top_patients
    from backend.inference import get_engine, FEATURES
    from backend.training import training_manager
except ImportError:
    from db import init_db, insert_vital, get_latest_vitals, get_top_patients
    from inference import get_engine, FEATURES
    from training import training_manager

app = FastAPI()
init_db()
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
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "120"))
# Minimum risk required to send risk-based Discord alerts (0-100)
MIN_DISCORD_RISK = int(os.getenv("MIN_DISCORD_RISK", "90"))

# Internal cache for last-sent timestamps (ingest runs on a threadpool)
_last_alert_sent = {}
_alert_lock = threading.Lock()

# Key used when applying a per-patient cooldown (aggregate alerts)
_PATIENT_COOLDOWN_KEY = "__patient_alert__"


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
    # Fire in background thread to avoid blocking the request
    threading.Thread(target=send_discord_alert, args=(message,), daemon=True).start()


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


@app.get("/health")
def health():
    model_loaded = inference_engine.model is not None or bool(inference_engine.models)
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "ensemble_members": len(inference_engine.models),
        "degraded": inference_engine.degraded,
        "load_error": inference_engine.load_error,
    }


@app.post("/ingest")
def ingest_vital(vital: VitalRecord):
    """Ingest a vital sign reading, compute risk score, and store."""
    vital_dict = vital.model_dump(exclude_none=True)

    # Compute risk score via inference engine (None = model missing/failed)
    risk_score = inference_engine.add_vital(vital_dict['patient_id'], vital_dict)
    if risk_score is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable: inference failed or no model loaded. "
                   f"load_error={inference_engine.load_error}",
        )
    vital_dict['risk_score'] = risk_score

    # Store to database
    insert_vital(vital_dict)
    _dispatch_discord_live_alerts(vital_dict, risk_score)

    return {"patient_id": vital_dict['patient_id'], "risk_score": risk_score, "stored": True}


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


@app.get("/metrics")
def get_metrics():
    """Return deployed-model metrics with a stable schema for the dashboard.

    Canonical source is ml/metrics.json (deployed ensemble). Keys are mapped
    to the flat schema the frontend expects: auc/accuracy/precision/recall.
    """
    metrics_path = os.path.join('ml', 'metrics.json')
    if not os.path.exists(metrics_path):
        # Try run directories
        import glob
        runs = sorted(glob.glob('ml/training_runs/run_*/metrics.json'), key=os.path.getmtime, reverse=True)
        if runs:
            metrics_path = runs[0]
        else:
            return {"error": "No trained model metrics found"}
    with open(metrics_path) as f:
        raw = json.load(f)
    mapped = dict(raw)
    mapped["auc"] = raw.get("fresh_holdout_auc", raw.get("val_auc", raw.get("auc")))
    mapped["accuracy"] = raw.get("fresh_holdout_accuracy", raw.get("val_accuracy", raw.get("accuracy")))
    mapped["recall"] = raw.get("fresh_holdout_recall", raw.get("val_recall", raw.get("recall")))
    mapped["precision"] = raw.get("fresh_holdout_precision", raw.get("val_precision", raw.get("precision")))
    return mapped


@app.get("/patient/{patient_id}/explain")
def explain_patient(patient_id: str):
    """Return SHAP feature importance and attention weights for a patient.

    SHAP runs on exactly the normalized window the primary model scores, so
    the explanation cannot disagree with the preprocessing behind the score.
    """
    if inference_engine.model is None:
        return {"error": "No model loaded"}
    X = inference_engine.get_model_input(patient_id)
    if X is None:
        return {"error": f"No data for patient {patient_id}"}

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

@app.post("/training/start")
def start_training(config: TrainingConfig):
    """Start a new training job with the provided configuration."""
    config_dict = config.model_dump()
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
    all_jobs = training_manager.get_all_jobs()
    return {
        "jobs": [job.to_dict() for job in all_jobs]
    }


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
