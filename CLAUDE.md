# CLAUDE.md — SynCura Project Guide

SynCura is a **Predictive ICU Monitoring System**: an Attention-based LSTM predicts patient
deterioration risk in real time from continuous vitals + labs, served by a FastAPI backend and
visualized in a React/Vite dashboard, with SHAP + attention explainability and Discord/Telegram
alerting. Trained on the PhysioNet 2012 Challenge dataset.

This file is the up-to-date orientation doc. `AGENTS.md` also exists but describes an **older**
6-feature/60-minute-window model — prefer this file where the two disagree (see "Known drift" below).

## Current model state (check `ml/metrics.json` for the live numbers)

- **12 features**, **90-minute window**, config name `f12-h96-w90` (12 features, hidden=96, window=90).
- Features: `HR, RespRate, Temp, NISysABP, NIDiasABP, SpO2, GCS, BUN, Creatinine, WBC, Platelets, Glucose`.
- AUC 0.798, accuracy 0.771, recall 0.681, precision 0.375 (11 epochs, early-stopped).
- Got here via iterative grid search (30+ configs, 5 rounds) — see commit `a491e11` for the full story:
  6→12 features, window 60→90, hidden 64→96, population normalization, proximity labeling, SaO2→SpO2 alias.
- `ml/scaler.json` holds the population mean/std from the training split — inference must use these,
  not per-window stats (that was a fixed bug, see commit `c4203db`).

## ⚠️ Known drift: backend/inference.py is NOT in sync with the current model

`backend/inference.py` hardcodes `FEATURES = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'SpO2']`
(6 features) and `window_size=60`, but `ml/models/lstm_baseline.pt` was last trained with the 12-feature,
90-minute config above. Loading that checkpoint into a 6-feature `AttentionLSTMModel` will fail or
silently misbehave. **Before relying on live `/ingest` risk scores, reconcile `FEATURES`/`window_size`
in `backend/inference.py` (and the SHAP path in `backend/app.py`'s `/explain` endpoint) with whatever
model is actually in `ml/models/lstm_baseline.pt`.** Always update `ml/train.py` and
`backend/inference.py` together when changing features or window size — this is the #1 way this
project breaks silently.

## Directory map

```
ml/               ML pipeline (PyTorch)
  train.py            Main training script — early stopping, proximity labeling, population norm
  train_lstm.py       LSTMModel + AttentionLSTMModel definitions
  dataset.py          PhysioNet loader, sliding-window construction
  preprocess.py       Z-score normalization, NaN interpolation
  explain.py          SHAP (KernelSHAP) feature importance
  run_experiments.py  Automated grid search across configs (used to reach AUC 0.798)
  scaler.json         Population mean/std from training split (required by inference)
  metrics.json        Latest run's metrics — mirrors ml/training_runs/run_*/metrics.json
  models/lstm_baseline.pt   Deployed checkpoint

backend/          FastAPI REST API
  app.py              Endpoints: /health /ingest /patients /patient/{id} /scores /metrics
                       /patient/{id}/explain, plus /training/* job-management endpoints
  inference.py        RiskScoreEngine — loads model, per-patient vital buffer, thread-safe scoring
  training.py         TrainingManager — background training jobs with progress streaming
  db.py               SQLite (backend/data/vitals.db)
  replay.py           Replays PhysioNet data into the API over HTTP or MQTT (demo/testing)
  mqtt_subscriber.py  MQTT subscriber for real sensor ingest

frontend/         React 18 + Vite + Tailwind
  src/App.jsx                 Shell, routing, dashboard layout
  src/simulationContext.jsx   Client-side simulation engine (12 synthetic patients) — NOT wired to backend
  src/components/
    WelcomePage.jsx        Landing page — check for hardcoded stats before trusting them
    SensorWaveform.jsx     SVG waveform + explainability overlay
    TrainingConfig.jsx / TrainingMonitor.jsx / TrainingJobsList.jsx   Training job UI
    SimulatedDataFeed.jsx  Tabular live-data view
    ArchitecturePage.jsx   In-app architecture docs

discordbot/, chatbot-tele/   Discord and Telegram alert bots
firmware/esp32_max30105/     ESP32 + MAX30105 sensor firmware (publishes over MQTT)
scripts/                     One-off utilities (export demo patients, test ingest, purge alerts)
```

## Architecture

```
Sensors / PhysioNet replay --> POST /ingest --> SQLite (vitals.db)
                                    |
                                    v
                          AttentionLSTM (RiskScoreEngine)
                                    |
                                    v
                           risk_score (0-100)
                          +---------+---------+
                          |                   |
                 React Dashboard      Discord / Telegram alerts
              (GET /patients /scores /explain)
```

## Common commands

```powershell
# Train (full set-a; see README.md for exact PhysioNet paths)
python -m ml.train --physionet "<path>\set-a" --outcomes "<path>\Outcomes-a.txt" `
  --epochs 25 --patience 7 --stride 15 --batch-size 128 --lr 0.0003

# Smoke test only (fast, not reportable)
python -m ml.train --physionet "<path>\set-a" --outcomes "<path>\Outcomes-a.txt" `
  --max-patients 100 --epochs 2

# Backend
pip install -r backend\requirements.txt
uvicorn backend.app:app --reload --port 8000

# Frontend
cd frontend; npm install; npm run dev

# Both at once (Windows)
.\start-dev.ps1

# Replay PhysioNet data into a running backend
python backend\replay.py --mode http --url http://localhost:8000/ingest --physionet "<path>\set-a" --speed 10 --max-patients 5
```

## Key API endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | status |
| POST | `/ingest` | vital JSON in, `{patient_id, risk_score, stored}` out |
| GET | `/patients` | top 6 by risk |
| GET | `/patient/{id}` | details + recent vitals |
| GET | `/scores` | all live scores |
| GET | `/metrics` | reads `ml/metrics.json` (falls back to latest `ml/training_runs/run_*/metrics.json`) |
| GET | `/patient/{id}/explain` | SHAP importance + attention weights — depends on `FEATURES` in `inference.py` matching the loaded model |
| POST | `/training/start`, GET `/training/jobs`, GET `/training/{id}/progress` | background training job lifecycle (NDJSON progress stream) |

## Conventions & gotchas

- **Python**: match existing style, no comments unless the logic is genuinely non-obvious.
- **JS/JSX**: functional components + hooks, Tailwind utility classes.
- No new dependencies without checking what's already in `requirements.txt`/`package.json` first.
- `RiskScoreEngine` (backend/inference.py) uses `threading.Lock()` — keep any changes thread-safe.
- Frontend `simulationContext.jsx` is entirely client-side synthetic data; it does **not** read live
  scores back from the backend. Don't assume dashboard numbers reflect real `/scores` output.
- `WelcomePage.jsx` may have hardcoded model stats — verify against `/metrics` before citing its numbers.
- `chart.js` and `socket.io-client` are declared in `frontend/package.json` but currently unused.
- No automated test suite exists yet (backend or ML).
- `.env` holds secrets (Discord webhook, etc.) — never print or commit its contents; `.env.example` is the template.

## Docs already in the repo (read these for depth, this file is the map)

- `README.md` — setup/run instructions, feature log of implemented capabilities.
- `AGENTS.md` — earlier project guide; **feature/window numbers are stale** (says 6 features/60 min), rest is broadly accurate.
- `PROJECT_REPORT.md` — full mini-project report: base paper, literature review, methodology, results tables (also has some stale numbers — e.g. results section predates the 12-feature run; `ml/metrics.json` is the source of truth for current metrics).
- `LITERATURE_REVIEW.md` — base paper (Wang, Bai & Jin 2026) + 8 supporting papers, with SynCura's positioning (~0.78–0.93 AUC realistic range vs. richer multi-database systems reaching 0.93–0.97).
- `zulfapp1 (1) (2).pptx` — academic presentation deck for this project (P.A. College of Engineering, CSE dept). Rebuilt 2026-09-14 to reflect the current SynCura content (12-feature model, AUC 0.798, current architecture) — it previously contained an unrelated mushroom-contamination-detection project's slides by mistake. If asked to update it again, source facts from `ml/metrics.json` and this file rather than the stale figures in `PROJECT_REPORT.md`/`AGENTS.md`.
