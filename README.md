# SynCura — Predictive ICU Monitoring System

> Real-time ICU **in-hospital mortality-risk** research prototype using an attention-based LSTM, served through a FastAPI backend and a React dashboard with SHAP + temporal-attention explainability.
> (Labels are PhysioNet `In-hospital_death`; "deterioration risk" in the UI means mortality risk unless a separate deterioration label is defined.)

![Python](https://img.shields.io/badge/python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-REST-green) ![React](https://img.shields.io/badge/React-18-61dafb) ![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-orange) ![Status](https://img.shields.io/badge/status-research%20prototype-yellow)

## Results (deployed model)

3-model logit-averaged ensemble (`s48 + c93 + s45`, 12 features, 90-min windows), per `ml/deployed_manifest.json` and `ppt/figs/metrics.json`:

| Split | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|
| Validation | **0.840** | 0.738 | 0.816 | 0.724 | 0.357 | 0.496 |
| Fresh unseen 20% set-B holdout | **0.844** (95% CI 0.836–0.852) | 0.747 | 0.807 | 0.737 | 0.345 | 0.483 |

**Read these numbers honestly:**
- `c93` trained on set-a + 80% of set-b, so full set-b evaluation is N/A; the 20% set-b holdout was unseen by weights but used during ensemble selection — not a locked final test.
- The CI is a window-level bootstrap.
- Repeated gating on the same validation split makes validation AUC optimistic — keep the holdout gate on every deploy decision.
- `ml/dataset.py` uses causal per-window interpolation (no future leakage). Checkpoints trained before that fix must be retrained for comparable numbers.

## Architecture

```
Patient Vitals --> [Backend /ingest] --> [SQLite DB]
                     |
                     v
            [AttentionLSTM Ensemble]
            (2-layer LSTM-96 + additive temporal attention,
             12 features x 90 timesteps, sigmoid output)
                     |
                     v
              Risk Score (0-100)
                     |
          +----------+----------+
          |                     |
     [React Dashboard]    [Discord/Telegram Alerts]
     (SHAP + attention      (threshold-based)
      explanations)
```

**Model input:** 12 features × 90 timesteps (90-minute window). Training: BCEWithLogitsLoss with pos_weight, Adam (lr 1e-4, step decay), early stopping. Population stats from train split only (`ml/scaler.json`).

### Feature set (12)

| # | Feature | PhysioNet field | Normal range |
|---|---|---|---|
| 1 | Heart Rate | HR | 60–100 bpm |
| 2 | Respiratory Rate | RespRate | 12–20 /min |
| 3 | Temperature | Temp | 36.1–37.2 °C |
| 4 | Systolic BP | NISysABP | 90–140 mmHg |
| 5 | Diastolic BP | NIDiasABP | 60–90 mmHg |
| 6 | SpO2 (via SaO2 alias) | SaO2 | 95–100% |
| 7 | GCS | GCS | 3–15 |
| 8 | BUN | BUN | 6–24 mg/dL |
| 9 | Creatinine | Creatinine | 0.6–1.2 mg/dL |
| 10 | WBC | WBC | 4.5–11 ×10³/µL |
| 11 | Platelets | Platelets | 150–450 ×10³/µL |
| 12 | Glucose | Glucose | 70–140 mg/dL |

A 20-feature variant was tested and scored worse — the 12-feature set stands.

## Repository layout

```
PROJ/
├── ml/               # Training + eval (train.py, dataset.py, sweep_*.py, pick_ensemble.py)
│   ├── models/       # Weights (gitignored; ensemble served from models/ensemble/)
│   ├── training_runs/# Timestamped run artifacts (metrics.json only)
│   ├── deployed_manifest.json  # Deployed ensemble definition
│   └── scaler.json   # Train-split normalization stats
├── backend/          # FastAPI: ingest, scoring, SHAP explain, training jobs, replay
├── frontend/         # React 18 + Vite + plain CSS (see THEME.md for the design system)
├── ppt/              # Final deck (SynCura_Deck_V3_FINAL.pptx), figs/metrics.json, script
├── discordbot/       # Discord alert bot
├── chatbot-tele/     # Telegram chatbot
├── firmware/         # IoT firmware
├── LITERATURE_REVIEW.md # Literature review (current; chronological table + DOIs)
├── PROJECT_REPORT.md # Full project report
└── start-dev.ps1     # Launch backend + frontend together (Windows)
```

> Contributor guide for AI agents: see `AGENTS.md`.

## Datasets

Large data files are **never committed** — they live on the team Google Drive and are fetched with one command:

```powershell
pip install gdown
python scripts/download_data.py --folder-id <drive-folder-id>   # or set SYNCURA_DATA_FOLDER_ID
```

| Dataset | Contents | Status |
|---|---|---|
| PhysioNet 2012 Challenge | 8,000 ICU stays; current training + holdout data | `data/predicting-mortality-...-2012-1.0.0/` (see `ml/paths.py`) |
| PhysioNet Challenge 2019 (sepsis) | 20,336 ICU stays, hourly vitals/labs, per-hour sepsis labels; 11/12 SynCura features map directly (no GCS) | On Drive → lands in `data/challenge-2019-1.0.0/` (verified by file count) |
| MIMIC-III / MIMIC-IV / eICU demos (100 pts / 2.5k stays) | Schema reference + ETL testbeds only — too small to train on | `data/mimic-*-demo*/`, `data/eicu-crd-demo-2.0.1/` |
| Simulated + Kaggle snapshots | Pipeline benchmarking and tabular baselines only — never clinical evidence | `data/hospital-deterioration-dataset/`, `data/kaggle-*/` |
| MIMIC-IV full / eICU / HiRID | Credentialed (PhysioNet login + CITI + DUA); **never re-upload — DUAs forbid redistribution** | Each teammate credentials individually |

Full inventory, coverage notes, and rules for adding datasets: **`DATA.md`**. Column-level schemas for every dataset, the canonical 12-feature table, and the exact proximity-label definition: **`docs/DATASET_SCHEMAS.md`**.

## Quickstart

### 0) Prerequisites

- Python 3.10, Node.js 18+, PowerShell 5.1+ (Windows)
- PhysioNet 2012 Challenge dataset (set-a for training, set-b for holdout)

### 1) Train the model

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml\requirements.txt
python -m ml.train `
  --physionet "<data>\set-a" `
  --outcomes "<data>\Outcomes-a.txt" `
  --epochs 25 `
  --patience 7 `
  --stride 15 `
  --batch-size 128 `
  --lr 0.0003
```

Full set-a (4,000 patients), 90-minute windows, proximity labeling (last 12h of stay), population normalization, early stopping. Metrics → `ml/training_runs/exp_*/<config>/metrics.json`; model → `ml/models/lstm_baseline.pt`; scaler → `ml/scaler.json`.

Smoke test only (not reportable): add `--max-patients 100 --epochs 2`.

### 2) Start the backend

```powershell
pip install -r backend\requirements.txt
uvicorn backend.app:app --reload --port 8000
```

Loads the 3-model ensemble from `ml/models/ensemble/*.pt` (falls back to `lstm_baseline.pt`), initializes SQLite at `backend/data/vitals.db`, serves on `http://localhost:8000`.

### 3) Replay patient data (new terminal)

```powershell
pip install pandas requests paho-mqtt
python backend\replay.py `
  --mode http `
  --url http://localhost:8000/ingest `
  --physionet "<data>\set-a" `
  --speed 10 `
  --max-patients 5
```

Streams 5 patients' vitals at 10× speed into the backend.

### 4) Run the dashboard

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173/`. Or start backend + frontend together from the repo root:

```powershell
.\start-dev.ps1
```

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | API status |
| POST | `/ingest` | Ingest vital JSON, returns risk score |
| GET | `/patients` | Top 6 patients by risk score |
| GET | `/patient/{patient_id}` | Patient details + recent vitals |
| GET | `/scores` | All live risk scores |
| GET | `/metrics` | Latest training metrics |
| GET | `/patient/{patient_id}/explain` | SHAP importance + attention weights |
| POST | `/training/start` | Start a training job |
| GET | `/training/jobs` | List training jobs |
| GET | `/training/{job_id}` | Job status |
| GET | `/training/{job_id}/progress` | Stream progress (NDJSON) |

Example:

```powershell
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" `
  -d '{"patient_id":"P001","timestamp":1000,"HR":85,"SpO2":98,"RespRate":16,"Temp":37,"NISysABP":120,"NIDiasABP":80}'
curl http://localhost:8000/patients
curl http://localhost:8000/patient/P001/explain
```

## Dashboard features

- **Live risk board** — ranked patients, risk dial, status buckets (`Stable`, `Watch`, `High`, `Critical`), trend sparklines, update timestamps
- **Scenario simulation** — `Baseline Mix`, `Respiratory Decline`, `Septic Shock`, `Cardiac Stress`, `Recovery Trend`, with pause/resume/reset and a shared `/simulated-data` feed tab
- **Alerts** — live stream + counter for critical escalation, low SpO2, high RR, fever trends
- **Explainability** — per-patient "Inspect impact": signed per-vital contributions, waveform overlay
- **Threshold tuning** — interactive risk slider with live sensitivity / specificity / precision / false alarms, NEWS2 (≥7) baseline comparison, lead-time estimate

## Documents

| Document | Path |
|---|---|
| Final presentation deck (16 slides) | `ppt/SynCura_Deck_V3_FINAL.pptx` |
| Presentation script | `ppt/SynCura_Presentation_Script.md` |
| Literature review (chronological, v1.3) | `LITERATURE_REVIEW_DOCUMENT.md` (+ `.docx`) |
| Full literature survey | `LITERATURE_REVIEW.md` |
| Project report | `PROJECT_REPORT.md` |
| Deployment notes | `DEPLOYMENT_READY.md` |

## Team

P.A. College of Engineering — Department of Computer Science & Engineering:

- Abdul Ahad Ikkeri (4PA24CS002)
- Fathima Reeha (4PA24CS026)
- Fizan Feroz (4PA24CS032)

## Limitations (prototype disclaimer)

- Research prototype: retrospective US ICU data (2012), no prospective trial, no fairness/subgroup analysis, no calibration report.
- Frontend simulation is client-side and does not read back from the backend.
- Explanations (attention/SHAP) describe model behaviour, not proven physiology.
- Not a medical device. Do not use for clinical decisions.

## Troubleshooting

- **Frontend won't load / CPU spikes:** keep one Vite server; restart with `npm run dev -- --host 127.0.0.1 --port 5173`; check port 5173 is free; `npm install; npm run build` if stale.
- **Backend can't find model:** place ensemble weights under `ml/models/ensemble/` or a fallback at `ml/models/lstm_baseline.pt` (gitignored — add with `git add -f`).
- **Full-data training is slow:** ~5–7 min/epoch at stride 15; use stride 30 for sweeps. Deployment uses window 90 regardless of training stride.
