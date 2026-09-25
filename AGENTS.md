# AGENTS.md — SynCura Project Guide for AI Agents

## Project Overview

SynCura is a **Predictive ICU Monitoring System** that uses deep learning (LSTM with attention) to predict patient deterioration risk in real-time. It combines a FastAPI backend, React/Vite frontend, and PyTorch ML pipeline trained on PhysioNet 2012 ICU data.

## Quick Commands

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Train the ML model (with attention + early stopping + SpO2).
# Paths resolve via ml/paths.py -> PROJ/data (override: $env:SYNCURA_DATA_ROOT="D:\ml-data").
python ml/train.py `
  --physionet "data\predicting-mortality-of-icu-patients-the-physionetcomputing-in-cardiology-challenge-2012-1.0.0\predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0\set-a" `
  --outcomes "data\predicting-mortality-of-icu-patients-the-physionetcomputing-in-cardiology-challenge-2012-1.0.0\predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0\Outcomes-a.txt" `
  --epochs 20 --max-patients 100 --patience 5

# Data directories (see DATA.md for the full inventory; code must use ml/paths.py, never hardcode)
#   data\predicting-...-2012-1.0.0\predicting-...-2012-1.0.0\set-a              = 1519 patients (fast, legacy subset)
#   data\predicting-...-2012-1.0.0\predicting-...-2012-1.0.0\set-a_full\set-a  = 4000 patients (FULL set-a, use for training)
#   data\predicting-...-2012-1.0.0\predicting-...-2012-1.0.0\set-b_full\set-b  = 4000 patients (set-b, use as holdout)
#   Outcomes-a.txt = labels (4000 rows), Outcomes-b.txt = set-b labels (4000 rows)

# Latest full-training sweep (best: 3-model mixed ensemble s48+c93+s45, val AUC 0.840, fresh holdout 0.844)
python -m ml.sweep_xval       # full set-a vs original val split (round 15, best single 0.833)
python -m ml.pick_ensemble    # greedy holdout-gated ensemble selection (round 18)

# Run backend API
pip install -r backend\requirements.txt
uvicorn backend.app:app --reload --port 8000

# Run frontend
cd frontend
npm install
npm run dev

# Start both (Windows)
.\start-dev.ps1
```

## Directory Structure

```
PROJ/
├── ml/                          # Machine learning pipeline
│   ├── train.py                 # Main training script (early stopping, SpO2, attention)
│   ├── train_lstm.py            # LSTMModel + AttentionLSTMModel definitions
│   ├── dataset.py               # PhysioNet data loader, creates sliding windows
│   ├── preprocess.py            # Z-score normalization, NaN interpolation
│   ├── explain.py               # SHAP-based feature importance (KernelSHAP)
│   ├── eval_shap.py             # (Legacy) SHAP evaluation stub
│   ├── bench_batch.py           # Batch benchmarking
│   ├── run_experiments.py       # Automated multi-config sweeps
│   ├── sweep_*.py               # One-off experiment sweeps:
│   │   ├── sweep_tight.py       #   LR scheduling / warm-start (round 8)
│   │   ├── sweep_finetune.py    #   warm-start fine-tune (round 9)
│   │   ├── sweep_last.py        #   step-decay cadence (round 10)
│   │   ├── sweep_seed.py        #   multi-seed winner (round 11)
│   │   ├── sweep_full.py        #   full set-a stride-30 scan (round 12)
│   │   ├── sweep_ws2.py         #   low-LR warm-start full-data (round 14)
│   │   ├── sweep_feats.py       #   feature-count A/B f12/f16/f20 (round 13)
│   │   ├── sweep_xval.py        #   FULL set-a vs original val split (round 15, BEST single 0.833)
│   │   ├── sweep_seeds2.py      #   multi-seed full-data (round 16, seeds 41-46)
│   │   └── sweep_seeds3.py      #   6 more full-data seeds (round 17, seeds 47-52; pool = 16)
 │   ├── pick_ensemble.py         # Round 18: greedy holdout-gated ensemble selection (old 4-model: 0.837)
│   ├── eval_combos.py           # Candidate-ensemble precision check
│   ├── ensemble_swa.py          # SWA / logit-avg ensemble experiments
│   ├── campaign_auto.py         # Detached results campaign (E1->E2->ensemble->E4, stops at holdout >= 0.85)
│   ├── scaler.json              # Population normalization stats (set-a train split)
│   ├── scaler_combo8k.json      # Scaler for the set-a + 80% set-b member (c93)
│   ├── deployed_manifest.json   # SERVED ensemble: members, checkpoints, features, AUC, threshold
│   ├── ensemble_best.json       # Ensemble selection manifest (members, checkpoints, metrics)
│   ├── requirements.txt         # numpy, pandas, scikit-learn, torch, shap, matplotlib
│   ├── models/                  # Scratch weights GITIGNORED; ensemble/{s48,c93,s45}.pt TRACKED for deploy
│   └── training_runs/           # Timestamped training run outputs (metrics.json only)
│
├── backend/                     # FastAPI REST API
│   ├── app.py                   # Main API: /health, /ingest, /patients, /scores, /metrics, /explain
│   ├── inference.py             # RiskScoreEngine: loads AttentionLSTM, real-time scoring
│   ├── training.py              # TrainingManager: background training jobs
│   ├── db.py                    # SQLite database (vitals storage)
│   ├── replay.py                # PhysioNet data replay (HTTP/MQTT ingest)
│   ├── mqtt_subscriber.py       # MQTT subscriber for vital signs
│   └── requirements.txt         # fastapi, uvicorn, torch, sqlalchemy, etc.
│
├── frontend/                    # React 18 + Vite + Tailwind CSS
│   ├── src/
│   │   ├── App.jsx              # Main shell, routing, dashboard layout
│   │   ├── simulationContext.jsx # Client-side simulation engine (12 patients)
│   │   ├── theme/tokens.css     # Shared light/dark semantic tokens
│   │   ├── motion/              # Motion CSS and page-visibility utilities
│   │   ├── components/
│   │   │   ├── WelcomePage.jsx       # Landing page with hero
│   │   │   ├── SensorWaveform.jsx    # SVG waveform charts
│   │   │   ├── TrainingConfig.jsx    # Training configuration form
│   │   │   ├── TrainingMonitor.jsx   # Real-time training progress
│   │   │   ├── TrainingJobsList.jsx  # List of training jobs
│   │   │   ├── SimulatedDataFeed.jsx # Tabular simulated data view
│   │   │   └── ArchitecturePage.jsx  # System architecture docs
│   │   └── welcome.css
│   ├── tailwind.config.js       # Tailwind content scanning and token aliases
│   ├── postcss.config.js        # Tailwind and Autoprefixer pipeline
│   ├── package.json             # react, react-router-dom, axios, chart.js
│   └── index.html
│
├── discordbot/                  # Discord alert bot
├── chatbot-tele/                # Telegram chatbot
├── firmware/                    # IoT firmware (if applicable)
├── scripts/                     # Utility scripts
├── .env                         # Environment variables (secrets)
├── .env.example                 # Environment template
├── start-dev.ps1                # Start backend + frontend together
├── setup.ps1                    # Initial project setup
└── README.md                    # Project documentation
```

## Architecture

```
Patient Vitals --> [Backend /ingest] --> [SQLite DB]
                     |
                     v
            [AttentionLSTM Model]
                     |
                     v
              Risk Score (0-100)
                     |
          +----------+----------+
          |                     |
     [Frontend Dashboard]   [Discord/Telegram Alerts]
```

## ML Model Architecture

**AttentionLSTMModel** (defined in `ml/train_lstm.py`):

- Input: 12 features x 90 timesteps (window = 90 min, stride 30 at training)
- 2-layer LSTM (hidden_size=96, dropout=0.3), additively attends over time steps
- Batch normalization + dropout
- Sigmoid output for binary mortality prediction (trained with BCEWithLogitsLoss, pos_weight = neg/pos)
- Adam optimizer, weight_decay=1e-4, lr=1e-4 halved every 6 epochs (step decay)
- Trains on population-normalized inputs using `ml/scaler.json` stats (train-split only)

**Current best (deployed, commit ec9c129):**
- 3-model logit-averaged ensemble `s48 + c93 + s45` (all 12 features, w=90, h=96), served from `ml/models/ensemble/*.pt` via multi-checkpoint support in `backend/inference.py`
- **Val AUC 0.840** (original stride-15 80/20 split), **fresh holdout AUC 0.844** (unseen 20% set-b; full set-b holdout N/A since `c93` trained on 80% of set-b)
- Ensemble picked by mixed greedy old+combo selection (`ml/pick_ensemble2.py`, manifest in `ml/ensemble_best.json`)
- Previous milestones: 0.837/0.807 4-model (holdout-gated); 0.833/0.806 single (full set-a training); 0.807/0.765 (1519-patient training)

## Key API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | API status |
| POST | `/ingest` | Ingest vital JSON, returns risk score |
| GET | `/patients` | Top 6 patients by risk score |
| GET | `/patient/{id}` | Patient details + recent vitals |
| GET | `/scores` | All live risk scores |
| GET | `/metrics` | Latest training metrics (AUC, accuracy, recall) |
| GET | `/patient/{id}/explain` | SHAP feature importance + attention weights |
| POST | `/training/start` | Start a new training job |
| GET | `/training/jobs` | List all training jobs |
| GET | `/training/{id}/progress` | Stream training progress (NDJSON) |

## Feature Set

The model uses **12 features** (must match between training and inference — see `ml/scaler.json` and `backend/inference.py`):

| Feature | PhysioNet Name | Normal Range |
|---------|---------------|--------------|
| Heart Rate | HR | 60-100 bpm |
| Respiratory Rate | RespRate | 12-20 /min |
| Temperature | Temp | 36.1-37.2 C |
| Systolic BP | NISysABP | 90-140 mmHg |
| Diastolic BP | NIDiasABP | 60-90 mmHg |
| SpO2 | SpO2 | 95-100% |
| GCS | GCS | 3-15 |
| BUN | BUN | 6-24 mg/dL |
| Creatinine | Creatinine | 0.6-1.2 mg/dL |
| WBC | WBC | 4.5-11 x10^3/uL |
| Platelets | Platelets | 150-450 x10^3/uL |
| Glucose | Glucose | 70-140 mg/dL |

Notes:
- PhysioNet 2012 has **no SpO2 column** — `ml/dataset.py` `PARAMETER_ALIASES` maps SpO2 -> SaO2.
- NaN handling: NaNs are filled with the population mean (from `ml/scaler.json`) before normalization.
- A 20-feature variant (adds K, Na, HCO3, Mg, HCT, pH, PaO2, PaCO2) was tested but scored **worse** (0.787 vs 0.807 full-holdout baseline) — stick with 12 features.

## Code Conventions

- **Python**: Follow existing style, no comments unless complex logic
- **JavaScript/JSX**: React functional components with hooks, Tailwind CSS classes
- **Frontend theme**: Use `frontend/src/theme/tokens.css` for both light and dark semantic tokens; do not introduce page-local color systems.
- **Frontend motion**: Use `frontend/src/motion/` utilities. Never use `transition: all`; animate only approved state properties and provide a reduced-motion path.
- **Frontend honesty**: Keep the simulation banner, research-prototype qualifiers, and not-HIPAA-ready note visible. Do not invent metrics, certifications, clinical claims, or prospective evidence.
- **No new dependencies** without checking existing ones first
- **Model compatibility**: Always update both `train.py` AND `inference.py` when changing features/architecture
- **Thread safety**: RiskScoreEngine uses `threading.Lock()` for concurrent access

## Testing

```powershell
# Smoke test the model (random data)
python ml/train_lstm.py

# Test backend starts
uvicorn backend.app:app --port 8000
curl http://localhost:8000/health

# Test ingestion
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"patient_id":"test","timestamp":1,"HR":85,"SpO2":98,"RespRate":16,"Temp":37,"NISysABP":120,"NIDiasABP":80}'
```

Frontend UI detector:

```powershell
npx impeccable detect frontend/src
```

## Known Issues

- Frontend simulation is client-side only (does not read back from backend)
- Model stats in frontend WelcomePage may still be hardcoded (check before modifying)
- `chart.js` and `socket.io-client` are in package.json but unused
- No unit tests exist yet
- `ml/models/*.pt` is gitignored, so scratch checkpoints need `git add -f`. The deployed
  ensemble `ml/models/ensemble/{s48,c93,s45}.pt` is tracked on purpose (Render boots from it,
  no retraining) — update those files and `ml/deployed_manifest.json` together.
- Full set-a (4000 patients) training is ~5-7 min/epoch at stride 15; use stride 30 (~2-3 min/epoch) for sweeps — deployment uses window 90 regardless of training stride
- When comparing runs: the 0.807/0.833/0.837 numbers all use the ORIGINAL 1519-subset 80/20 stride-15
  val split (seed 42); full-set-a sweeps that use a different split are NOT directly comparable. Deployed val is 0.840 on that same split.
- Honest generalization number is the fresh unseen 20% set-b holdout (currently 0.844); repeated gating on the same
  val split makes val AUC optimistic — keep the holdout gate on every deploy decision
