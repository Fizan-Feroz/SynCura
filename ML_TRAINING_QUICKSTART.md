# ML Training Quick Start Guide

## Installation

### 1. Update Dependencies

The frontend now requires `react-router-dom`. Install it:

```bash
cd frontend
npm install
```

### 2. Backend Already Updated

The `backend/training.py` module is already implemented with full training job management. The `app.py` file has been updated with training endpoints.

## Quick Start (5 Minutes)

### Start Backend

Run from the repository root so the `ml/` imports resolve:

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

Verify with: `curl http://localhost:8000/health`

### Start Frontend

```bash
cd frontend
npm run dev
```

Open: http://localhost:5173

### Create Your First Training Job

1. Click **Training** in the navigation bar
2. Click **New Training Job**
3. Fill in the form:
   - **PhysioNet Data Path**: Path to your PhysioNet 2012 data directory
   - **Outcomes File**: Path to `Outcomes-a.txt`
   - Leave other fields at defaults for quick testing
4. Click **Start Training**

You'll be redirected to the monitoring page showing:
- Real-time progress bar
- Live metric updates
- Current epoch counter

## Features

### Training Configuration Page

**URL:** `http://localhost:5173/training/new`

Configure training jobs:
- Data paths (PhysioNet dataset, outcomes file)
- Hyperparameters (epochs, batch size, learning rate, max patients)
- Vital features (multi-select which vital signs to include)

### Training Monitor

**URL:** `http://localhost:5173/training/{job_id}`

Monitor a single training job:
- Progress bar (epoch progress)
- Real-time metrics display:
  - Train Loss
  - Validation Accuracy
  - AUC (Area Under ROC Curve)
  - Accuracy
  - Precision
  - Recall
- Current status (pending/running/completed/failed)
- Error messages if job fails
- Job configuration details

### Training Jobs List

**URL:** `http://localhost:5173/training`

View all training jobs:
- Quick overview of all jobs
- Status badges (color-coded)
- Progress bar for each job
- Metrics preview
- Click any job for detailed monitoring
- "New Training Job" button to create another

## API Endpoints

### Check System Health

```bash
curl http://localhost:8000/health
```

Response: `{"status":"ok"}`

### Start Training Job

```bash
curl -X POST http://localhost:8000/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "physionet_path": "/path/to/physionet",
    "outcomes_path": "/path/to/Outcomes-a.txt",
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "max_patients": 100,
    "vital_features": ["HR", "RespRate", "Temp", "NISysABP", "NIDiasABP"]
  }'
```

Response includes `job_id` for monitoring.

### Get All Jobs

```bash
curl http://localhost:8000/training/jobs
```

### Get Specific Job

```bash
curl http://localhost:8000/training/{job_id}
```

### Stream Progress

```bash
curl http://localhost:8000/training/{job_id}/progress
```

Returns newline-delimited JSON updates as training progresses.

## Troubleshooting

### "Failed to start training" Error

**Check:**
1. Backend is running on port 8000
2. Data paths are correct and accessible
3. Outcomes file exists
4. Check browser console for detailed error

### Metrics Not Updating

**Check:**
1. Frontend is polling correctly (check network tab in browser dev tools)
2. Training job status is "running" (not pending/completed/failed)
3. VITE_API_URL is correctly set (should be `http://localhost:8000`)

### Module Import Errors

**Run (from the repository root):**
```bash
pip install -r backend/requirements.txt
```

Ensure all dependencies are installed:
- FastAPI
- PyTorch
- NumPy
- Scikit-learn
- SQLAlchemy

## Performance Notes

- **Training time:** Depends on dataset size and number of epochs
- **Memory usage:** ~2-4GB for full PhysioNet 2012 dataset
- **Network polling:** Frontend polls every 2 seconds (adjustable)
- **Concurrent jobs:** Limited by system resources and GPU availability

## Next Steps

1. **Prepare data:** Download PhysioNet 2012 dataset
2. **Adjust hyperparameters:** Try different learning rates, epochs, batch sizes
3. **Monitor training:** Watch metrics in real-time
4. **Compare runs:** Use job list to compare multiple training attempts
5. **Export model:** (Future feature) Download trained models

## Architecture Overview

```
Frontend                    Backend
---------                   -------
React App                   FastAPI Server
  ├─ Dashboard                ├─ /health
  ├─ TrainingJobsList          ├─ /ingest
  ├─ TrainingConfig            ├─ /patients
  └─ TrainingMonitor           ├─ /training/start (POST)
       (polls every 2s)        ├─ /training/jobs (GET)
                               ├─ /training/{id} (GET)
                               └─ /training/{id}/progress (STREAM)
                                      ↓
                               TrainingManager (threading)
                                      ↓
                               PhysioNet Data → LSTM Model
                                      ↓
                               Save Model + Metrics
```

## Useful Commands

### Monitor Backend Logs

```bash
# In backend terminal
# Already running with --reload flag
```

### Check Frontend Compilation

```bash
# In frontend terminal, watch for errors
npm run dev
```

### Manually Check Job Status (Python)

```python
import requests
job_id = "your-job-id"
resp = requests.get(f"http://localhost:8000/training/{job_id}")
job = resp.json()
print(f"Status: {job['status']}")
print(f"Epoch: {job['current_epoch']}/{job['total_epochs']}")
print(f"Loss: {job['metrics'].get('train_loss', 'N/A')}")
```

## What's Running Where

| Component | Location | Port | URL |
|-----------|----------|------|-----|
| Frontend React App | `frontend/` | 5173 | http://localhost:5173 |
| Backend FastAPI | `backend/` | 8000 | http://localhost:8000 |
| Training Service | `backend/training.py` | (background threads) | N/A |
| Database | `backend/icu.db` | (SQLite) | (local file) |

## Common Workflows

### Workflow 1: Quick Test Training

1. Start backend: `python -m uvicorn backend.app:app --reload --port 8000` (from the repo root)
2. Start frontend: `npm run dev`
3. Go to http://localhost:5173/training/new
4. Use small dataset (max_patients: 10, epochs: 2)
5. Monitor progress at http://localhost:5173/training/{job_id}
6. Should complete in ~5 minutes

### Workflow 2: Production Training

1. Start backend with production config
2. Set max_patients: 1000-10000
3. Set epochs: 10-20 (depends on convergence)
4. Use optimized hyperparameters
5. Monitor metrics for overfitting
6. Wait for completion (hours possible)

### Workflow 3: Hyperparameter Tuning

1. Start multiple training jobs with different configs
2. View all jobs at http://localhost:5173/training
3. Compare final metrics from each job
4. Use best configuration for production

## Support

For issues or questions:
1. Check error messages in browser console
2. Review backend logs for data loading errors
3. Verify data paths are accessible
4. Ensure all dependencies are installed
