# ML Training Interface Documentation

## Overview

The ICU Predictive Monitoring System now includes a comprehensive ML training interface that allows team members to:

- Configure and start new LSTM model training jobs with custom hyperparameters
- Monitor training progress in real-time with live metrics
- View training history across all jobs
- Track model performance with multiple evaluation metrics (AUC, Accuracy, Precision, Recall)

## Architecture

### Backend (FastAPI)

Training endpoints are exposed via the REST API (`backend/app.py`):

#### `POST /training/start`
Starts a new training job with the provided configuration.

**Request Body:**
```json
{
  "physionet_path": "/path/to/physionet/data",
  "outcomes_path": "/path/to/Outcomes-a.txt",
  "epochs": 5,
  "batch_size": 32,
  "learning_rate": 0.001,
  "max_patients": 100,
  "vital_features": ["HR", "RespRate", "Temp", "NISysABP", "NIDiasABP", "SpO2",
                     "GCS", "BUN", "Creatinine", "WBC", "Platelets", "Glucose"],
  "window": 90,
  "hidden_size": 96
}
```

Defaults match the serving contract (12 features in order, 90-min window,
hidden 96). Deviating configs train into isolated job artifacts and are
flagged non-promotable. Only one job runs at a time (second start → 409);
job IDs are UUID-suffixed; artifacts live at `ml/models/<job_id>.*` and
never overwrite serving files.

**Response:**
```json
{
  "job_id": "job-uuid-here",
  "status": "pending",
  "config": { ... }
}
```

#### `GET /training/jobs`
Returns all training jobs with their current status.

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job-uuid",
      "status": "running",
      "current_epoch": 2,
      "total_epochs": 5,
      "metrics": { "train_loss": 0.45, "val_accuracy": 0.78, ... },
      "config": { ... },
      "created_at": "2024-01-15T10:30:00Z",
      "error_message": null
    }
  ]
}
```

#### `GET /training/{job_id}`
Returns detailed information about a specific training job.

**Response:**
```json
{
  "job_id": "job-uuid",
  "status": "running",
  "current_epoch": 2,
  "total_epochs": 5,
  "metrics": {
    "train_loss": 0.4523,
    "val_accuracy": 0.782,
    "auc": 0.845,
    "accuracy": 0.78,
    "precision": 0.81,
    "recall": 0.75
  },
  "config": { ... },
  "error_message": null
}
```

#### `GET /training/{job_id}/progress`
Streams real-time training progress updates. Returns newline-delimited JSON (NDJSON format).

**Streaming Response (each line is valid JSON):**
```json
{"status": "running", "current_epoch": 1, "total_epochs": 5, "metrics": {...}}
{"status": "running", "current_epoch": 2, "total_epochs": 5, "metrics": {...}}
```

### Backend Training Service (`backend/training.py`)

The training service manages the complete job lifecycle:

**TrainingJob Class:**
- Stores job metadata (job_id, status, epochs, config)
- Maintains real-time metrics (train_loss, val_accuracy, auc, accuracy, precision, recall)
- Provides thread-safe progress tracking via queue
- Supports serialization to JSON for API responses

**TrainingManager Class:**
- Manages job queue and lifecycle
- Spawns background training worker threads
- Loads PhysioNet 2012 data with sliding windows
- Trains LSTM model with progress callbacks
- Evaluates model on validation set
- Saves trained model to disk
- Supports concurrent job execution (limited by system resources)

**Key Methods:**
```python
job = training_manager.create_job(config_dict)  # Create job
training_manager.start_training(job.job_id)      # Start training (background thread)
job = training_manager.get_job(job_id)           # Get job by ID
jobs = training_manager.get_all_jobs()           # List all jobs
progress = training_manager.get_job_progress(job_id)  # Get progress update
```

### Frontend (React)

Three new components handle the training UI:

#### TrainingJobsList (`frontend/src/components/TrainingJobsList.jsx`)
- Lists all training jobs with status badges
- Shows progress bars for running jobs
- Displays real-time metrics preview
- Polls backend every 3 seconds for updates
- "New Training Job" button to create job
- Click job to view detailed monitoring

**Features:**
- Status filtering (pending, running, completed, failed)
- Sorting by creation time
- Quick metrics overview
- Error display for failed jobs

#### TrainingConfig (`frontend/src/components/TrainingConfig.jsx`)
- Form to configure new training job
- Inputs for data paths, hyperparameters, feature selection
- File picker UI (future enhancement)
- Form validation
- Submits to `/training/start` endpoint
- Redirects to monitoring page on success

**Form Fields:**
- PhysioNet data path (required)
- Outcomes file path (required)
- Epochs (1-100, default 5)
- Batch size (1-256, default 32)
- Learning rate (0.00001-0.1, default 0.001)
- Max patients (10-10000, default 100)
- Vital features (multi-select checkboxes)

#### TrainingMonitor (`frontend/src/components/TrainingMonitor.jsx`)
- Real-time training progress dashboard
- Progress bar (epoch progress)
- Live metric cards (train loss, val accuracy, AUC, accuracy, precision, recall)
- Status badge with color coding
- Job configuration display
- Error messages for failed jobs
- Polls backend every 2 seconds while training
- Auto-updates when job completes

**Polling Strategy:**
- GET `/training/{job_id}` every 2 seconds during training
- Displays metrics as they update in real-time
- Stops polling when status becomes "completed" or "failed"

### Frontend Routing

Updated `App.jsx` includes React Router with training routes:

```javascript
/                    → Dashboard (mock patient data)
/training            → TrainingJobsList component
/training/new        → TrainingConfig component
/training/{jobId}    → TrainingMonitor component
```

## Data Flow

### Training Job Lifecycle

```
1. User fills TrainingConfig form
   ↓
2. POST /training/start with config
   ↓
3. Backend creates TrainingJob (status: "pending")
   ↓
4. Backend spawns background thread running _train_worker()
   ↓
5. Worker loads PhysioNet data and creates sliding windows
   ↓
6. Worker trains LSTM model for N epochs:
   - Each epoch: train on batch, validate, compute metrics
   - Push progress update to job.progress_queue
   - Update job.current_epoch, job.metrics
   ↓
7. Worker evaluates final model, saves to disk
   ↓
8. Job status: "completed" (or "failed" if error)
   ↓
9. Frontend polls and displays final metrics
```

### Real-Time Update Flow

```
Frontend polls GET /training/{job_id}
    ↓
Backend returns current job state with latest metrics
    ↓
Frontend updates progress bar, metric cards
    ↓
Repeat every 2 seconds (or until job completes)
```

## Configuration

### Environment Variables

Add to `.env` (see `.env.example`):

```
BACKEND_PORT=8000
VITE_API_URL=http://localhost:8000
MODEL_PATH=./backend/models/
PHYSIONET_DATA_PATH=/path/to/physionet/data
```

### Frontend Dependencies

`frontend/package.json` already includes everything needed:
- `react-router-dom` (routing)
- `axios` (HTTP requests)
- `gsap` + `@gsap/react` (scroll-driven motion)
- `@fontsource-variable/archivo` (self-hosted variable font)

Run `npm install` in `frontend/`. Styling is plain CSS against the tokens in
`src/theme/tokens.css` — there is no Tailwind and no Tailwind dependency. See
`THEME.md` before changing any component's appearance.

Note: the backend is started from the repo root as `uvicorn backend.app:app`, not
from inside `backend/`, so the `ml/` imports resolve.

## Usage

### Starting the System

1. **Backend** (run from the repository root):
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --reload --port 8000
```

2. **Frontend:**
```bash
cd frontend
npm install
npm run dev  # Vite dev server on port 5173
```

### Creating a Training Job

1. Navigate to http://localhost:5173/training
2. Click "New Training Job"
3. Fill in the form:
   - PhysioNet path: `/path/to/physionet/2012` (or your data location)
   - Outcomes path: `/path/to/Outcomes-a.txt`
   - Epochs: 5
   - Other hyperparameters: use defaults or adjust
   - Features: select vital signs to include
4. Click "Start Training"
5. Automatically redirected to monitoring page

### Monitoring Training

- Progress bar shows epoch progress
- Metrics update in real-time (train loss, val accuracy, etc.)
- Status badge shows current state
- View configuration and error messages if job fails
- Click "Back to Training" to return to job list

### Viewing All Jobs

- Navigate to http://localhost:5173/training
- See all jobs in chronological order
- Status color-coded (pending=yellow, running=blue, completed=green, failed=red)
- Click any job to view detailed monitoring
- "New Training Job" button to create another job

## Performance Considerations

### Backend
- **Threading:** Each training job runs in a background thread (single-threaded training)
- **Memory:** LSTM model + data can require significant memory
- **Disk:** Trained models saved to `./models/` directory

### Frontend
- **Polling:** Every 2 seconds during training (adjust if needed)
- **Network:** Minimal payload (JSON with metrics)
- **Re-renders:** Only when metrics/status change

## Future Enhancements

1. **WebSocket Support:** Replace polling with real-time WebSocket updates
2. **File Picker UI:** Browser file selection for data paths
3. **Model Management:** Download/delete/export trained models
4. **Comparison View:** Compare metrics across multiple training runs
5. **Early Stopping:** Configurable early stopping based on validation metrics
6. **Distributed Training:** Support for multiple GPU/parallel training
7. **Checkpoint Loading:** Resume interrupted training from checkpoint
8. **Hyperparameter Tuning:** GridSearch/RandomSearch UI

## Troubleshooting

### Training Job Never Starts
- Check backend logs for errors
- Verify data paths are correct and accessible
- Ensure FastAPI server is running on port 8000

### Metrics Not Updating
- Check frontend console for axios errors
- Verify VITE_API_URL environment variable matches backend port
- Ensure CORS is enabled on backend (should be by default)

### Model Training Fails
- Check backend training.py logs for data loading errors
- Verify PhysioNet data format matches expected structure
- Check disk space for model saving
- Review error_message field in failed job details

### Frontend Routes Not Working
- Verify react-router-dom is installed (`npm install react-router-dom`)
- Check that frontend is being served on port 5173
- Clear browser cache and hard refresh (Ctrl+F5)

## Code Examples

### Manual Job Status Check (Python)

```python
import requests
import json

# Start a job
response = requests.post('http://localhost:8000/training/start', json={
    'physionet_path': '/data/physionet',
    'outcomes_path': '/data/Outcomes-a.txt',
    'epochs': 5,
    'batch_size': 32
})
job_id = response.json()['job_id']

# Monitor job
import time
while True:
    resp = requests.get(f'http://localhost:8000/training/{job_id}')
    job = resp.json()
    print(f"Status: {job['status']}, Epoch: {job['current_epoch']}/{job['total_epochs']}")
    print(f"Loss: {job['metrics'].get('train_loss', 'N/A')}")
    
    if job['status'] in ['completed', 'failed']:
        print("Job finished!")
        break
    time.sleep(2)
```

### Manual Frontend Request (JavaScript)

```javascript
const apiUrl = 'http://localhost:8000';

// Start training
const startResp = await fetch(`${apiUrl}/training/start`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    physionet_path: '/data/physionet',
    outcomes_path: '/data/Outcomes-a.txt',
    epochs: 5
  })
});
const {job_id} = await startResp.json();

// Check status
const jobResp = await fetch(`${apiUrl}/training/${job_id}`);
const job = await jobResp.json();
console.log(`Training: ${job.current_epoch}/${job.total_epochs} epochs`);
```

## Files Modified/Created

**Modified:**
- `backend/app.py` - Added training endpoints
- `frontend/src/App.jsx` - Added routing setup
- `frontend/package.json` - Added react-router-dom dependency

**Created:**
- `backend/training.py` - Training job management service
- `frontend/src/components/TrainingConfig.jsx` - Configuration form
- `frontend/src/components/TrainingMonitor.jsx` - Progress monitoring
- `frontend/src/components/TrainingJobsList.jsx` - Job list view

## References

- [FastAPI Streaming Responses](https://fastapi.tiangolo.com/advanced/custom-response/)
- [React Router Documentation](https://reactrouter.com/)
- [PyTorch LSTM Training](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [PhysioNet 2012 Dataset](https://physionet.org/content/challenge-2012/)
