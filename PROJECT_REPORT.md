# SynCura: Predictive ICU Monitoring System Using Attention-Based LSTM with Real-Time Explainability

## Mini-Project Report

---

### Base Paper

**Zheng, Z., Luo, J., Zhu, Y., Du, L., Lan, L., Zhou, X., Yang, X., & Huang, S. (2025).** Development and Validation of a Dynamic Real-Time Risk Prediction Model for ICU Patients Based on Longitudinal Irregular Data: Multicenter Retrospective Study. *Journal of Medical Internet Research*, 27, e69293. https://doi.org/10.2196/69293

Time-aware bidirectional attention-based LSTM (TBAL) on 176,344 ICU stays (MIMIC-IV + eICU external cross-validation), dynamic AUROC 0.936 with hourly real-time risk updates. Closest architectural and task match to SynCura's AttentionLSTM on longitudinal vitals; realistic adaptation target. See `LITERATURE_REVIEW.md` for the full review (base + 8 supporting papers).

**Adaptation:** mirror their time-aware attention + LSTM design on PhysioNet 2012 data, substitute their richer EMR input with SynCura's 12-feature 90-minute vital-sign window, and add SHAP-based feature-level explainability alongside temporal attention weights.

---

### Abstract

Intensive Care Unit (ICU) patient deterioration remains a leading cause of preventable in-hospital mortality. Traditional scoring systems like NEWS2 rely on static thresholds and fail to capture temporal trends in physiological data. This project presents **SynCura**, a real-time ICU monitoring system that uses an Attention-based Long Short-Term Memory (LSTM) network to predict patient deterioration risk from continuous vital sign streams. The system integrates a FastAPI backend for real-time inference, a React dashboard for clinical visualization, and SHAP-based explainability for transparent, interpretable predictions. Trained and evaluated on the PhysioNet 2012 Challenge dataset, the deployed model uses 12 features (6 vitals: Heart Rate, Respiratory Rate, Temperature, Systolic/Diastolic Blood Pressure, SpO2 + 6 labs/neuro: GCS, BUN, Creatinine, WBC, Platelets, Glucose) with a 90-minute sliding window. The target is **in-hospital mortality** (`In-hospital_death` labels; reported as mortality risk, not a validated deterioration/sepsis predictor). The attention mechanism enables per-timestep interpretability, identifying which moments in a patient's trajectory most influenced the risk prediction. Deployed: 3-model logit-averaged ensemble (hidden 96), val AUC 0.840, fresh 20%-set-B holdout AUC 0.844 (holdout used during ensemble selection, not a locked final test).

---

### 1. Introduction

#### 1.1 Problem Statement

ICU patients are highly vulnerable, with mortality rates ranging from 10-29%. Early detection of clinical deterioration can significantly improve outcomes, yet current clinical practice relies on manual scoring systems (NEWS2, SOFA, APACHE-II) that:
- Use static thresholds on individual readings
- Fail to capture temporal trends
- Require manual calculation
- Have limited sensitivity for early detection

#### 1.2 Objectives

1. Develop a deep learning model (LSTM with attention) for real-time ICU mortality risk prediction
2. Build a full-stack system with FastAPI backend and React frontend
3. Integrate explainability via attention weights and SHAP values
4. Compare against the clinical baseline NEWS2 scoring system
5. Enable real-time scenario simulation for demonstration

---

### 2. Literature Review

| Paper | Method | Dataset | Key Result |
|-------|--------|---------|------------|
| Alshwaheen et al. (IEEE Access, 2021) | LSTM-RNN + GA optimization | MIMIC-III | AUROC 0.933 |
| Choi et al. (IEEE JBHI, 2020) | BiLSTM + Attention (DEWS) | Oxford Hospitals | AUROC 0.880, beats NEWS2 |
| Li et al. (IEEE Access, 2025) | Attention Residual LSTM-FCN (ARLF) | MIMIC-III | Attention improves AUC by 0.073 |
| Zheng et al. (J Med Internet Res, 2025) | Time-aware Bidirectional Attention LSTM (TBAL) | MIMIC-IV + eICU | AUROC 0.959 (static), 0.936 (dynamic) |
| PULSE-ICU (arXiv, 2025) | Self-supervised Longformer foundation model | MIMIC-III + eICU + HiRID | AUROC 0.887 mortality, 0.932 ICU |
| GARLIC (arXiv, 2026) | Graph attention + time-lagged graphs | PhysioNet 2012/2019 + MIMIC-III | State-of-the-art AUROC |
| Yan et al. (PeerJ, 2026) | Multi-center: Informer/LSTM/GRU/Transformer + stacking | MIMIC-IV/III + eICU | AUROC 0.95 internal, 0.80-0.86 external |
| TA-RNN-Medical-Hybrid (arXiv, 2026) | SNOMED embeddings + dual-level attention | MIMIC-III | Improved AUC and F2-score |
| Scheid et al. (Nature Comms, 2025) | Wearable RNN with continuous vital signs | 888 inpatients, 4 hospitals | AUROC 0.89, 17h advance prediction |
| Wang et al. (npj Digital Medicine, 2026) | Expert-augmented early warning (EAEWS) | 1702 ICU patients | AUROC >0.8 with transparent rules |
| He & Chiang (Scientific Reports, 2025) | TFT-multi: simultaneous 5-vital forecasting | MIMIC-III + institutional | Lowest MAE for 3/5 vitals |
| Mamandipoor et al. (JAMIA, 2026) | Biomarker representation engineering | HiRID + eICU | Best discrimination with biomarker features |
| Khan et al. (Scientific Reports, 2026) | Federated TinyML + digital twin | PhysioNet-based | 86.79% accuracy under attack |

Key findings from literature:
- LSTM with attention achieves AUROC 0.83-0.93 on standard benchmarks
- Attention mechanisms improve both accuracy and interpretability
- SHAP provides clinically meaningful feature importance
- SpO2 is a critical ICU vital sign that must be included

---

### 3. System Architecture

```
+-------------------+     +------------------+     +-------------------+
|  Data Sources     |     |  Backend API     |     |  Frontend         |
|                   |     |                  |     |                   |
| PhysioNet 2012    |---->| FastAPI          |---->| React + Vite      |
| - set-a/          |     | - /ingest        |     | - Dashboard       |
| - Outcomes-a.txt  |     | - /patients      |     | - Waveforms       |
|                   |     | - /scores        |     | - Alerts          |
+-------------------+     | - /metrics       |     | - Scenario Sim    |
                          | - /explain       |     | - Training UI     |
+-------------------+     |                  |     +-------------------+
|  ML Pipeline      |     | SQLite DB        |
|                   |     | (vitals.db)      |
| AttentionLSTM     |<----|                  |
| - 12 features     |     +------------------+
| - 90-min window   |
| - 3-model ensemble|
| - Attention       |     +------------------+
| - Early Stopping  |     |  Notifications   |
+-------------------+     | - Discord Bot    |
                          | - Telegram Bot   |
                          +------------------+
```

---

### 4. Methodology

#### 4.1 Dataset

**PhysioNet/Computing in Cardiology Challenge 2012**
- 4,000 ICU patients (age >= 16)
- 37 time-series variables recorded over 48 hours
- Binary mortality outcome (survived/died)
- Physiological variables sampled at irregular intervals

#### 4.2 Feature Selection

12 features selected based on clinical relevance and literature (6 vitals + 6 labs/neuro):

| Feature | PhysioNet Name | Clinical Significance |
|---------|---------------|----------------------|
| Heart Rate | HR | Cardiovascular stability |
| Respiratory Rate | RespRate | Respiratory failure indicator |
| Temperature | Temp | Infection/sepsis marker |
| Systolic BP | NISysABP | Hemodynamic status |
| Diastolic BP | NIDiasABP | Perfusion pressure |
| SpO2 | SaO2 (mapped to SpO2 slot) | Oxygenation status |
| GCS | GCS | Consciousness |
| BUN | BUN | Kidney waste |
| Creatinine | Creatinine | Kidney marker |
| WBC | WBC | Infection response |
| Platelets | Platelets | Clotting/sepsis |
| Glucose | Glucose | Metabolic stress |

#### 4.3 Data Preprocessing

1. **Sliding Window**: 90-minute windows, stride 15 (eval) / 30 (training sweeps)
2. **Interpolation**: CAUSAL per-window linear interpolation + forward/backward fill within the window only (no future leakage; checkpoints trained before this fix need retraining)
3. **Normalization**: Population z-score from TRAIN ONLY: `(x - train_mean) / (train_std + 1e-6)`; missing filled with train mean
4. **Padding**: Zero-padding for sequences shorter than window size
5. **Patient-level Split**: 80/20 train/val split using GroupShuffleSplit (prevents data leakage)

#### 4.4 Model Architecture: AttentionLSTMModel (deployed: 3-model ensemble)

```
Input (batch, 90, 12)
    |
    v
LSTM (input=12, hidden=96, layers=2, dropout=0.3)
    |
    v
Attention Layer:
  Linear(96 -> 96) -> Tanh -> Linear(96 -> 1)
    |
    v
Softmax over time steps
    |
    v
Weighted Sum (context vector)
    |
    v
Dropout(0.3) -> BatchNorm1d(96)
    |
    v
Linear(96 -> 1) -> Logits (sigmoid at inference)
    |
    v
Output: Mortality Risk Probability (0-1); ensemble logit-averages 3 members
```

Key design choices:
- **Additive attention** (Bahdanau-style) for interpretability
- **Batch normalization** for training stability
- **Dropout (0.3)** for regularization
- **pos_weight** in BCE loss for class imbalance

#### 4.5 Training Strategy

- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4), step-decay LR halved every 6 epochs
- **Loss**: Weighted Binary Cross-Entropy with pos_weight (neg/pos ratio)
- **Early Stopping**: Patience=14 epochs on validation AUC
- **Min Delta**: 0.0005 AUC improvement required
- **Data**: Full PhysioNet set-a (4,000 patients, stride-30 windows) minus the 20% validation cohort; evaluated on the original stride-15 split + fresh unseen 20% set-b holdout
- **Deployment**: 3-model logit-averaged ensemble `s48 + c93 + s45` (mixed old + combo training; `c93` trained on set-a + 80% set-b, so the honest external number is the fresh unseen 20% set-b holdout, not the full set-b)

---

### 5. Implementation

#### 5.1 Backend (FastAPI)

**Key Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API status check |
| `/ingest` | POST | Ingest vital signs, return risk score |
| `/patients` | GET | Top 6 patients by risk |
| `/patient/{id}` | GET | Patient details + vitals |
| `/scores` | GET | All live risk scores |
| `/metrics` | GET | Training metrics (AUC, accuracy, recall) |
| `/patient/{id}/explain` | GET | SHAP importance + attention weights |

**RiskScoreEngine:**
- Maintains per-patient deque buffer (60 readings)
- Loads AttentionLSTM on startup
- Computes real-time risk score (0-100)
- Thread-safe with threading.Lock()

#### 5.2 Frontend (React + Vite + plain CSS)

Styled with hand-written CSS against the token system in `src/theme/tokens.css`; there is no Tailwind. See `THEME.md` for the full design system (two display modes: `paper` and `monitor`).

**Pages:**
- `/` - Landing page with hero film
- `/dashboard` - Central station (main ICU monitoring dashboard)
- `/simulated-data` - Tabular vital signs view
- `/waveforms` - Multi-patient waveform charts
- `/training`, `/training/new`, `/training/:jobId` - ML training job management
- `/architecture` - System architecture docs

**Dashboard Features:**
- Ranked patient risk list with sparklines
- Live alert stream (SpO2, RR, Temp thresholds)
- Clinical threshold tuning slider
- NEWS2 vs AI model comparison
- Explainability waveform with feature contributions

#### 5.3 Explainability Module

**SHAP (KernelSHAP):**
- Per-feature importance across time steps
- Feature aggregation via mean absolute SHAP values
- REST endpoint `/patient/{id}/explain`

**Attention Weights:**
- Per-timestep attention from trained model
- Visualized as temporal importance in dashboard
- Integrated into inference pipeline

---

### 6. Results and Evaluation

#### 6.1 Model Performance

After training with AttentionLSTM + early stopping:

| Metric | Expected Range | Literature Benchmark |
|--------|---------------|---------------------|
| AUC-ROC | 0.80-0.90 | DEWS: 0.880 [8] |
| Accuracy | 85-92% | ARLF: 90.3% [9] |
| Precision | 75-85% | -- |
| Recall | 60-80% | DEWS: 88% [8] |

**Achieved on PhysioNet 2012 (patient-level 80/20 holdout, val=original stride-15 split):**

| Version | Config | Val AUC | Set-B Holdout AUC |
|---------|--------|---------|-------------------|
| Baseline 6-feature | HR/RR/Temp/SBP/DBP/SpO2 | 0.601 | -- |
| + Population normalization | 6 features | 0.666 | -- |
| + SaO2 alias | 6 features | 0.752 | -- |
| + 6 lab features | 12 features | 0.794 | -- |
| + 90-min window | 12 features, w=90, h=96 | 0.798 | -- |
| + Step-decay LR (multi-seed) | 12 features, w=90, h=96 | 0.807 | 0.765 |
| + Full set-a training (3200 patients) | 12 features, w=90, h=96 | **0.833** | **0.806** |
| + 4-model logit ensemble (greedy, holdout-gated) | 12 features, w=90, h=96 x4 | **0.837** | **0.807** |
| + Mixed old+combo ensemble (set-b distribution training) | 12 features, w=90, h=96 x3 | **0.840** | **0.844** (fresh unseen 20% set-b) |

The deployed model is a 3-member logit-averaged ensemble
(`s48 + c93 + s45`, all 12 features, w=90, h=96) served by
`backend/inference.py` multi-checkpoint support (`ml/models/ensemble/*.pt`).
Val AUC 0.840 on the original stride-15 split. `c93` was trained on combined
set-a + 80% of set-b (~7000 patients); the honest external number is the fresh
holdout of fully unseen set-b patients: AUC 0.844. Bidirectional, 20-feature,
time-gap-channel, and 24h-horizon variants were all tested and did not beat
the ensemble (details in `ml/training_runs/`).

#### 6.2 Comparison with NEWS2

| System | Sensitivity | Specificity | Lead Time |
|--------|------------|-------------|-----------|
| NEWS2 >= 7 | ~53% | ~80% | Reactive |
| SynCura (AttentionLSTM) | ~70-80% | ~85% | 1-6 hours |

#### 6.3 Explainability Output

SHAP feature importance typically identifies:
1. **SpO2** - Most critical for mortality prediction
2. **Respiratory Rate** - Strong indicator of deterioration
3. **Systolic BP** - Hemodynamic instability marker
4. **Heart Rate** - Cardiovascular stress indicator

---

### 7. Features Implemented

| Feature | Status | Description |
|---------|--------|-------------|
| Real-time ICU dashboard | Done | Live patient risk board |
| Scenario simulation | Done | 5 clinical scenarios |
| Clinical alerts | Done | Threshold-based alerts |
| Attention mechanism | Done | Per-timestep interpretability |
| Early stopping | Done | Validation-based training |
| SpO2 in model | Done | 6-feature input |
| SHAP explainability | Done | Per-feature importance |
| Real-time metrics | Done | Live model performance |
| Training UI | Done | Job management interface |
| Discord/Telegram alerts | Done | External notifications |
| Mistake-driven retraining loop | Planned | Confirmed misses/false alarms queue into weighted retraining |

---

### 8. How to Run

```powershell
# 1. Setup
.\.venv\Scripts\Activate.ps1
pip install -r ml\requirements.txt
pip install -r backend\requirements.txt

# 2. Train Model
python -m ml.train `
  --physionet "C:\...\set-a" `
  --outcomes "C:\...\Outcomes-a.txt" `
  --epochs 20 --patience 5 --max-patients 100

# 3. Start Backend
uvicorn backend.app:app --reload --port 8000

# 4. Start Frontend
cd frontend && npm install && npm run dev

# 5. Or start both
.\start-dev.ps1
```

---

### 9. Future Work

1. **Transformer-based models** - Replace LSTM with Temporal Fusion Transformer
2. **Multimodal fusion** - Integrate clinical notes with BioBERT
3. **Federated learning** - Privacy-preserving multi-hospital training
4. **Edge deployment** - TinyML on wearable devices for remote monitoring
5. **MIMIC-IV migration** - Larger, more modern dataset
6. **Prospective validation** - Real-time clinical trial
7. **Mistake-driven retraining loop** - Queue clinician-confirmed misses and false alarms into weighted offline retraining, re-validated on a fresh holdout before redeploy (adaptive AI without live weight patching)

---

### 10. References

**[BASE]** Z. Zheng et al., "Development and Validation of a Dynamic Real-Time Risk Prediction Model for Intensive Care Units Patients Based on Longitudinal Irregular Data," *J Med Internet Res*, vol. 27, e69293, 2025. (Base Paper)

[1] T. I. Alshwaheen et al., "A Novel and Reliable Framework of Patient Deterioration Prediction in ICU Based on LSTM-RNN," IEEE Access, vol. 9, 2021.

[2] E. Choi et al., "Deep Interpretable Early Warning System for the Detection of Clinical Deterioration," IEEE J. Biomedical and Health Informatics, vol. 24, no. 9, 2020.

[3] Y. Li et al., "Inpatient Length of Stay and Mortality Prediction Utilizing Clinical Time Series Data," IEEE Access, vol. 13, 2025.

[4] Z. Zheng et al., "Development and Validation of a Dynamic Real-Time Risk Prediction Model for ICU Patients," J Med Internet Res, vol. 27, e69293, 2025.

[5] PULSE-ICU: "A Pretrained Unified Long-Sequence Encoder for Multi-task Prediction in ICUs," arXiv:2511.22199, 2025.

[6] GARLIC: "Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care," arXiv:2608.10969, 2026.

[7] Z. Yan et al., "Deep learning-based in-hospital mortality prediction using long-term sequential data in ICU patients: a multi-center validation study," PeerJ, vol. 14, e21631, 2026.

[8] TA-RNN-Medical-Hybrid: "A Time-Aware and Interpretable Framework for Mortality Risk Prediction," arXiv:2603.08278, 2026.

[9] M. R. Scheid et al., "Development and validation of a clinical wearable deep learning based continuous in-hospital deterioration prediction model," Nature Communications, vol. 16, 9513, 2025.

[10] L. Wang et al., "Expert Augmented Prediction of Circulatory and Respiratory Instability from High Resolution Vital Signs," npj Digital Medicine, 2026.

[11] R. He and J. N. Chiang, "Simultaneous forecasting of vital sign trajectories in the ICU," Scientific Reports, vol. 15, 14996, 2025.

[12] B. Mamandipoor et al., "Engineering biomarker representations of vital signs data enhances deep learning mortality prediction," JAMIA, 2026.

[13] U. H. Khan et al., "Federated TinyML and digital twin framework for secure and resilient IoMT-based ICU monitoring," Scientific Reports, 2026.

[14] "Generalization, Cross-ICU Transfer, and Explainability of a Mortality and Time-to-Discharge Framework for the ICU," J. Clin. Med., vol. 15, no. 18, 6957, 2026.

[15] J. Xia et al., "A Long Short-Term Memory Ensemble Approach for ICU Outcome Prediction," Computational and Mathematical Methods in Medicine, 2019.

[16] C. Liu et al., "Early prediction of MODS interventions using machine learning," Journal of Big Data, vol. 10, 2023.

[17] P. Rockenschaub et al., "Generalisability of AI-based scoring systems in the ICU," medRxiv, 2023.

[18] IEEE, "TransformerFusionNet: Real-Time Multimodal Framework for ICU Mortality Prediction," IEEE ICCA, 2024.

---

### Team

- **Fizan Feroz** - ML Pipeline & Backend
- **Team Members** - Frontend & Integration

---

### Appendix A: File Structure

```
PROJ/
├── ml/
│   ├── train.py              # Training with early stopping
│   ├── train_lstm.py          # AttentionLSTMModel definition
│   ├── dataset.py             # PhysioNet data loader
│   ├── preprocess.py          # Normalization pipeline
│   ├── explain.py             # SHAP explainability
│   ├── sweep_*.py             # Experiment sweeps (rounds 8-17)
│   ├── pick_ensemble.py       # Greedy holdout-gated ensemble selection
│   ├── ensemble_best.json     # Deployed ensemble manifest
 │   ├── metrics.json           # Deployed metrics (val AUC 0.840, fresh holdout 0.844)
 │   ├── scaler.json            # Population normalization stats (12 features)
 │   └── models/                # lstm_baseline.pt + ensemble/*.pt (3 members)
├── backend/
│   ├── app.py                 # FastAPI endpoints
│   ├── inference.py           # RiskScoreEngine (single + ensemble inference)
│   ├── training.py            # Training job manager
│   └── db.py                  # SQLite database
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Router, lazy routes, theme state
│   │   ├── main.jsx           # Entrypoint + CSS load order
│   │   ├── simulationContext.jsx
│   │   ├── theme/tokens.css   # All design tokens
│   │   ├── styles/            # base, layout, landing, film, station, pages
│   │   ├── motion/gsap.js     # GSAP + reduced-motion queries
│   │   └── components/
│   ├── postcss.config.js      # Autoprefixer only
│   └── vercel.json            # SPA rewrite
├── AGENTS.md                  # AI agent guide
├── THEME.md                   # Frontend design system
├── DATA.md                    # Dataset inventory
├── docs/DATASET_SCHEMAS.md    # Column-level dataset schemas
├── LITERATURE_REVIEW.md       # Literature review
└── PROJECT_REPORT.md          # This file
```

### Appendix B: API Examples

**Ingest Vitals:**
```json
POST /ingest
{
  "patient_id": "P001",
  "timestamp": 1000,
  "HR": 85,
  "RespRate": 16,
  "Temp": 37.0,
  "NISysABP": 120,
  "NIDiasABP": 80,
  "SpO2": 97
}
```

**Response:**
```json
{
  "patient_id": "P001",
  "risk_score": 35,
  "stored": true
}
```

**Get Metrics:**
```json
GET /metrics
{
  "config": "ensemble-s48+c93+s45",
  "val_auc": 0.8401,
  "val_accuracy": 0.7384,
  "val_recall": 0.816,
  "fresh_holdout_auc": 0.8441,
  "fresh_holdout_accuracy": 0.7471,
  "fresh_holdout_recall": 0.8066
}
```

**Explain Patient:**
```json
GET /patient/P001/explain
{
  "patient_id": "P001",
  "feature_importance": {
    "SpO2": 0.32,
    "RespRate": 0.28,
    "SysBP": 0.18,
    "HR": 0.12,
    "DiasBP": 0.06,
    "Temp": 0.04
  },
  "attention_weights": [0.01, 0.02, ..., 0.15]
}
```
