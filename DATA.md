# SynCura — Dataset Guide

All datasets live under `PROJ/data/` (≈2.3 GB). That folder is **gitignored**:
bytes travel via the team Google Drive (`scripts/download_data.py`), never via git.
Paths resolve through `ml/paths.py` (`SYNCURA_DATA_ROOT` override, default `PROJ/data`).

**Column-level schemas live in [`docs/DATASET_SCHEMAS.md`](docs/DATASET_SCHEMAS.md)**
so they stay versioned with the code. This file is the inventory; that file is the schema.

## Directory map

```
data/
├── predicting-mortality-...-challenge-2012-1.0.0/   # TRAINING data (extracted)
├── predicting-mortality-...-challenge-2012-1.0.0.zip
├── challenge-2019-1.0.0/training/training_setA/     # 20,336 PSV files
├── mimic-iv-demo-2.2/                               # extracted MIMIC-IV demo
├── eicu-crd-demo-2.0.1/                             # 31 .csv.gz tables
├── hospital-deterioration-dataset/                  # simulated benchmark + docs
├── kaggle-icu-mortality/                            # 15k-row tabular CSV
├── kaggle-icu-risk-score/                           # 1k-row snapshot CSV
├── kaggle-sepsis-mimic-style/                       # 5k-subject wide CSV
├── mimic-iii-clinical-database-demo-1.4.zip
├── mimic-iv-clinical-database-demo-2.2.zip
└── mimic-iv-ed-demo-2.2.zip                         # ED only — not used
```

## Dataset cards

### 1. PhysioNet 2012 Challenge — THE training data (real)
- 8,000 ICU stays (set-a 4,000 train/val + set-b 4,000 holdout), 48 h each.
- Format: one `.txt` per patient, long `Time,Parameter,Value` rows + `Outcomes-*.txt` (`In-hospital_death`).
- Covers all 12 SynCura features (SpO2 read as SaO2).
- Loader: `ml/dataset.py` (90-min windows, stride 15, proximity labels). Proximity means only windows *starting* within the last `--horizon-hours` (default 12) of the stay are kept, each labeled with the patient's whole-stay `In-hospital_death` outcome. The 12 h is a window filter, not a prediction horizon. See `docs/DATASET_SCHEMAS.md`.

### 2. Challenge 2019 (sepsis) set A — scale-up data (real)
- 20,336 ICU stays, hourly rows, pipe-delimited `.psv`, per-hour `SepsisLabel` (~9% positive).
- Maps to 11/12 features (HR, O2Sat, Temp, SBP, DBP, Resp, BUN, Creatinine, Glucose, WBC, Platelets; **no GCS** — the 2019 schema has no GCS column).
- Use: second training corpus / deterioration-label experiments. Needs its own loader (`challenge2019_to_features.py`, not yet written).

### 3. MIMIC-III demo (100 pts, 136 stays) + MIMIC-IV demo (100 pts, 140 stays) — ETL testbeds (real)
- Full relational schemas (CHARTEVENTS/LABEVENTS + dictionaries); all 12 features verified present
  (MIMIC-IV GCS = Eye/Motor/Verbal items 220739/223900/223901, summed to total).
- Use: develop and debug the MIMIC→12-features mapper. **Never train on these (n=100).**

### 4. eICU demo 2.0.1 — external-validation testbed (real)
- ~2,500 stays across 20 hospitals, 31 tables. Use: test the external-validation code path.

### 5. hospital-deterioration-dataset — pipeline benchmark (SIMULATED, not real)
- 10,000 simulated admissions, 72 h hourly vitals+labs, fully observed, next-12 h deterioration label.
- Use: end-to-end pipeline smoke tests and benchmarking harness development.
- **Never report metrics from this as clinical evidence.**

### 6. Kaggle snapshots — toy tables (real-ish, aggregated)
- `kaggle-icu-mortality`: 15,000 patients × 1 row (mean/std/max/min aggregates + `mortality_label`, APACHE/SOFA). Tabular-baseline playground only.
- `kaggle-icu-risk-score`: 1,000 snapshot rows with a precomputed `risk_score`. Sanity checks only.
- `kaggle-sepsis-mimic-style`: 5,000 subjects × 77 aggregate columns. Feature playground only.
- None are time series — **unsuitable for the LSTM**.

### 7. MIMIC-IV-ED demo — skipped (wrong setting: emergency dept, no ICU mortality labels)

## Not here (credentialed — each teammate gets their own access, never re-upload)
MIMIC-IV full (65k+ ICU stays) · eICU-CRD full · HiRID — PhysioNet login + CITI + signed DUA.
**Redistributing these (including via team Drive) violates the DUA.**

## Adding a new dataset
1. Drop it under `data/<name>/` (gitignored automatically).
2. Add a card above (source · n · format · label · 12-feature coverage · intended use).
3. If it's large + open, upload to team Drive and register it in `scripts/download_data.py`.
4. If it's credentialed, document the access steps here instead of the files.
