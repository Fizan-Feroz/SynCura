# SynCura — Dataset Guide

All datasets live under `PROJ/data/` (≈2.3 GB). That folder is **gitignored**:
bytes travel via the team Google Drive (`scripts/download_data.py`), never via git.
Paths resolve through `ml/paths.py` (`SYNCURA_DATA_ROOT` override, default `PROJ/data`).

> **SCOPE (2026-09-29): modeling uses PhysioNet 2012 (§1) and Challenge 2019 (§2) ONLY.**
> Everything else in this file is DISREGARDED — documented below for provenance
> (files may still sit on disk) but never to be trained, evaluated, or reported on.
> Mirrored in `ml/dataset_registry.py`, which enforces it in code.

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
- Use: second training corpus / deterioration-label experiments. Loader: `ml/challenge2019_to_features.py`, reusing the canonical `SERVING_FEATURES` order and `ml/dataset.py` windowing. Its patient-level label collapses `SepsisLabel`, so it means sepsis onset rather than in-hospital death; do not pool it with PhysioNet labels. Policy: cleared for `train`, not for `deploy_train` in `ml/dataset_registry.py`.

## Disregarded datasets (out of scope — files may remain on disk, never use)

- MIMIC-III demo + MIMIC-IV demo (100 pts each): ETL testbeds only, never train.
- eICU demo 2.0.1 (~2,500 stays, 20 hospitals): external-validation code-path test only.
- hospital-deterioration-dataset: SIMULATED, incomplete 12-feature coverage — never clinical evidence.
- Kaggle snapshots (`kaggle-icu-mortality`, `kaggle-icu-risk-score`, `kaggle-sepsis-mimic-style`): aggregated toy tables, no time axis, unusable by the LSTM.
- MIMIC-IV-ED demo: wrong care setting, no ICU mortality label.
- Credentialed (never pursued under current scope): MIMIC-IV full · eICU-CRD full · HiRID.

## Adding a new dataset
1. Drop it under `data/<name>/` (gitignored automatically).
2. Add a card above (source · n · format · label · 12-feature coverage · intended use).
3. If it's large + open, upload to team Drive and register it in `scripts/download_data.py`.
4. If it's credentialed, document the access steps here instead of the files.
