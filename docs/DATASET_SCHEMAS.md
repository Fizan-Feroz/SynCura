# Dataset schemas

Column-level reference for datasets that live under `data/`. Because `data/` is
gitignored (see `.gitignore`), the schema facts we depend on are recorded here so
they are versioned with the code. Dataset *bytes* never enter this repo.

## PhysioNet 2012 Challenge — the training corpus (real)

Source: PhysioNet Computing in Cardiology Challenge 2012, "Predicting mortality of
ICU patients". Requires a PhysioNet account plus a signed DUA; never redistribute.

- 4,000 set-a stays (train/val) + 4,000 set-b stays (holdout), 48 h each.
- One `.txt` per patient, long-format `Time,Parameter,Value` rows, variable number
  of rows per timestamp (multiple measurements can share a time value).
- Labels in `Outcomes-a.txt` / `Outcomes-b.txt`, column `In-hospital_death` (0/1).
- Loader: `ml/dataset.py`. Windows are causal, forward-fill only, no backward-fill
  and no whole-stay interpolation (`ml/dataset.py:182`).

### The 12 model features

The canonical order lives in `backend/inference.py:39` and is mirrored as
`FEATURES_12` in each sweep script. Training and serving **must** agree on it.

| # | Feature | PhysioNet parameter | Notes |
|---|---------|--------------------|-------|
| 1 | `HR` | Heart Rate | bpm |
| 2 | `RespRate` | Respiratory Rate | /min |
| 3 | `Temp` | Temperature | deg C |
| 4 | `NISysABP` | Arterial BP [Systolic] | mmHg |
| 5 | `NIDiasABP` | Arterial BP [Diagonal] | mmHg |
| 6 | `SpO2` | `SaO2` | PhysioNet 2012 has **no** SpO2 column; `PARAMETER_ALIASES` in `ml/dataset.py:24` maps `SpO2` -> `SaO2` |
| 7 | `GCS` | Glasgow Coma Scale | total, 3-15 |
| 8 | `BUN` | BUN | mg/dL |
| 9 | `Creatinine` | Creatinine | mg/dL |
| 10 | `WBC` | WBC | x10^3/uL |
| 11 | `Platelets` | Platelets | x10^3/uL |
| 12 | `Glucose` | Glucose | mg/dL |

Columns 1-6 are vitals, 7-12 are scalar/lab. A 20-feature variant (adds K, Na,
HCO3, Mg, HCT, pH, PaO2, PaCO2) scored worse (0.787 vs 0.807) and was dropped.

### Label modes

`ml/dataset.py` `create_sequences_from_physionet(..., label_mode=...)`:

- `all` — every window carries the whole-stay outcome. Noisy for early windows.
- `last` — only the final window per patient. Clean labels, small dataset.
- `proximity` (**default**) — keep only windows whose **start** falls within the
  last `horizon_hours` of the stay, and label them with the patient's whole-stay
  `In-hospital_death` outcome. Implementation at `ml/dataset.py:176`:

  ```python
  cutoff = max(seq_len, len(df_raw) - int(horizon_hours * 60))
  starts = range(cutoff, len(df_raw), stride)
  ```

  Two things this does **not** mean: the 12 h is a *window filter*, not a
  prediction horizon, and the label is stay-level rather than "died within 12 h".
  End-of-record is treated as an approximation of outcome time, since death or
  discharge can occur after the 48 h record ends (`ml/dataset.py:126`).

`--horizon-hours` is swept in `ml/sweep_horizon.py` (24 h vs 12 h).

## Challenge 2019 set A — scale-up corpus (real)

- 20,336 ICU stays, hourly, pipe-delimited `.psv`, per-row `SepsisLabel` (~9% positive).
- Covers **11 of the 12** features. The missing one is `GCS` (position 7 in the
  table above); there is no GCS column in the 2019 schema.
- Needs its own loader, `challenge2019_to_features.py`, which is not written yet.

## MIMIC-III demo (100 patients / 136 stays) and MIMIC-IV demo (100 / 140) — ETL testbeds (real)

- Full relational schemas (CHARTEVENTS, LABEVENTS, plus dictionaries).
- All 12 features are present. MIMIC-IV GCS is stored as Eye / Motor / Verbal
  component items (itemids 220739, 223900, 223901) and must be summed to a total.
- Use for developing and debugging the MIMIC -> 12-feature mapper. **Never train on
  these** (n=100).

## eICU demo 2.0.1 — external-validation testbed (real)

- ~2,500 stays across 20 hospitals, 31 `.csv.gz` tables (plus a `sqlite/` copy).
- Use to exercise the external-validation code path. Not used for the reported model.

## hospital-deterioration-dataset — pipeline benchmark (SIMULATED, not real)

Third-party dataset, vendored locally only:
<https://github.com/tarekmasryo/hospital-deterioration-dataset> (CC-BY-4.0,
v1.0.3). Rules-based and probabilistic simulation. **No real patients, no
identifiable information.** `CITATION.cff` in that repo carries the required
citation.

- 10,000 simulated admissions, hourly rows, stays capped at 72 h, fully observed
  (no missing values). Time is hours from admission.

`patients.csv` — one row per patient:

`patient_id`, `age`, `gender`, `comorbidity_index`, `admission_type`,
`baseline_risk_score`, `los_hours`, `deterioration_event`,
`deterioration_within_12h_from_admission`, `deterioration_hour`

`deterioration_event` is 0/1 for any event during the stay;
`deterioration_within_12h_from_admission` is 0/1 for an event in the first 12 h;
`deterioration_hour` is the hour of the **first** event, or `-1` if none.
`baseline_risk_score` is a simulation parameter on a 0-1 scale, **not** a clinical
score.

`vitals_timeseries.csv` — one row per `(patient_id, hour_from_admission)`:

`patient_id`, `hour_from_admission`, `heart_rate`, `respiratory_rate`, `spo2_pct`,
`temperature_c`, `systolic_bp`, `diastolic_bp`, `oxygen_device`, `oxygen_flow`,
`mobility_score`, `nurse_alert`

`oxygen_flow` is exactly `0.0` whenever `oxygen_device == "none"` and positive only
for `nasal`, `mask`, `hfnc`, `niv`.

`labs_timeseries.csv` — one row per `(patient_id, hour_from_admission)`:

`patient_id`, `hour_from_admission`, `wbc_count`, `lactate`, `creatinine`,
`crp_level`, `hemoglobin`, `sepsis_risk_score`

Note the naming difference from PhysioNet: this dataset uses snake_case
(`spo2_pct`, `systolic_bp`, `wbc_count`) and carries no GCS, BUN, or Platelets, so
it does not map onto all 12 model features. It also has no `NISysABP`/
`NIDiasABP` distinction.

Use for end-to-end pipeline smoke tests and benchmarking-harness development.
**Never report metrics from this dataset as clinical evidence.**

## Kaggle snapshots — toy tables (aggregated, not time series)

- `kaggle-icu-mortality`: 15,000 patients x 1 row, mean/std/max/min aggregates plus
  `mortality_label`, APACHE and SOFA. Tabular-baseline playground.
- `kaggle-icu-risk-score`: 1,000 snapshot rows with a precomputed `risk_score`.
  Sanity checks.
- `kaggle-sepsis-mimic-style`: 5,000 subjects x 77 aggregate columns. Feature
  playground.

None of these are time series, so all three are unsuitable for the LSTM.

## Skipped: MIMIC-IV-ED demo 2.2

Emergency-department data, wrong setting, no ICU mortality labels.

## Not present — credentialed

MIMIC-IV full (65k+ ICU stays), eICU-CRD full, and HiRID all require PhysioNet
login, CITI training, and a signed DUA. Each teammate gets individual access.
Redistributing them, including through the team Drive, violates the DUA.
