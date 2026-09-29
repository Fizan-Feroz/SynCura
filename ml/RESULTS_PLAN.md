# Results Improvement Plan — SynCura (streaming-compatible only)

## Where we stand

- Deployed: 3-model ensemble v2 (`e12+e13+c53`: two E1 causal seeds + one Challenge-2019-warm-started seed finetuned on the E1 set), val AUC **0.840**, fresh 20%-set-B holdout **0.834** (excl. locked slices; manifest in `ml/deployed_manifest.json`, figures regenerated via `ml/plot_results.py`).
- Retired: v1 (`s48+c93+s45`, val 0.840 / holdout 0.844) was trained with **whole-stay interpolation**, so its numbers are not comparable to anything causal. Preserved at `ml/deployed_manifest.v1.json` for history.
- Rounds 1–3 verdict (20+ seeds, 8 config families on PhysioNet 2012 + Challenge 2019 only, per scope): every single-model variant converges to ~0.82 locked (locked-A 0.828, locked-B 0.818). C19 pretraining and multi-task joint training showed zero-to-negative transfer as singles; only cross-family ensemble diversity moved the needle (+0.005 to honest 0.840). GRU-D (E2c, seeds 26/27) in flight; ATE loss deliberately skipped (needs the paper's causal-discovery machinery).
- Consequence: the 0.840/0.844 v1 numbers are **not comparable** to anything trained after the fix.

## The #1 bet: gap (time-since-observation) channels

ICU sampling is irregular **on purpose** — crashing patients get measured more often. Whole-stay
interpolation erases that signal; causal interpolation keeps values honest but still discards
*when* things were measured. Gap channels restore it: 12 extra features = minutes since each
vital was actually observed (0 = measured now, capped at window length).

- Already implemented: `ml/dataset.py:create_sequences_from_physionet(..., gap_channels=True)`
  outputs `(N, 90, 24)`. Verified shapes 2026-09-24.
- Literature support: RealMIP (Xie 2025) shows missingness handling transfers across hospitals;
  TBAL (Zheng 2025) is explicitly time-aware. Gap channels are the cheap, streaming-safe version.
- Expected gain: +0.005–0.02 AUC based on clinical time-series literature. Costs nothing at
  serving time except 12 more input dims.
- Constraint respected: fully causal, forward-only, no future info.

## Experiment matrix (run in order; stop when gates fail)

| ID | Config | Seeds | Question answered |
|----|--------|-------|-------------------|
| E0 | smoke (`--smoke`) | — | Does the runner work end-to-end? (~3 min) |
| E1 | baseline f12-h96-w90, lr1e-4, wd1e-4, do0.3 | 3 (11,12,13) | Honest causal baseline |
| E2a | gap f24-h96-w90, same hypers | 3 (21,22,23) | Do gap channels help? |
| E2b | gap f24-h128-w90 (only if E2a ≥ E1) | 2 (24,25) | Does width stack with gaps? |
| E2c | GRU-D cell (mask + delta + decay; Che et al. 2018, review S9) — only if E2a stalls | 2 (26,27) | Does in-cell decay beat input-level gaps? |
| E3 | winner config +3 seeds → greedy ensemble | — | Best deployable ensemble |
| E4 | ablations (only if E3 stalls) | 1 each | horizon 6h / 24h; window 120; dropout 0.2/0.4; lr 3e-4; **multiplicative value×feature fusion (MedFuse, S10)**; **BCE + α·ATE loss (CRISP, S11)** |

Fixed for all runs: proximity labels, 12h horizon (except E4), stride 30 train / 15 val,
pos_weight = neg/pos, Adam, step-decay LR halving every 6 epochs, early stopping on val AUC
(patience 14, min_delta 5e-4, max 40 epochs). Val = original stride-15 split (seed 42).
Fresh holdout = 20% set-b patients (seed 123) — reported, and used as a promotion gate
(it guided selection before, so it is NOT a locked final test; say so in the deck).

## Promotion gates (all must hold)

1. Candidate val AUC > current deployed val (0.8404).
2. Fresh holdout AUC ≥ 0.83 (no collapse off-distribution).
3. Config is streaming-servable: forward-only, causal, input contract documented.
4. `ml/plot_results.py` regenerated: ROC + metrics table + patient-level AUC + Brier/ECE.
5. Deck + TALKING_POINTS numbers updated in the same commit as the manifest.

## Serving work required for gap models (E2+)

Gap models need `input_size=24` and live gap computation. Design (not yet implemented):

1. `ml/deployed_manifest.json`: add `"input_size": 24`, gap feature names
   (`["GAP_HR", ...]`), per-member scalers (24-dim).
2. `backend/inference.py`: track per-patient last-observation timestamps in
   `RiskScoreEngine.add_vital`; build the 24-dim vector (12 values + 12 gaps, capped at
   window length); normalize with the 24-dim member scaler.
3. `backend/db.py`: no change (gaps are derived, not stored).
4. Test: extend `backend/tests/test_api.py` with a 24-dim manifest fixture.
5. `inference.py` already refuses a manifest whose `arch.input_size != len(FEATURES)`
   with a clear warning instead of a silent misshape — do not bypass this.

Baseline (E1) models serve with **zero** backend changes.

## Commands

```powershell
# 0. Validate the loop (minutes)
.\.venv\Scripts\python.exe -m ml.improve_results --smoke

# 1. Causal baseline (~3 x 40-80 min)
.\.venv\Scripts\python.exe -m ml.improve_results --config baseline --seeds 11 12 13

# 2. Gap channels (~3 x 40-80 min, slightly slower: 24-dim input)
.\.venv\Scripts\python.exe -m ml.improve_results --config gap --seeds 21 22 23

# 3. Greedy ensemble over a finished run dir (val-gated, holdout-reported)
.\.venv\Scripts\python.exe -m ml.improve_results --ensemble ml/training_runs/<run_dir>

# 4. Regenerate deck figures from the NEW deployed checkpoints
.\.venv\Scripts\python.exe -m ml.plot_results
```

## Time estimates (RTX 4060 Laptop, CUDA)

- E0 smoke: ~3 min
- One full seed (7k patients, stride 30, ≤40 epochs, early stopping): 40–80 min
- E1 (3 seeds): 2–4 h
- E2a (3 seeds): 2–4 h (+~15% for 24-dim)
- E3 selection: minutes (reuses cached predictions)
- Full campaign: one overnight run

## External validation roadmap (concrete, from 2026 literature)

Vague "eICU/MIMIC validation" is replaced with: **BlendedICU first** — harmonized
eICU-CRD + MIMIC-IV + AmsterdamUMCdb + HiRID (~20k patients each), the exact recipe the
2026 four-database pooling study (review S13) used to prove pooling beats single-source
training (+8.0% composite) and that internal numbers overstate external AUC by up to 13.8%.
Sequence after E3: (1) BlendedICU pooled evaluation of the frozen ensemble (no retraining —
measures the true generalization gap); (2) pooled fine-tuning if the gap exceeds ~0.05;
(3) MIMIC-IV-Ext-CLIF for contemporary data. Target band per Mamandipoor et al. (S12):
**0.84–0.87 external AUC** for a structured-data system. Full dataset table in
`LITERATURE_REVIEW.md` §5 (includes HiRID 2-min resolution, DACMI imputation benchmark).

## What NOT to try (streaming constraint)

- Bidirectional LSTMs (need the full sequence; break live scoring).
- Whole-stay imputation tricks (reintroduces the leakage just removed).
- Test-time peeking (normalizing with holdout stats, tuning on the holdout repeatedly
  without reporting it — the current deck already discloses holdout-guided selection).
- Chasing val AUC alone without the holdout gate (repeated gating already inflates val).

## Reporting checklist (per experiment)

- [ ] val AUC / acc / sens / spec / prec / F1
- [ ] fresh-holdout AUC + 95% CI (window bootstrap) + patient-level AUC (max-prob per patient)
- [ ] Brier score + ECE (calibration)
- [ ] seed + full config + scaler path recorded in run `metrics.json`
- [ ] deck/table updated only after gates pass
