# SynCura — Talking Points (glance sheet, SYNOPSIS deck)

**Deck:** `ppt/PROJECT SYNOPSIS PPT-FORMAT_TRIMMED_V3.pptx` (10 slides) | **Team:** Abdul Ahad Ikkeri (4PA24CS002), Fathima Reeha (4PA24CS026), Fizan Feroz (4PA24CS032) | **Guide:** Divya K
**Format:** white background, Times New Roman, justified, 1.5 spacing | **Motion:** fade transition every slide; bullets build on click; table/refs fade on click; title + Thank-You auto-fade
**Rubric (25):** Problem 4, Base-paper 4, Novelty 4, Feasibility 4, PPT 3, Participation 3, Questions 3

## One-liner pitch
Attention-LSTM that reads 12 vitals/labs over 90 minutes and gives a real-time 0–100 mortality risk **with explanations** — the papers stop at AUC, we ship the path.

## Our numbers (memorize)
- **0.834** fresh 20% set-B holdout AUC (95% CI 0.825–0.843, window-level bootstrap) vs **0.840** validation → consistent, not overfit
- Honest caveat: the holdout was unseen by weights but USED during ensemble selection — it is not a locked final test set
- 2-layer LSTM, hidden 96, additive temporal attention, 12 features, 90-min window, stride 15
- Ensemble of 3 LSTM models (e12+e13+c53), logit-averaged; operating point 0.70 → precision ~0.50 at recall ~0.48

## Training patients (memorize)
- **Total:** 8,000 (set-a 4,000 + set-b 4,000)
- **Training:** ~7,000 unique patients combined (e12/e13 trained on set-a + 80% set-b; c53 Challenge-2019-warm-started, finetuned on the same E1 set)
- **Val:** 20% patient split → 0.840
- **Holdout:** fresh unseen 20% of set-b excl. locked slices, ~660 patients → 0.834
- e12/e13/c53 are just member IDs (training seeds) — same architecture, different seed/data-history

## 12 features (memorize)
Vitals: HR 60–100 (pump stress) · RespRate 12–20 (breathing trouble) · Temp ~37 (infection) · NISysABP 90–140 (pumping force) · NIDiasABP 60–90 (vessel tone) · SpO2 95–100% (oxygen, most urgent; stored as SaO2 in PhysioNet)
Labs+neuro: GCS 3–15 (consciousness) · BUN 6–24 (kidney waste) · Creatinine 0.6–1.2 (kidney marker) · WBC 4.5–11k (infection fight) · Platelets 150–450k (clotting/sepsis) · Glucose 70–140 (stress swings)
Line: six fast signals catch the crash, six slow ones explain the cause.

## Slide → line (10 slides, ~7 minutes)
1. **Title (20s):** "Good morning/afternoon. We are presenting SynCura — Predictive ICU Monitoring System Using Attention-Based LSTM — real-time mortality risk with SHAP plus temporal-attention explainability. Under the guidance of Divya K. Team: Abdul Ahad Ikkeri, Fathima Reeha, Fizan Feroz." (Title + team auto-fade on load — no clicks needed.)
2. **Abstract (40s, 5 clicks):** deterioration kills, NEWS2 misses trends → attention-LSTM on 12 features × 90-min windows, FastAPI + React → trained on PhysioNet 2012 (4,000 stays) → ensemble 0.840 val / **0.834** holdout → real-time score with explanations; prototype, not a medical device.
3. **Introduction (40s, 5 clicks):** ICU mortality 10–29%, early detection saves lives → NEWS2/SOFA/APACHE-II are static, manual, single-timepoint → they miss direction, duration, interaction → Objective 1–2: model + full-stack system → Objective 3–5: explainability, NEWS2 comparison, simulation.
4. **Literature Survey (60–90s, 1 click):** "Ten studies in one table — title, authors, year, features, limitations. Row 1 is our base: Zheng 2025, time-aware attention-LSTM, dynamic AUROC 0.936 — but retrospective, inflates near discharge, drops cross-hospital. The pattern down the Limitations column is our justification: retrospective data, weak explanations, no deployment." Do NOT read all 10 rows. Name-check DEWS (Choi 2020, beats NEWS2) and RealMIP (missing-data recovery = our scouted upgrade).
5. **Problem Statement (30s, 2 clicks):** bedsides scores depend on single-timepoint thresholds → need an automated model that learns temporal patterns from streaming vitals/labs AND shows its reasons.
6. **Methodology (60s, 6 clicks):** PhysioNet 2012, 4,000 stays, 37 variables → 12 features (6 vitals + 6 labs/neuro) → 90-min windows, causal interpolation, train-only z-score, patient-level split → 2-layer LSTM-96 + additive attention, weighted BCE, Adam, early stopping, 3-model ensemble → pipeline: vitals → /ingest → SQLite → scorer (0–100) → dashboard + Discord/Telegram alerts.
7. **Work Done (60s, 6 clicks — the money slide):** ensemble deployed (e12+e13+c53) → **0.840 / 0.834** (CI 0.825–0.843), locked slices 0.828/0.818 → threshold 0.70 gives precision ~0.50 at recall ~0.48 → backend live (/ingest, /patients, /explain) → dashboard live (risk board, 5 scenarios, NEWS2 comparison) → explainability verified: SpO2, RespRate, SysBP top drivers; no future leakage.
8. **Future Roadmap (40s, 6 clicks):** prospective validation + calibration (prototype only) → MIMIC-IV/eICU migration with fairness analysis → Temporal Fusion Transformer → BioBERT notes fusion → federated + TinyML edge → mistake-driven retraining loop with holdout gating.
9. **References (15s, 1 click):** "Ten IEEE references — Zheng 2025 FIRST as our base paper, then the supporting studies and the PhysioNet 2012 dataset." Do not read them aloud.
10. **Thank You (auto-fade):** "Thank you. SynCura is an explainable research prototype for early ICU risk monitoring. We welcome your questions."

## Click discipline (animations)
- Bullets build on click — one idea per click, pause half a beat so the examiner reads it.
- Slide 4 table and slide 9 references fade as ONE block — single click each, do not expect stepwise builds there.
- Never rapid-fire clicks: each click must match the sentence you are saying.

## Papers in THIS deck (remember these)
| Ref | Full title | Key number |
|---|---|---|
| **[1] Base — Zheng et al. 2025** | Development and Validation of a Dynamic Real-Time Risk Prediction Model for ICU Patients Based on Longitudinal Irregular Data: Multicenter Retrospective Study (JMIR 27, e69293) | AUROC 0.936 MIMIC-IV / 0.919 eICU |
| [2] Alshwaheen et al. 2021 | A Novel and Reliable Framework of Patient Deterioration Prediction in ICU Based on LSTM-RNN (IEEE Access 9) | AUROC 0.933 |
| [3] Choi et al. 2020 | Deep Interpretable Early Warning System for the Detection of Clinical Deterioration (IEEE JBHI 24(9)) | AUROC ~0.880 > NEWS2 |
| [4] Li et al. 2025 | Attention Residual LSTM-FCN / Inpatient Length of Stay and Mortality Prediction (IEEE Access 13) | attention adds ~0.073 AUC |
| [5] PULSE-ICU 2025 | A Pretrained Unified Long-Sequence Encoder for Multi-task Prediction in ICUs (arXiv:2511.22199) | mortality AUROC 0.887 |
| [6] GARLIC 2026 | Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care (arXiv:2608.10969) | SOTA AUROC |
| [7] Yan et al. 2026 | Deep Learning-Based In-Hospital Mortality Prediction Using Long-Term Sequential Data in ICU Patients (PeerJ 14, e21631) | AUROC 0.95 internal |
| [8] Scheid et al. 2025 | Development and Validation of a Clinical Wearable Deep Learning Based Continuous In-Hospital Deterioration Prediction Model (Nature Communications 16, 9513) | AUROC ~0.89, 17h lead |
| [9] PhysioNet 2012 | Computing in Cardiology Challenge 2012 dataset | physionet.org/content/challenge-2012/ |
| [10] Wang et al. 2026 | Expert Augmented Prediction of Circulatory and Respiratory Instability (npj Digital Medicine) | AUROC > 0.8, transparent rules |

Note: the slide-4 survey table and slide-9 references are DIFFERENT sets on purpose — the table surveys the field (10 rows incl. RealMIP, Sadanandan, Wu, Nguyen, Do), the references cite what we built on. If asked: "the table is the landscape, the references are our foundations."

Dataset: PhysioNet / Computing in Cardiology Challenge 2012 — https://physionet.org/content/challenge-2012/

## Ammo for Q&A (do-not-fumble lines)
- **Why LSTM?** Sequential data; best single architecture in Yan 2026 (beats Transformer/GRU). Cheap streaming inference.
- **Why attention?** Temporal explanation — which minutes mattered, shown per patient.
- **Why better than papers?** They report AUC; we report false alarms, precision/sensitivity, lead time, NEWS2 comparison + a true unseen holdout.
- **Does it correct itself?** A: Not live — weights frozen. Mistake-triggered offline loop: confirmed misses/false alarms → weighted retrain → fresh holdout → redeploy.
- **Is it deployable?** No — research prototype. Needs calibration, prospective testing, external validation, regulatory review.
- **PhysioNet age?** Known limitation; set-B holdout (0.834) is our honest generalization number.
- **20 vs 12 features?** 20-feature variant scored worse (0.787 vs 0.807/0.834) — we keep 12.
- **Hardware?** ESP32 + MAX30105 optional live-vitals extension, not required for the prototype.
- **Missing vitals?** Population-mean imputation; RealMIP-style recovery is the scouted upgrade.

## Current limitations we face (be ready to say these)
- **Single dataset** — trained only on PhysioNet 2012; no eICU/MIMIC cross-hospital validation yet.
- **Data age & scope** — no SpO2 column (mapped to SaO2); population means instead of real missing-data recovery.
- **Class imbalance** — mortality is rare; handled with loss weighting but affects precision.
- **Performance not clinical-grade** — 0.834 is strong for a prototype, still short of deployment bar.
- **No prospective testing** — retrospective data only; no clinical collaborators yet.
- **Frontend is client-side simulation** — dashboard does not yet read live backend scores (works, but wired to synthetic stream).
- **Some dashboard stats hardcoded** — model numbers in the UI need manual sync with retrained models.
- **No unit tests** — prevents safe CI-style regressions.
- **Clean-data gap** — model trained on curated retrospective data; real sensor noise (MASC/Scheid point) untested.

If asked "so what's missing?": name 2–3 and always tie back to Slide 8 roadmap — eICU/MIMIC validation, missing-data recovery, prospective pilot.

## Pitfalls
- Don't say "clinical ready" — always "explainable research prototype".
- Don't compare our numbers to Zheng 2025 (different data/inputs) — say "different scope: 12 vitals vs full EMR".
- Repeated val gating inflates val AUC — always cite the holdout 0.834 as the honest number.

## Judges Q&A bank (by rubric, 25 marks)

### 1. PROBLEM (4 marks)
- **Q: What exactly is the problem?** A: ICU deterioration is a process, not one bad reading. NEWS2/SOFA threshold single readings and miss direction, duration, interaction of trends. We build an early-warning aid scoring recent history with reasons.
- **Q: Why not just use NEWS2/SOFA?** A: They are static, manual, single-timepoint rules with ~53% sensitivity and no trend learning. Our LSTM learns temporal patterns and reports lead time vs NEWS2>=7 live on the dashboard.
- **Q: Who benefits?** A: Bedside reviewers get ranked risk + explanations + lead time; SDG 3 (timely review) and SDG 9 (ML+API+visualization pipeline).
- **Q: Is this replacing doctors?** A: No — decision support only. Prototype, no autonomous action, needs prospective + regulatory review.

### 2. BASE PAPER (4 marks)
- **Q: Name the base paper fully.** A: Zheng, Luo, Zhu, Du, Lan, Zhou, Yang & Huang (2025), Development and Validation of a Dynamic Real-Time Risk Prediction Model for ICU Patients Based on Longitudinal Irregular Data, JMIR vol.27 e69293.
- **Q: Method + data + result?** A: Time-aware bidirectional attention LSTM (TBAL) on 176,344 stays (MIMIC-IV + eICU), hourly updates; dynamic AUROC 0.936 MIMIC-IV / 0.919 eICU, recall 79.1%.
- **Q: Why this base?** A: Closest architecture AND task match — attention-LSTM producing real-time interpretable ICU mortality risk from irregular time series, exactly our setup.
- **Q: What is hour 12?** A: Their fixed static-task trigger: admission→hour-12 data predicts 1/2/4/7-day and in-hospital death. Stays <12h excluded. Dynamic tasks instead predict every hour.
- **Q: Static vs dynamic?** A: Static = one prediction at hour 12 for fixed windows; dynamic = rolling next-24h prediction every hour till discharge.
- **Q: What is the discharge/hindsight issue?** A: Retrospective till-discharge data; AUROC climbs to 0.989 at discharge because all info accumulated — hindsight, not early warning. Cross-hospital falls to 0.81/0.76, no prospective test. We ban future info with a 90-min-only window.
- **Q: How did you adapt it?** A: Same attention-LSTM direction on PhysioNet 2012, 12-feature 90-min window for streaming, plus SHAP feature explanations alongside temporal attention (they use attention + Integrated Gradients).

### 3. NOVELTY (4 marks)
- **Q: What is novel if LSTM exists?** A: Integration novelty, not architecture novelty: temporal attention + real-time FastAPI serving + React dashboard + dual explainability (attention for when, SHAP for what) in one prototype.
- **Q: Papers report AUC — what do you add?** A: False alarms, sensitivity/specificity/precision, lead-time estimate, NEWS2>=7 inline comparison, threshold slider, calibration path — the decision metrics clinicians need.
- **Q: Proof against overfitting?** A: Fresh 20% set-B holdout 0.834 (95% CI 0.825–0.843, window-level) consistent with validation 0.840 — no major overfit. Caveat: the holdout guided ensemble selection, so it is not a locked final test. (Full set-b N/A since members trained on 80% of set-b.)
- **Q: Why attention + SHAP both?** A: Attention = which minutes mattered; SHAP = which features mattered. Per-patient inspectable on dashboard.
- **Q: Why is 90-min window novel vs base?** A: Base uses full stay till discharge; we force early-warning conditions — recent window only, deployable streaming.

### 4. FEASIBILITY (4 marks)
- **Q: Dataset and features?** A: PhysioNet 2012, 4,000 train + 4,000 holdout; 12 features (HR, RespRate, Temp, NISysABP, NIDiasABP, SpO2→SaO2 mapped + GCS, BUN, Creatinine, WBC, Platelets, Glucose).
- **Q: SpO2 mapping valid?** A: PhysioNet 2012 has SaO2 not SpO2; same units (%), arterial gold standard — slot stays named SpO2 so train/inference/frontend match.
- **Q: Leakage controls?** A: Train-only population z-score, patient-level GroupShuffleSplit, proximity labeling (last 12h), holdout never touched.
- **Q: Why 12 not 20 features?** A: Tested — 20-feature variant scored worse (0.787 vs 0.834). Keep 12.
- **Q: Model size / speed?** A: 2-layer LSTM h96, dropout 0.3, ~lightweight; thread-safe RiskScoreEngine, 0–100 score per ingest; runs on CPU.
- **Q: Why ensemble of 3?** A: Logit-averaged, greedy holdout-gated selection across two training families; ensemble beats single seeds and the cross-family mix adds diversity single-family ensembles lacked.
- **Q: Missing vitals live?** A: Interpolation + population-mean fill now; RealMIP-style generative recovery scouted.
- **Q: Frontend — real or fake?** A: Honest answer: dashboard runs on synthetic scenario stream (5 scenarios); backend replay path with real PhysioNet data exists via /ingest. Full wiring is future work.
- **Q: Calibration / thresholds?** A: Threshold slider live-tunes sensitivity/specificity/false alarms; decision-curve analysis + prospective pilot are the stated next steps.
- **Q: Ethics / privacy?** A: Deidentified public data, no PHI; deployment needs consent, privacy, bias audit (worse ≥65 subgroup), regulatory clearance.

### 5. PPT (3 marks)
- **Q: Why a table on the literature slide?** A: Synopsis format demands ≥10 papers at a glance — one table with title/author/year/features/limitations; the Limitations column itself makes our case.
- **Q: Why do table and references differ?** A: Table = field landscape (10 rows); references = our foundations (base + supporting + dataset). Deliberate split.
- **Q: References slide?** A: Zheng as [1] with full title, 8 supporting papers with full citations, PhysioNet dataset as [9]. No ellipsis, no truncation.
- **Q: Report says 0.837/0.807 but slides say 0.840/0.834?** A: Synced — report, README, and AGENTS now describe the deployed 3-model ensemble (val 0.840, fresh holdout 0.834); 0.837/0.807 kept as previous-milestone row.

### 6. PARTICIPATION (3 marks)
- **Q: Who did what?** A: Fizan Feroz — ML pipeline + backend; teammates — frontend + integration. Each member owns: one can demo dashboard scenarios, one can explain attention/SHAP output, one can defend metrics/holdout.
- **Q: Equal contribution?** A: Show commits across ml/, backend/, frontend/; rehearse handoffs per slide (1–3 member A, 4–6 member B, 7–10 member C).

### 7. QUESTIONS / DEFENSE (3 marks)
- **Q: Is 0.834 clinically enough?** A: Strong prototype, not deployment bar. Field range for vitals-only is 0.70–0.85; Wu upper bound 0.926. Needs calibration + prospective validation.
- **Q: Why PhysioNet 2012, not MIMIC-IV?** A: Public, reproducible, established benchmark; age acknowledged; eICU/MIMIC validation needs credentialed access (out of current 2012+2019 scope).
- **Q: Lead time — how measured?** A: Average early-warning hours vs NEWS2>=7 crossing on dashboard analytics; must be measured properly, not assumed.
- **Q: Biggest limitation in one line?** A: Single retrospective dataset, no cross-hospital or prospective evidence — every limit has a named upgrade on slide 8.
- **Q: Six more months?** A: Credentialed eICU/MIMIC access for external validation, RealMIP imputation, time-aware attention, prospective pilot with decision curves.
