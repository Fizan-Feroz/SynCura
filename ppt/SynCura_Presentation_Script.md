# SynCura Presentation Script

## Suggested Duration

Approximately 8 to 10 minutes for 16 slides (~30 seconds per slide, more on slides 5, 9, 10). Keep the problem and proposed solution clear, and spend extra time on the base paper, results, and paper-vs-reality gap.

## Slide 1: Title and Team Details

Good morning/afternoon everyone.

We are presenting our project, **SynCura: Predictive ICU Monitoring System Using Attention-Based LSTM with Real-Time Explainability**.

SynCura is a software-based predictive ICU monitoring prototype. It uses an attention-based LSTM to estimate deterioration risk from recent clinical measurements, provides explanations for the prediction, and displays the result through a dashboard.

Our team members are Abdul Ahad Ikkeri, Fathima Reeha, and Fizan Feroz. Their USNs are shown on the slide.

## Slide 2: Problem Statement

ICU patients can deteriorate rapidly, and early identification is important for timely clinical review.

Current scores such as NEWS2 and SOFA use thresholds and structured assessments. These tools are useful and clinically familiar, but they do not directly learn the complete temporal pattern — direction, duration, and interaction — of changing patient observations.

Our problem is to build an early-warning aid that analyzes recent patient history and produces an understandable risk estimate. SynCura is a research prototype. It is not intended to replace clinicians or make autonomous medical decisions.

## Slide 3: Motivation / Need for the Project

We selected this problem because deterioration is often a process rather than a single event. Continuous ICU data contains trends and interactions that are difficult to summarize manually.

Earlier warning gives clinicians more time to review the patient and decide on intervention.

Explainability is equally important. A risk score alone is not sufficient; users must be able to inspect which recent time steps and which features influenced the prediction.

Therefore, SynCura combines temporal modeling, real-time inference, a dashboard, and explanations in one workflow.

## Slide 4: Existing System / Related Work

This slide is split deliberately, so examiners see the structure at a glance.

On top, the **base paper — Zheng et al. (2025)**: time-aware bidirectional attention LSTM for real-time ICU mortality on MIMIC-IV plus eICU, dynamic AUROC 0.936. This is the architecture and task we adapt.

Below, the **supporting papers**: Wang, Bai and Jin (2026) on explainable LSTM, Yan et al. (2026) with plain LSTM AUC 0.802, Sadanandan (2026) on multimodal fusion, Wu et al. (2024) on vitals-only prediction, Xie et al. (2025) on RealMIP missing-data recovery, Choi et al. (2020) on interpretable early warning, Li et al. (2025) on attention residual LSTM, and Scheid et al. (2025) on wearables. The 2017 and 2023 papers were removed from this slide to keep the review current and focused.

The gap we identify: most studies stop at AUC, while SynCura connects temporal prediction, real-time serving, dashboard visualization, and explanation.

## Slide 5: Base Paper / Reference Paper

Our base paper is **Development and Validation of a Dynamic Real-Time Risk Prediction Model for ICU Patients Based on Longitudinal Irregular Data: Multicenter Retrospective Study**, by Zheng, Luo, Zhu, Du, Lan, Zhou, Yang, and Huang, published 2025 in the *Journal of Medical Internet Research*, volume 27, article e69293.

It uses 176,344 ICU stays from MIMIC-IV with eICU cross-validation, modeling irregular longitudinal EMR — vitals, labs, medications — with hourly risk updates. Their TBAL model reaches dynamic AUROC 0.936 on MIMIC-IV and 0.919 on eICU, recall 79.1%.

Two details examiners may ask: **hour 12** is their fixed trigger for static tasks — data from admission to hour 12 predicts death in 1/2/4/7 days and overall in-hospital mortality. And the caveat we state openly: the study is retrospective till discharge, so AUROC rises to 0.989 at discharge because all information has accumulated — hindsight, not early warning — while cross-hospital transfer falls to 0.81 and 0.76 with no prospective test.

SynCura adapts the attention-LSTM direction to PhysioNet 2012 with a focused 12-feature, 90-minute window using only past data, and adds SHAP feature-level explanations alongside temporal attention.

## Slide 6: Proposed Solution

SynCura is an attention-based LSTM over a rolling 90-minute window of 12 clinical features: heart rate, respiratory rate, temperature, systolic and diastolic pressure, oxygen saturation, GCS, BUN, creatinine, WBC, platelets, and glucose.

It returns a 0-to-100 risk score with two explanations: temporal attention for influential time steps, SHAP for influential features.

Three differences from the literature: we report false alarms, sensitivity, specificity, precision, lead time, and a NEWS2 comparison instead of AUC only; our fresh unseen 20% set-B holdout AUC of 0.834 sits just under the 0.840 validation, inside the confidence interval; and we ship an end-to-end FastAPI path with alerts and HTTP/MQTT ingestion, not just a trained model.

Fourth, SynCura is designed to learn from its mistakes. Every confirmed miss and false alarm is queued, clinician-reviewed, and weighted into the next offline retrain — so each version fails less where the last one did. We never patch weights on a single live case: the mistake triggers the lesson, batches teach it, and a fresh holdout re-validates it.

## Slide 7: Methodology / Proposed Approach

First, PhysioNet 2012 with 12 selected features. Measurements are interpolated into regular sequences: 90-minute windows, 15-minute stride, population z-score normalization computed from training data only to avoid leakage.

The model is a two-layer LSTM, hidden size 96, dropout, batch normalization, additive temporal attention, sigmoid risk output.

FastAPI serves inference, React shows risk and trends, SHAP plus attention explain each prediction at feature and time-step level.

## Slide 8: Expected Outcome

A working real-time dashboard showing deterioration risk from recent measurements: 0-to-100 score, per-patient explanations, trend views.

Evaluation targets a realistic 0.78–0.93 AUC range for a single-dataset vitals-plus-labs system, with earlier warning versus NEWS2 measured properly, not assumed.

The goal is a feasible, explainable research prototype — not clinical deployment readiness.

## Slide 9: Experimental Result

Our deployed ensemble achieves 0.840 validation AUC and **0.834 on the fresh unseen 20% set-B holdout**. Holdout just under validation but inside the 95% CI (0.825–0.843) suggests no major overfitting — we state it cautiously. (Full set-b holdout is N/A because members trained on 80% of set-b.)

Point at the figure: the ROC curve reproduced from the deployed checkpoints gives val AUC 0.840, and the confusion matrix at threshold 0.5 shows recall 0.762 at precision 0.354. That recall-precision trade is the number a clinician cares about.

Beyond AUC we report sensitivity, specificity, precision, false-alarm count, and lead-time estimate, benchmarked against NEWS2.

Remaining before any clinical claim: calibration, threshold analysis, and prospective validation.

## Slide 10: Limitations & Improvement Path

This is our paper-versus-real-world slide, and how SynCura becomes a better paper.

First, paper versus reality: base reports 0.936 overall and 0.989 at discharge — hindsight from full-stay data — falling to 0.81/0.76 cross-hospital with no bedside test.

Second, reality is messier: US-only retrospective data, missing and asynchronous vitals, age bias above 65, single-center MIMIC data.

Third, SynCura is stricter: 90-minute window only, no future or discharge info, holdout 0.834 just under validation 0.840 but inside the CI.

Fourth, we report what papers skip: false alarms, precision, lead time versus NEWS2, calibration.

Finally, the scouted path: external eICU/MIMIC validation, RealMIP-style imputation, time-aware attention, prospective pilot with decision-curve analysis.

## Slide 11: Technology / Tools Required

Python for ML and backend, JavaScript for frontend. PyTorch, FastAPI, SQLite, scikit-learn, SHAP; React, Vite, Tailwind CSS. Dataset is PhysioNet 2012 Challenge. ESP32 plus MAX30105 is an optional future live-vital extension.

## Slide 12: SDG Relevance

SDG 3, Good Health and Well-Being: earlier deterioration detection supports timely review and monitoring. SDG 9, Industry, Innovation and Infrastructure: explainable deep learning in a modern ML-plus-API-plus-visualization workflow.

## Slide 13: Future Scope

Pure forward roadmap, no overlap with slide 10: calibration plus decision-curve analysis for bedside thresholds; subgroup and fairness validation across age, sex, and hospitals; edge and TinyML deployment for wearable live vitals; a prospective clinical pilot with collaborators; and privacy-preserving multi-hospital training.

## Slide 14: Conclusion

SynCura demonstrates an end-to-end explainable ICU risk-monitoring workflow from time-series input to dashboard output. Temporal attention plus SHAP show which time steps and features drove each prediction. The ensemble reached 0.840 validation and 0.834 holdout AUC — supporting the prototype concept while requiring external and clinical validation before real-world use.

## Slide 15: References

Ten entries: Zheng 2025 base paper FIRST as [1], then the 8 supporting studies matching slide 4, plus the PhysioNet 2012 dataset as [10]. Full titles with venues — no truncated citations. Verify final DOIs against publisher records.

## Slide 16: Thank You

Thank you for listening. SynCura is an explainable research prototype for early ICU deterioration-risk monitoring. We welcome your questions and suggestions.

## Common Questions and Answers

### What is hour 12?

Hour 12 is the base paper's fixed trigger for static tasks: data from admission to hour 12 predicts death in 1/2/4/7 days and overall in-hospital mortality. Stays under 12 hours are excluded for lack of data. Dynamic tasks instead predict every hour.

### Why do paper numbers differ from real-world performance?

Retrospective full-stay data accumulates all information till discharge, so AUROC peaks at 0.989 at discharge — hindsight. Cross-hospital transfer drops to 0.81/0.76, with US-only data, missing values, and age bias. SynCura bans future info with a 90-minute-only window and reports the unseen holdout as the honest number.

### Why did you choose an LSTM?

Sequential data; best single architecture in Yan 2026 (beats Transformer/GRU). Cheap streaming inference.

### Why did you add attention?

Assigns importance to time steps — temporal explanation of which recent trajectory parts drove risk.

### Does the system correct itself?

Not live — weights are frozen after training. Correction is mistake-triggered and offline: confirmed misses and false alarms queue into weighted retraining, then a fresh holdout re-validates before redeploy. That is also the safe, regulatable way to do adaptive AI.

No. Research prototype. Needs calibration, prospective testing, external validation, privacy, safety, and regulatory review.

### Why use PhysioNet 2012?

Public, established ICU time-series dataset enabling reproducible experiments. Age and single-source limits acknowledged; set-B holdout is our generalization evidence.

### What is the novelty of SynCura?

Integration of temporal attention, real-time API serving, dashboard visualization, and dual explainability in one prototype — an engineering and integration contribution, not a new LSTM architecture claim.

### What is the main limitation?

Single public dataset, missing/irregular measurements, no external validation yet, no prospective clinical testing.

### How does SynCura compare with reviewed papers?

Papers optimize AUC; SynCura optimizes the path from data to decision — false alarms, sensitivity, specificity, precision, lead time, NEWS2 comparison, unseen holdout 0.834 alongside 0.840 validation.

### What would make SynCura better with more time?

External multi-center validation, generative missing-data recovery, irregular-time awareness, multimodal notes, graph attention, prospective testing.
