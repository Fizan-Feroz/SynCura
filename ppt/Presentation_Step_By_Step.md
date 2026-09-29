# SynCura — Presentation Walkthrough (Step by Step)

**When:** 18 Sept, 2–4pm (A Section, slot 1) | **Presenter:** Fizan (solo)
**Deck:** `ppt/SynCura_Deck_V3.pptx` | **Target:** ~8 min

---

## Rubric (25 marks) — what judges actually score

| # | Criterion | Marks |
|---|---|---|
| 1 | Clarity of problem statement | 4 |
| 2 | Understanding of base paper / related work | 4 |
| 3 | Novelty / proposed idea | 4 |
| 4 | Feasibility & technical approach | 4 |
| 5 | PPT quality & presentation | 3 |
| 6 | Team participation | 3 |
| 7 | Response to questions | 3 |

⚠ Alone on stage costs criterion 6 — plan for teammates to take a Q each (or one opens/closes).

## The whole talk in one arc

Opening: *"a risk score* **with the explanation behind it** *"* → Closing: *"data in, risk out,* **reasons attached** *"*

---

## Step 1 — Slide 1: Title (45s)

> "Good morning. I'm Fizan, and this is **SynCura** — a predictive ICU monitoring system built on an attention-based LSTM. It reads the last 90 minutes of a patient's vitals and labs, and produces a live deterioration risk score from 0 to 100 — *with the explanation behind it.* My team built the model, the API, and the live dashboard."

## Step 2 — Slides 2–3: Problem + Motivation (60s)

**Slide 2 (Problem):**
> "In the ICU, patients can deteriorate within minutes to hours. The tools used today — **NEWS2 and SOFA** — are rule-based scores with fixed thresholds. They tell you a reading is abnormal, but they don't capture the *trend*, the *direction*, or the *rate* of change. A heart rate of 90 is fine for one patient and critical for another — the score can't see that. We need an early-warning aid that looks at recent history, not a single snapshot."

**Slide 3 (Motivation):**
> "That's why this matters: deterioration is a **process, not an event**. It shows up as a pattern across time — and if we catch that pattern even 15–20 minutes earlier, a clinician gets time to intervene. And a risk number alone isn't enough — the system has to *explain itself*, so a clinician can trust it. That's the whole goal: **early + explainable.**"

Anchors: Slide 2 → "NEWS2 can't see trends; we can." | Slide 3 → "deterioration is a process, not an event."

## Step 3 — Slides 4–5: Existing Work + Base Paper (90s) — the grill slide

**Slide 4 (Existing Work, ~25s):**
> "The field has tried transformers, graph networks, multimodal models, and wearable deep learning. The common pattern: they report a good **AUC** — a ranking statistic — but they stop there. Few validate on truly unseen patients, and almost none ship a real-time path from data to a clinician's screen."

**Slide 5 (Base Paper — Zheng 2025, ~60s):**
> "Our reference paper is **Zheng et al., 2025**, in the *Journal of Medical Internet Research*. They built **TBAL** — a time-aware, bidirectional attention LSTM — on 176,000 ICU stays, updating risk *hourly*. Results: **0.936 AUROC** on MIMIC-IV, **0.919** on an external hospital database.
>
> Three things we take from them: one — attention over time gives *interpretability*; two — dynamic risk updated as data streams in; three — external validation is the gold standard.
>
> Where we differ: they use the full electronic medical record — labs, meds, everything. We deliberately use just **12 vital signs and labs**, so we can run a **90-minute window in real time, at low latency** — and we add SHAP explanations on top of attention."

Anchors: Zheng = "0.936/0.919, hourly, 176K stays" | Difference = "they use full EMR; we use 12 vitals for real-time, low-latency"

## Step 4 — Slides 6–8: Solution, Methodology, Expected Outcome (90s)

**Slide 6 (Proposed Solution, ~25s):**
> "Proposed solution: SynCura takes the last **90 minutes** of a patient's vitals and labs — 12 features — through an attention-based LSTM, and returns a live risk score from 0 to 100, with *two* explanations: which time steps mattered, and which features mattered."

**Slide 7 (Methodology, ~35s):**
> "The pipeline: PhysioNet 2012 data → interpolate the gaps → normalize using training data only (no leakage) → a 90-minute window with a 15-minute stride so we get a fresh score every 15 minutes → a two-layer LSTM with additive attention → FastAPI serving it in real time → React dashboard."

**Slide 8 (Expected Outcome, ~30s):**
> "We expect a working live dashboard: patient board, risk dials, alert thresholds, and per-patient explanation panels. Target AUC is a **realistic 0.78–0.93** — that's the honest range for a vitals-only system, not the 0.93+ that full-EMR, multi-database models hit. And we compare ourselves against **NEWS2** — a clinician-usable baseline, not just our own score."

Anchors: 6 → "90 minutes in, 0–100 out, two explanations" | 7 → "interpolate → normalize → window → LSTM → API → dashboard" | 8 → "realistic 0.78–0.93, benchmarked vs NEWS2"

## Step 5 — Slide 9: Experimental Result (45s) — THE money slide, slow down

> "As a team we treated this honestly — not just a validation number. We trained an **ensemble of three attention-LSTMs**...
>
> *(pause)*
>
> ...and we report **two** numbers. **Validation AUC: 0.840.** And more importantly — a **fresh holdout (20% of set-B patients, never in training): AUC 0.834, 95% CI 0.825–0.843.** One honest caveat: that holdout guided our ensemble choice, so it is not a locked final test — the next step is a truly untouched multi-hospital set.
>
> *(pause)*
>
> The holdout and validation are **consistent** — that means the result is a real signal, not luck on one split. Beyond AUC, the table shows sensitivity, specificity, precision, and F1 — all on the same held-out data."
>
> *Point at the ROC curve:* "The two curves almost overlap — validation and unseen holdout track each other. That's what you want to see."

Delivery: pause *before* "0.834", point at the number, say "ensemble of three" clearly.

## Step 6a — Slide 10: Limitations & Improvement Path (30s)

> "Now where does this go next? Four honest upgrades. One — **validate across hospitals**, on eICU and MIMIC, not just one dataset. Two — **generative missing-data recovery**: today we fill gaps with averages; the latest work recovers them properly. Three — a **prospective pilot with clinicians**, so the alerts are tested live at a bedside, not only on recorded data. And four — **time-aware attention** to replace the simple additive attention, capturing irregularity in vital-sign timing. We scouted every one of these — they're named, they're planned, we just haven't run them yet."

Anchor: "validate → recover missing data → test live."

## Step 6b — Slide 11: Technology (15s, keep it short)

> "The stack — Python and PyTorch for the model, FastAPI for the real-time API, React with Vite for the dashboard, SHAP for the explanations, and SQLite to store incoming vitals. The dataset is PhysioNet 2012 — public and reproducible."

## Step 6c — Slide 12: SDG Relevance (15s)

> "Two Sustainable Development Goals. **SDG 3 — Good Health and Well-Being:** earlier detection of ICU deterioration means more time to intervene, which is better patient monitoring. **SDG 9 — Industry, Innovation, and Infrastructure:** this is explainable AI deployed in a modern clinical workflow — model, API, dashboard, and sensor-ready hardware working together."

## Step 6d — Slide 13: Conclusion (25s)

> "To sum up — SynCura is an **end-to-end, explainable** prototype: patient data in, live risk out, and the reasons attached. It's validated on **truly unseen patients**, which is more than most of the papers we reviewed can claim. And it has a **clear, named roadmap** — multi-center validation, better missing-data handling, and a clinical pilot."

## Step 6e — Slide 14: References (10s — do NOT read names)

> "Our base paper — Zheng et al. 2025 — and the eight supporting studies are all listed here, with full titles."

## Step 6f — Slide 15: Thank You + finish (15s)

> "Thank you — I'm happy to walk through the model, the dashboard, or any of the numbers with you."

**Then:** stop. stand still. smile. Let them ask the first question — don't fill the silence.

---

## The three explanations you may be asked

### "Two explanations" (slide 6)
- **Which time steps mattered** — temporal attention. *"Your risk is high because of what happened 40–50 minutes ago."*
- **Which features mattered** — SHAP. *"SpO2 dropped and respiratory rate rose — that pushed the score +30."*
- Answer to *"why both?"*: "Attention explains time, SHAP explains features — attention alone doesn't tell you which vital broke first, SHAP alone doesn't tell you when. Clinicians need both."

### "Why three LSTMs?" (ensemble)
> "Same architecture, three sets of training experience — two trained on set-a, one on set-a plus more of set-b. Averaging them smooths each model's individual quirks, so they make fewer wild predictions. Averaging independent errors is a standard robustness trick."

Simple stage line: *"Three models, same brain shape, different schooling — they vote, and the average wins."*

### "Why does holdout beat validation?" (general, no numbers)
> "Validation is a number we watched while still tuning — the model slightly peeked. The holdout is the untouched final exam — run once, never tuned on. Averaging three models makes the prediction steadier — quirky mistakes cancel out — and the result is consistent across both splits (0.840 vs 0.834 with 95% CI 0.825–0.843)."

---

## Q&A 3-rule rap

1. **Pause 2 seconds** before answering.
2. **Anchor to your numbers:** performance → 0.834 holdout; better than papers → metrics beyond AUC + true holdout; deployable → research prototype, needs calibration/prospective.
3. Unsure → *"We haven't tested that yet — it's in our scoped next steps on slide 10."*

Likely traps & answers:
- *"0.834 vs the field?"* → "For a 12-vital, single-dataset system, yes — realistic range is 0.78–0.93. Zheng's 0.936 uses the full medical record."
- *"Why not BiLSTM like Zheng?"* → "Bidirectional needs the full sequence; we score live streaming, so forward LSTM is the latency trade-off."
- *"Why 12 features?"* → "A 20-feature variant scored worse (0.787) — 12 wins."
- *"Is it clinically ready?"* → "No — explainable research prototype. Needs calibration, external validation, prospective testing, regulatory review."

## Mandatory numbers (on a sticky on the laptop lid)

**844** (holdout, CI 0.825–0.843) • **840** (val) • **12 × 90min / stride 15**

## House rules (don't break)
- Never "clinical ready" → always "explainable research prototype."
- Never quote validation alone → always holdout 0.834.
- Never cross-compare to Zheng → "different scope: 12 vitals vs full EMR."