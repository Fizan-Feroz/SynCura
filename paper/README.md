# SynCura: IEEE Conference Research Paper

This directory contains the camera-ready, complete LaTeX source files and high-resolution figures for the **SynCura** IEEE-style research paper:

> **Title:** *SynCura: A Level-3 Multi-Scale Hybrid Deep Learning System for Real-Time Predictive ICU Monitoring with Dual Explainability and Multi-Tier Alerting*  
> **Authors:** Abdul Ahad Ikkeri, Fathima Reeha T. A., Fizan Feroz, and Divya K.  
> **Affiliation:** Department of Computer Science and Engineering, P. A. College of Engineering, Mangalore, India (Affiliated to Visvesvaraya Technological University, Belagavi)

---

## Directory Contents

- **`syncura_ieee_paper.tex`**: The main IEEE conference paper manuscript in double-column format.
- **`references.bib`**: BibTeX database with 26 complete, verified academic citations.
- **`IEEEtran.cls`**: Official IEEE LaTeX document class (version 1.8b).
- **`figures/`**: 10 publication-quality figures at 300 DPI:
  - `fig1_roc_curves.png`: AUROC curves across cohorts and baselines.
  - `fig2_pr_curves.png`: Precision-Recall curves showing 3.3x prevalence enrichment.
  - `fig3_threshold_tradeoffs.png`: Sensitivity, specificity, and Youden's J across decision thresholds.
  - `fig4_calibration_curves.png`: Probability calibration and Platt scaling reliability diagram.
  - `fig5_confusion_matrices.png`: Confusion matrices across operating tiers.
  - `fig6_benchmark_comparison.png`: Architectural benchmark comparison vs. SAPS-I and sequential models.
  - `fig7_ablation_study.png`: Feature modality ablation analysis.
  - `fig8_subgroup_fairness.png`: Subgroup fairness audit across age, sex, and clinical specialties.
  - `fig9_persistence_filtering.png`: $k=1, 2, 3$ persistence filtering mitigating alarm fatigue.
  - `fig10_shap_importance.png`: Global TreeSHAP feature importance ranking.
- **`validate_latex.py`**: Python script to verify brace balance, environment parity, citation integrity, and figure existence.

---

## How to Compile

### 1. On Overleaf (Recommended)
1. Zip the contents of this `PROJ/paper/` directory:
   - `syncura_ieee_paper.tex`
   - `references.bib`
   - `IEEEtran.cls`
   - `figures/`
2. Open [Overleaf](https://www.overleaf.com), click **New Project** $\rightarrow$ **Upload Project**, and select the zip file.
3. Set the compiler to **pdfLaTeX** (default) or **XeLaTeX**, and set the main document to `syncura_ieee_paper.tex`.
4. Click **Recompile** to generate the camera-ready PDF.

### 2. Locally via Command Line (TeX Live / MiKTeX / MacTeX)
Run the standard four-pass compilation sequence:

```bash
pdflatex syncura_ieee_paper.tex
bibtex syncura_ieee_paper
pdflatex syncura_ieee_paper.tex
pdflatex syncura_ieee_paper.tex
```

Or using `latexmk`:

```bash
latexmk -pdf syncura_ieee_paper.tex
```

---

## Verification

To run automated integrity checks:

```bash
python validate_latex.py
```
