# -*- coding: utf-8 -*-
"""SynCura — Project Synopsis V4 in the official college template order.

Sections follow PROJECT-PPT-FORMAT.pptx:
Title, Abstract, Introduction, Literature Survey, Problem Statement, Objectives,
Proposed Methodology, Software & Hardware Requirement Specification, Design,
Implementation, Project Demo Screenshots, Result and Discussion,
Conclusion & Future Scope, References, Thank You — plus a Limitations slide.

Verified facts:
  split_patient_ids.json       9,802 train / 800 val / 638 holdout
  level3_summary.json          hybrid holdout 0.843
  clinical_validation_summary  AUPRC 0.469, Brier 0.152, ECE 0.193, SAPS-I 0.675
  threshold_optimization       16,944 windows, operating curve
  deployed_manifest.json       ensemble 0.840 val / 0.834 holdout
  backend/inference.py         serves the Attention-LSTM ensemble only
  ml/explain.py                KernelSHAP for the served model
"""
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

INK = RGBColor(0x11, 0x11, 0x11)
SOFT = RGBColor(0x44, 0x44, 0x44)
MUTED = RGBColor(0x66, 0x66, 0x66)
NAVY = RGBColor(0x1F, 0x38, 0x64)
RULE = RGBColor(0xBF, 0xBF, 0xBF)
FILL = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Times New Roman"
SW, SH = 13.333, 7.5
M = 0.85
CW = SW - 2 * M

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "PROJECT SYNOPSIS PPT-FORMAT_V4.pptx")
SHOTS = os.path.join(HERE, "shots")
ASSETS = os.path.join(HERE, "assets")


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def text(slide, x, y, w, h, value, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
         spacing=1.5, italic=False, anchor=MSO_ANCHOR.TOP, space_after=0,
         caps=False, char_spacing=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, line in enumerate(value.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = line
        f = r.font
        f.name = FONT
        f.size = Pt(size)
        f.bold = bold
        f.italic = italic
        f.color.rgb = color
        if caps:
            r._r.get_or_add_rPr().set("cap", "all")
        if char_spacing is not None:
            r._r.get_or_add_rPr().set("spc", str(int(char_spacing * 100)))
    return box


def bullets(slide, x, y, w, h, items, size=17, spacing=1.35, gap=6, color=INK, mark="\u2022"):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.JUSTIFY
        p.line_spacing = spacing
        p.space_after = Pt(gap)
        pPr = p._p.get_or_add_pPr()
        if mark:
            pPr.set("marL", str(int(0.30 * 914400)))
            pPr.set("indent", str(int(-0.30 * 914400)))
            bu = pPr.makeelement(
                "{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": mark}
            )
            pPr.append(bu)
        lead, rest = item if isinstance(item, tuple) else (None, item)
        if lead:
            r = p.add_run()
            r.text = lead
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = True
            r.font.color.rgb = color
        r = p.add_run()
        r.text = rest
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.color.rgb = color
    return box


def header(slide, title=None):
    slide.shapes.add_picture(os.path.join(ASSETS, "pace-logo.jpeg"), Inches(0.45), Inches(0.20),
                             width=Inches(1.30))
    slide.shapes.add_picture(os.path.join(ASSETS, "pa-college-crest.jpeg"),
                             Inches(SW - 0.65 - 0.60), Inches(0.12), width=Inches(0.60))
    text(slide, 1.95, 0.20, SW - 3.9, 0.34, "P.A. COLLEGE OF ENGINEERING", size=20, bold=True,
         align=PP_ALIGN.CENTER, spacing=1.0, char_spacing=0.4)
    text(slide, 1.95, 0.54, SW - 3.9, 0.24, "MANGALURU", size=10, bold=True,
         align=PP_ALIGN.CENTER, spacing=1.0, char_spacing=1.2)
    text(slide, 1.95, 0.75, SW - 3.9, 0.28, "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING",
         size=12.5, bold=True, align=PP_ALIGN.CENTER, spacing=1.0, char_spacing=0.3)
    if title:
        text(slide, M, 1.16, CW, 0.5, title, size=26, bold=True, align=PP_ALIGN.CENTER,
             spacing=1.0, caps=True, char_spacing=0.4)


def footer(slide, n):
    text(slide, M, 7.12, 5.0, 0.24, "DEPT. OF CSE  \u00b7  2026-27", size=9.5, color=MUTED, spacing=1.0)
    text(slide, SW - M - 1.2, 7.12, 1.2, 0.24, str(n), size=9.5, color=MUTED,
         align=PP_ALIGN.RIGHT, spacing=1.0)


def notes(slide, value):
    slide.notes_slide.notes_text_frame.text = value


def stat(slide, x, y, w, h, value, label, sub):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid(); box.fill.fore_color.rgb = FILL
    box.line.color.rgb = RULE; box.line.width = Pt(0.75)
    box.shadow.inherit = False
    text(slide, x + 0.2, y + 0.10, w - 0.4, 0.5, value, size=28, bold=True, color=NAVY, spacing=0.95)
    text(slide, x + 0.2, y + 0.66, w - 0.4, 0.28, label, size=11, bold=True, color=INK, spacing=1.0, caps=True)
    text(slide, x + 0.2, y + 0.94, w - 0.4, 0.45, sub, size=10, color=SOFT, spacing=1.05)


def table(slide, x, y, w, rows, widths, header_row, size=10.5, row_h=0.4, header_h=0.34):
    shape = slide.shapes.add_table(len(rows) + 1, len(widths), Inches(x), Inches(y), Inches(w),
                                   Inches(header_h + row_h * len(rows)))
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = False
    for i, wd in enumerate(widths):
        tbl.columns[i].width = Inches(wd)
    for j, val in enumerate(header_row):
        cell = tbl.cell(0, j)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.06)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = val
        r.font.size = Pt(size); r.font.bold = True; r.font.name = FONT; r.font.color.rgb = WHITE
    tbl.rows[0].height = Inches(header_h)
    for i, row in enumerate(rows, start=1):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if i % 2 else FILL
            cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.06)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = val
            r.font.size = Pt(size); r.font.name = FONT; r.font.color.rgb = INK
            if j == 0:
                r.font.bold = True
    return tbl


def box(slide, x, y, w, h, title, sub=None, fill=FILL, title_size=10.5, sub_size=8.5):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = RULE; shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    text(slide, x + 0.08, y + 0.07, w - 0.16, 0.28, title, size=title_size, bold=True,
         color=INK, align=PP_ALIGN.CENTER, spacing=1.0)
    if sub:
        text(slide, x + 0.08, y + 0.30, w - 0.16, h - 0.34, sub, size=sub_size, color=SOFT,
             align=PP_ALIGN.CENTER, spacing=1.0)


def arrow(slide, x, y, w=0.26, h=0.11):
    from pptx.enum.shapes import MSO_CONNECTOR
    from pptx.oxml.ns import qn
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y + 0.05),
                                    Inches(x + w), Inches(y + 0.05))
    ln.line.color.rgb = NAVY
    ln.line.width = Pt(1.25)
    lnPr = ln.line._get_or_add_ln()
    tail = lnPr.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    lnPr.append(tail)


prs = Presentation()
prs.slide_width = Inches(SW)
prs.slide_height = Inches(SH)
n = 0

# ------------------------------------------------------------------ 1 title
s = blank(prs); n += 1
header(s)
text(s, M, 1.62, CW, 0.3, "PROJECT SYNOPSIS  \u2014  VERSION 4", size=13, bold=True, color=NAVY,
     align=PP_ALIGN.CENTER, spacing=1.0, char_spacing=0.8)
text(s, M, 1.98, CW, 0.72, "SynCura", size=44, bold=True, align=PP_ALIGN.CENTER, spacing=1.0)
s.shapes.add_picture(os.path.join(ASSETS, "syncura-mark.png"), Inches(4.50), Inches(2.06), width=Inches(0.58))
text(s, M + 0.8, 2.72, CW - 1.6, 0.9,
     "Predictive ICU Monitoring System using a Multi-Scale Attention-LSTM\nand LightGBM Hybrid with Real-Time Explainability",
     size=18, align=PP_ALIGN.CENTER, spacing=1.12)
text(s, M, 3.92, CW, 0.3, "Submitted in partial fulfilment of the requirements for the project synopsis",
     size=12, italic=True, color=MUTED, align=PP_ALIGN.CENTER, spacing=1.0)
text(s, M + 1.2, 4.55, 4.6, 0.3, "Under the guidance of", size=11.5, color=SOFT, spacing=1.0)
text(s, M + 1.2, 4.85, 4.6, 0.35, "Divya K", size=16, bold=True, spacing=1.0)
text(s, M + 1.2, 5.18, 4.6, 0.3, "Department of CSE", size=11, color=SOFT, spacing=1.0)
text(s, M + 6.6, 4.55, 4.6, 0.3, "Submitted by", size=11.5, color=SOFT, spacing=1.0)
text(s, M + 6.6, 4.85, 4.8, 0.95,
     "Abdul Ahad Ikkeri  \u2014  4PA24CS002\nFathima Reeha  \u2014  4PA24CS026\nFizan Feroz  \u2014  4PA24CS032",
     size=13.5, spacing=1.12)
footer(s, n)
notes(s, "Title slide in the official format. V4 supersedes the pre-development synopsis: shipped system, verified "
         "results, and the limitations we actually hit.")

# ------------------------------------------------------------------ 2 abstract
s = blank(prs); n += 1
header(s, "Abstract")
body = [
    ("Problem. ", "ICU deterioration is often visible hours before a crash, but NEWS2, SOFA and APACHE-II score a single "
                  "time point and miss direction, duration and interaction between vitals."),
    ("Method. ", "SynCura models 12 vitals and labs over 90-minute causal windows, fused with 8 static covariates and "
                 "36 twenty-four-hour trend features, and adds a LightGBM branch over 116 engineered clinical features."),
    ("Data. ", "PhysioNet 2012 Set A/B/C (12,000 stays). The audited split uses 9,802 training patients, 800 validation "
               "patients and 638 unseen Set-B holdout patients, with zero patient overlap."),
    ("Results. ", "The hybrid reaches 0.843 AUROC on 16,944 holdout windows, against 0.675 for SAPS-I on the same cohort; "
                  "at its best-accuracy threshold the LightGBM branch reaches 86.7% accuracy at 96.0% specificity. The "
                  "deployed three-model Attention-LSTM ensemble reports 0.840 validation and 0.834 holdout AUROC."),
    ("System. ", "FastAPI ingestion and scoring, a React central station with live, replay and demo sources, risk-ranked "
                 "beds, ward alerts with audible cues, and SHAP plus temporal-attention explanations through the API."),
    ("Status. ", "A research prototype. It is not a medical device, has no prospective or external validation, and its "
                 "probabilities are not yet calibrated (ECE 0.193)."),
]
bullets(s, M, 1.78, CW, 5.2, body, size=15, spacing=1.3, gap=8, mark=None)
footer(s, n)
notes(s, "Six sentences: problem, method, data, results, system, status. The caveat is deliberate.")

# ------------------------------------------------------------------ 3 introduction
s = blank(prs); n += 1
header(s, "Introduction")
text(s, M, 1.72, 5.65, 0.3, "Background", size=14, bold=True, color=NAVY, spacing=1.0, caps=True, char_spacing=0.5)
bullets(s, M, 2.05, 5.55, 4.4, [
    "ICU mortality ranges from 10% to 29%; most deterioration is preceded by hours of measurable drift.",
    "NEWS2, SOFA and APACHE-II are computed by hand at rounds from one moment in time, so gradual trends stay invisible.",
    "Single-threshold alarms produce late escalation and alarm fatigue, and give no reason for the score.",
    "Existing models report AUROC but rarely calibration, alarm burden or a working bedside path.",
], size=14, spacing=1.26, gap=8)
text(s, 6.75, 1.72, 5.7, 0.3, "Scope of this work", size=14, bold=True, color=NAVY, spacing=1.0, caps=True, char_spacing=0.5)
bullets(s, 6.75, 2.05, 5.6, 4.4, [
    ("Task. ", "In-hospital mortality risk for ICU stays, scored continuously from streaming vitals and labs."),
    ("Not a diagnosis. ", "The output is a risk score with reasons, not a sepsis label, a diagnosis or an autonomous decision."),
    ("Deliverable. ", "A trained model, a serving API, a dashboard, explanations and a documented evaluation."),
    ("Constraint. ", "Only public, de-identified PhysioNet data; no hospital data leaves the machine."),
], size=13.5, spacing=1.24, gap=8)
footer(s, n)
notes(s, "Frame the clinical gap and the exact scope. The scope sentence prevents the sepsis question.")

# ------------------------------------------------------------------ 4 literature
s = blank(prs); n += 1
header(s, "Literature Survey")
rows = [
    ("Dynamic real-time ICU risk model", "Zheng 2025", "Vitals, labs, meds", "Retrospective; drops cross-hospital"),
    ("DL mortality on long ICU sequences", "Yan 2026", "ICD + temporal data", "AUC 0.802; stacking complexity"),
    ("Multimodal deterioration with notes", "Sadanandan 2026", "Time series + BERT", "Privacy, compute; low AUPRC"),
    ("Vital-signs-only real-time risk", "Wu 2024", "BP, HR, RR, temp", "8x data; no explanation"),
    ("Attention-LSTM risk (our ancestor)", "Nguyen 2017", "PhysioNet 2012", "Legacy; limited external validity"),
    ("Generative real-time imputation", "Xie 2025", "Recovered vitals", "Needs multi-hospital data"),
    ("DEWS interpretable early warning", "Choi 2020", "Vitals + attention", "Attention is not causation"),
    ("Attention residual LSTM-FCN", "Li 2025", "Clinical time series", "Complex; limited external validation"),
    ("Expert-augmented early warning", "Wang 2026", "Rules + ML", "Rule maintenance; single health system"),
    ("Wearable deterioration prediction", "Scheid 2025", "Wearable vitals", "Sensor noise and dropouts"),
]
table(s, M, 1.66, CW, rows, [4.35, 1.55, 2.55, 3.17],
      ["Study", "Author \u00b7 Year", "Inputs", "What it leaves open"], size=9.5, row_h=0.40, header_h=0.32)
text(s, M, 6.12, CW, 0.7,
     "Ten studies, one pattern: strong discrimination on retrospective cohorts, weak calibration reporting, and almost "
     "no deployed bedside path. SynCura targets the deployment and explanation gap.",
     size=11.5, spacing=1.2, italic=True, color=SOFT)
footer(s, n)
notes(s, "Name the base paper (Zheng 2025) and DEWS (Choi 2020). The last column justifies the project.")

# ------------------------------------------------------------------ 5 problem
s = blank(prs); n += 1
header(s, "Problem Statement")
text(s, M, 1.75, CW, 1.1,
     "ICU patients need continuous risk assessment, but bedside scores depend on single-time-point thresholds and manual "
     "calculation. Vital-sign trajectories, the interaction between variables and the duration of abnormality are ignored, "
     "which produces late detection, alarm fatigue and limited sensitivity for early deterioration.",
     size=15.5, spacing=1.4)
text(s, M, 2.98, CW, 0.3, "What a usable system must do", size=14, bold=True, color=NAVY, spacing=1.0, caps=True, char_spacing=0.5)
bullets(s, M, 3.32, CW, 2.9, [
    ("Learn time, not only level. ", "A sequence model over 90-minute windows, with 24-hour trend context."),
    ("Show its reasons. ", "Temporal attention for when the patient changed; SHAP for which signal drove the score; a "
                           "bedside action attached to every alert."),
    ("Survive a ward. ", "A real-time API, a dashboard that ranks beds continuously, and alerts that escalate without "
                         "repeating every tick."),
    ("Stay scoped. ", "Predict in-hospital mortality risk for ICU stays; not sepsis onset, not diagnosis, not an "
                      "autonomous clinical decision."),
], size=14.5, spacing=1.28, gap=8)
footer(s, n)
notes(s, "Three failure modes, three requirements, one scope sentence.")

# ------------------------------------------------------------------ 6 objectives
s = blank(prs); n += 1
header(s, "Objectives")
objs = [
    ("O1  Temporal model. ", "Learn mortality risk from 90-minute vital-sign windows instead of single time points, with "
                             "causal interpolation and no future leakage."),
    ("O2  Multi-scale context. ", "Add 8 static covariates and 36 twenty-four-hour trend features, plus a LightGBM branch "
                                  "over 116 engineered clinical features."),
    ("O3  Real-time system. ", "Build the full path: ingestion API, SQLite store, scoring engine and React dashboard that "
                               "ranks beds every five seconds."),
    ("O4  Explainability. ", "Return SHAP attribution and temporal attention for each score, and translate them into a "
                             "bedside action."),
    ("O5  Honest evaluation. ", "Report a locked holdout, confidence intervals, AUPRC, calibration, an operating curve and "
                                "a SAPS-I baseline — and state the limitations."),
]
y = 1.72
for lead, rest in objs:
    text(s, M, y, 0.7, 0.4, lead.split("  ")[0], size=16, bold=True, color=NAVY, spacing=1.0)
    p = s.shapes.add_textbox(Inches(M + 0.85), Inches(y - 0.02), Inches(CW - 0.85), Inches(0.92))
    tf = p.text_frame
    tf.word_wrap = True
    tf.margin_left = 0; tf.margin_right = 0; tf.margin_top = 0; tf.margin_bottom = 0
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.JUSTIFY
    para.line_spacing = 1.25
    r1 = para.add_run(); r1.text = lead.split("  ", 1)[1]
    r1.font.name = FONT; r1.font.size = Pt(13.5); r1.font.bold = True; r1.font.color.rgb = INK
    r2 = para.add_run(); r2.text = rest
    r2.font.name = FONT; r2.font.size = Pt(13.5); r2.font.color.rgb = INK
    y += 0.92
footer(s, n)
notes(s, "Five objectives, each tied to something demonstrable in the running system.")

# ------------------------------------------------------------------ 7 methodology
s = blank(prs); n += 1
header(s, "Proposed Methodology")
steps = [
    ("1. Corpus. ", "PhysioNet 2012 Set A, B and C (12,000 stays, 37 variables, 48-hour records, mortality labels)."),
    ("2. Windows. ", "90-minute causal windows; stride 30 for training and 15 for validation; interpolation uses only past "
                     "samples, and normalisation statistics come from the training split only."),
    ("3. Sequence branch. ", "Two-layer LSTM (hidden 96, dropout 0.3) with additive Bahdanau attention over time, fused "
                             "with an 8-dimension static vector and 36 twenty-four-hour trend features through a context MLP."),
    ("4. Tabular branch. ", "LightGBM gradient-boosted trees over 116 engineered features: summary moments, volatility, "
                            "shock indices and laboratory panels."),
    ("5. Fusion. ", "Branch probabilities are blended by a validation-tuned weight (0.1 neural + 0.9 LightGBM) and mapped "
                    "to a 0-100 risk."),
    ("6. Serving. ", "POST /ingest, SQLite, ensemble scorer, KernelSHAP and attention on request, React dashboard, "
                     "warning and critical sound cues."),
]
bullets(s, M, 1.72, CW, 4.9, steps, size=14, spacing=1.3, gap=8, mark=None)
text(s, M, 6.4, CW, 0.5,
     "Deployment uses the three-member Attention-LSTM ensemble (e12 + e13 + c53); the Level 3 hybrid is evaluated offline "
     "and is not yet in the serving path.",
     size=11, italic=True, color=MUTED, spacing=1.1)
footer(s, n)
notes(s, "Walk the six stages. Be explicit that the served model is the LSTM ensemble, not the hybrid.")

# ------------------------------------------------------------------ 8 SRS
s = blank(prs); n += 1
header(s, "Software & Hardware Requirement Specification")
text(s, M, 1.72, 5.55, 0.3, "Software requirements", size=14, bold=True, color=NAVY, spacing=1.0, caps=True, char_spacing=0.5)
bullets(s, M, 2.05, 5.5, 4.3, [
    ("Operating system: ", "Windows 10/11 or Linux (Ubuntu 22.04+)."),
    ("ML stack: ", "Python 3.11, PyTorch, NumPy, pandas, scikit-learn, LightGBM, SHAP, matplotlib."),
    ("Backend: ", "FastAPI + Uvicorn, SQLAlchemy + SQLite, python-dotenv, paho-mqtt for device ingest."),
    ("Frontend: ", "Node.js 20, Vite, React 18, React Router, GSAP; plain CSS design tokens."),
    ("Tooling: ", "Git, pytest, Ruff/Pylance; Render for the API, Vercel for the web app."),
], size=13, spacing=1.3, gap=8)
text(s, 6.9, 1.72, 5.55, 0.3, "Hardware requirements", size=14, bold=True, color=NAVY, spacing=1.0, caps=True, char_spacing=0.5)
bullets(s, 6.9, 2.05, 5.55, 4.3, [
    ("Development: ", "x86-64 laptop or desktop, 16 GB RAM, 20 GB free disk; GPU optional for retraining."),
    ("Serving: ", "1 vCPU and 1 GB RAM is sufficient for the deployed ensemble; no GPU at inference."),
    ("Storage: ", "2.3 GB for the public datasets (not committed); 90 KB for the deployed weights."),
    ("Client: ", "Any modern browser at 1280 px or wider; audio output for the alert cues."),
    ("Network: ", "HTTPS to the API; MQTT broker only if physical devices publish vitals."),
], size=13, spacing=1.3, gap=8)
text(s, M, 6.55, CW, 0.4,
     "Minimum viable deployment: one 1 vCPU / 1 GB instance for the API plus a static frontend; retraining is optional and not required to run the demo.",
     size=10.5, italic=True, color=MUTED, spacing=1.05)
footer(s, n)
notes(s, "Read the stack quickly; the point is that it runs on commodity hardware.")

# ------------------------------------------------------------------ 9 design
s = blank(prs); n += 1
header(s, "Design")
text(s, M, 1.62, CW, 0.28, "System architecture", size=13, bold=True, color=NAVY, spacing=1.0, caps=True, char_spacing=0.5)
arch = [
    ("Vitals & labs", "devices, replay,\ndemo engine"),
    ("FastAPI", "/ingest\nvalidation"),
    ("SQLite", "vitals store\n+ patients"),
    ("Risk engine", "3 x Attention-LSTM\nlogit average"),
    ("Explainer", "KernelSHAP +\nattention"),
    ("Dashboard", "React station\n+ bed profile"),
]
bw, gap = 1.72, 0.34
x = M
for i, (t, sub) in enumerate(arch):
    box(s, x, 1.95, bw, 0.95, t, sub)
    if i < len(arch) - 1:
        arrow(s, x + bw + 0.06, 2.35, 0.22)
    x += bw + gap

text(s, M, 3.22, CW, 0.28, "Work flow", size=13, bold=True, color=NAVY, spacing=1.0, caps=True, char_spacing=0.5)
flow = [
    ("1", "Acquire", "12 channels,\ntimestamped"),
    ("2", "Window", "90 min,\nstride 15-30"),
    ("3", "Normalise", "train-only\nz-score"),
    ("4", "Score", "0-100 risk\nevery 5 s"),
    ("5", "Explain", "attention +\nSHAP"),
    ("6", "Alert", "threshold,\ncue, action"),
]
x = M
for num, t, sub in flow:
    box(s, x, 3.55, bw, 0.95, f"{num}. {t}", sub, fill=WHITE)
    if num != "6":
        arrow(s, x + bw + 0.06, 3.95, 0.22)
    x += bw + gap

text(s, M, 4.72, CW, 0.3, "Data contract", size=13, bold=True, color=NAVY, spacing=1.0, caps=True, char_spacing=0.5)
bullets(s, M, 5.02, CW, 1.7, [
    ("Feature contract. ", "12 features in the order fixed by SERVING_FEATURES; scalers are rejected at load if the list "
                           "does not match."),
    ("Split contract. ", "Patient-level group split, 9,802 train / 800 validation / 638 holdout, zero overlap audited in code."),
    ("Serving contract. ", "The same window length (90) and normalisation are used in training, replay and live scoring."),
], size=12, spacing=1.18, gap=5)
text(s, M, 6.35, CW, 0.4,
     "Deployment: React build on Vercel, FastAPI on Render, SQLite on a persistent disk. The browser never holds the model or the weights.",
     size=10.5, italic=True, color=MUTED, spacing=1.05)
footer(s, n)
notes(s, "Two rows: the architecture at rest and the request flow. Then the three contracts that keep train and serve aligned.")

# ------------------------------------------------------------------ 10 implementation
s = blank(prs); n += 1
header(s, "Implementation")
cards = [
    ("Data pipeline", "Loaders for PhysioNet 2012 A/B/C and Challenge 2019; causal sliding windows; train-only scalers; "
                      "patient-level splits with locked slices."),
    ("Models", "AttentionLSTMModel (2 x 96, additive attention) and MultiScaleFusionAttentionLSTM; LightGBM on 116 tabular "
               "features; validation-gated ensemble selection."),
    ("Serving", "RiskScoreEngine loads the manifest ensemble, holds a thread lock, validates the feature order, and exposes "
                "18 endpoints including /ingest, /patients and /explain."),
    ("Explainability", "KernelSHAP (10 background samples, 100 evaluations) plus attention weights; clinical_explainer.py "
                       "maps them to reference ranges and bedside actions."),
    ("Frontend", "React 18 + Vite: central station, bed profile, live/replay/demo sources, threshold tuner, NEWS2 "
                 "comparison, warning and critical sound cues."),
    ("Quality", "Five pytest modules cover API, simulation, dataset policy, the 12-feature contract and locked evaluation; "
                "CI runs them plus the production build."),
]
x, y = M, 1.62
for i, (t, body) in enumerate(cards):
    cx = M + (i % 2) * 6.0
    cy = 1.62 + (i // 2) * 1.38
    box(s, cx, cy, 5.85, 1.22, t, None, fill=WHITE)
    text(s, cx + 0.18, cy + 0.36, 5.5, 0.85, body, size=10.5, color=SOFT, spacing=1.1)
text(s, M, 5.82, CW, 0.35,
     "3 ensemble members  ·  18 API endpoints  ·  5 pytest modules  ·  12 served features  ·  5 simulation scenarios",
     size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER, spacing=1.0)
text(s, M, 6.35, CW, 0.4,
     "Repository layout: ml/ (data, models, training, audit), backend/ (API, inference, simulation), frontend/ (dashboard), backend/tests/ (CI).",
     size=10.5, italic=True, color=MUTED, spacing=1.0)
footer(s, n)
notes(s, "Six implementation blocks. Mention the feature contract and the audit tests; those are what keep the numbers honest.")

# ------------------------------------------------------------------ 11 screenshots
s = blank(prs); n += 1
header(s, "Project Demo Screenshots")
big_w = 7.45
big_h = big_w * 1000 / 1600
s.shapes.add_picture(os.path.join(SHOTS, "dashboard-critical-dark.png"), Inches(M), Inches(1.62), width=Inches(big_w))
small_w = 4.15
small_h = small_w * 1000 / 1600
sx = SW - M - small_w
s.shapes.add_picture(os.path.join(SHOTS, "bed-profile-dark-alert.png"), Inches(sx), Inches(1.62), width=Inches(small_w))
for pic in (s.shapes[-2], s.shapes[-1]):
    pic.line.color.rgb = RULE
    pic.line.width = Pt(0.75)
text(s, M, 1.62 + big_h + 0.06, big_w, 0.5,
     "Central station, dark monitor theme: three beds in the critical band, red pulse, and an active critical alert.",
     size=10.5, italic=True, color=SOFT, spacing=1.1)
text(s, sx, 1.62 + small_h + 0.06, small_w, 0.3,
     "Bed profile (dark): alert, NEWS2 and driver view.",
     size=10, italic=True, color=SOFT, spacing=1.1)
bullets(s, sx, 4.78, small_w, 1.8, [
    ("Alert: ", "risk, pattern, cause and the next bedside action."),
    ("NEWS2: ", "rule-based score for the same bed, for comparison."),
    ("Drivers: ", "a simplified bed-state view, labelled as not SHAP."),
    ("Scope: ", "demo beds are local, not synced, not for clinical use."),
], size=10, spacing=1.14, gap=5)
text(s, sx, 6.5, small_w, 0.3,
     "Running build v1.12.1, dark monitor theme, septic-shock scenario.",
     size=9, italic=True, color=MUTED, spacing=1.0)
footer(s, n)
notes(s, "Point at the three critical beds and the alert card. Say plainly that this is a demo source, not live patients.")

# ------------------------------------------------------------------ 12 results
s = blank(prs); n += 1
header(s, "Result and Discussion")
stat(s, M, 1.7, 3.4, 1.55, "0.843", "Holdout AUROC", "hybrid, 16,944 windows")
stat(s, 4.5, 1.7, 3.4, 1.55, "86.7%", "Peak accuracy", "LightGBM branch, threshold 0.80")
stat(s, 8.15, 1.7, 3.4, 1.55, "0.675", "SAPS-I AUROC", "same holdout, +0.168")
table(s, M, 3.52, CW,
      [["0.50 (safety)", "76.6%", "78.3%", "76.3%", "35.3%"],
       ["0.65 (peak F1)", "82.3%", "61.5%", "85.7%", "41.6%"],
       ["0.80 (precision)", "86.7%", "30.3%", "96.0%", "55.3%"]],
      [2.55, 2.1, 2.3, 2.3, 2.25],
      ["Threshold", "Accuracy", "Sensitivity", "Specificity", "Precision"], size=11, row_h=0.42)
text(s, M, 5.35, 6.3, 1.4,
     "Curve: LightGBM branch, 16,944 windows. AUPRC 0.469 at 14.2% prevalence; Brier 0.152; ECE 0.193. "
     "The calibration gap is reported, not hidden.",
     size=11, spacing=1.2, color=SOFT)
text(s, 7.35, 5.35, 5.2, 1.4,
     "Deployed ensemble: 0.840 validation, 0.834 unseen holdout. A separate threshold-sweep run reports 0.847 on "
     "the same windows.",
     size=11, spacing=1.2, color=SOFT)
footer(s, n)
notes(s, "Lead with AUROC and the SAPS-I comparison, then the operating curve. If asked about 78% accuracy, explain "
         "14% prevalence: a majority-class model scores 85.8% with zero recall.")

# ------------------------------------------------------------------ 13 limitations
s = blank(prs); n += 1
header(s, "Limitations")
left = [
    ("Single-corpus evidence. ", "The reported holdout is PhysioNet 2012 Set B only; no external hospital has scored this model."),
    ("Small, imbalanced holdout. ", "638 patients and 14.2% positives; the AUROC interval spans 0.804-0.877."),
    ("Selection touched the holdout. ", "Members were chosen using validation gating and the holdout was consulted during selection, so it is not a sealed test set."),
    ("Poor calibration. ", "ECE 0.193 and Brier 0.152; probabilities are not reliable enough for absolute risk communication."),
    ("Alarm burden. ", "Precision is about 35% at the safety threshold, and cues can repeat every 1.2 seconds across escalating beds."),
]
right = [
    ("Mortality is not sepsis. ", "The label is in-hospital death; Challenge-2019 sepsis labels were never pooled, and that schema lacks GCS."),
    ("Level 3 is offline. ", "The hybrid is trained and evaluated, but serving still runs the Attention-LSTM ensemble."),
    ("Not a medical device. ", "No regulatory clearance, no prospective trial, no IRB protocol and no clinical sign-off."),
    ("Not HIPAA-ready. ", "No authentication or audit trail on patient data; the training UI is hidden by build flag, not secured."),
    ("Explanation is correlational. ", "Attention and SHAP show what the model used, not what caused the deterioration."),
]
bullets(s, M, 1.75, 5.6, 4.7, left, size=12, spacing=1.2, gap=8)
bullets(s, 6.9, 1.75, 5.6, 4.7, right, size=12, spacing=1.2, gap=8)
text(s, M, 6.45, CW, 0.5,
     "These are the first questions an examiner should ask, and the reason every claim here is scoped to a research prototype.",
     size=11, italic=True, color=MUTED, spacing=1.1)
footer(s, n)
notes(s, "Own the calibration error and the holdout-selection issue.")

# ------------------------------------------------------------------ 14 conclusion & future scope
s = blank(prs); n += 1
header(s, "Conclusion & Future Scope")
text(s, M, 1.7, 5.5, 0.3, "Conclusion", size=14, bold=True, color=NAVY, spacing=1.0, caps=True, char_spacing=0.5)
bullets(s, M, 2.02, 5.45, 2.6, [
    "A working, explainable ICU risk prototype: 0.843 holdout AUROC against 0.675 for SAPS-I on the same 638 patients.",
    "The full path is shipped: ingestion API, ensemble scorer, SHAP and attention explanations, dashboard and alerts.",
    "The evidence is honest but narrow: retrospective, single-corpus, uncalibrated, and not a medical device.",
    "Every reported number traces to a committed artifact (manifest, audit JSON) or a reproducible script.",
], size=12, spacing=1.2, gap=7)
text(s, 6.85, 1.7, 5.6, 0.3, "Future scope", size=14, bold=True, color=NAVY, spacing=1.0, caps=True, char_spacing=0.5)
bullets(s, 6.85, 2.02, 5.6, 2.6, [
    "Calibrate and publish decision-curve analysis; target ECE below 0.05.",
    "Seal the holdout, then validate externally on MIMIC-IV and eICU under credentialing.",
    "Promote the Level 3 hybrid into serving behind a latency budget and A/B test.",
    "Alarm workflow with acknowledgement and measured alarm burden.",
    "Multimodal notes and an IRB-approved prospective silent trial.",
    "Federated training and edge deployment for privacy-preserving monitoring.",
    "Publish a model card: cohort, intended use, failure modes and calibration.",
], size=12, spacing=1.18, gap=6)
text(s, M, 5.55, CW, 1.2,
     "Long term, SynCura would move from a research prototype to a regulated clinical tool only through prospective "
     "validation, calibration, clinician-in-the-loop evaluation and software-as-a-medical-device documentation. None of "
     "those steps has been taken yet, and the current system must not be used for patient care.",
     size=11.5, spacing=1.2, color=SOFT, italic=True)
footer(s, n)
notes(s, "Close with what is proven and what is not. Invite the hard question.")

# ------------------------------------------------------------------ 15 references
s = blank(prs); n += 1
header(s, "References")
left_refs = [
    "[1] Z. Zheng et al., \u201cDevelopment and validation of a dynamic real-time risk prediction model for ICU patients,\u201d J. Med. Internet Res., 2025.",
    "[2] Z. Yan et al., \u201cDeep learning-based in-hospital mortality prediction using long-term sequential ICU data,\u201d PeerJ, 2026.",
    "[3] B. Sadanandan, \u201cMultimodal ICU deterioration prediction,\u201d arXiv, 2026.",
    "[4] Y. Wu et al., \u201cVital signs for real-time ICU mortality risk,\u201d J. Big Data, 2024.",
    "[5] M. Nguyen et al., \u201cDeep learning to attend to risk in ICU,\u201d 2017.",
]
right_refs = [
    "[6] P. Xie et al., \u201cRealMIP: real-time risk with continuous data recovery,\u201d npj Digit. Med., 2025.",
    "[7] E. Choi et al., \u201cDeep interpretable early warning system for clinical deterioration,\u201d IEEE JBHI, 2020.",
    "[8] Y. Li et al., \u201cAttention residual LSTM-FCN for inpatient mortality,\u201d IEEE Access, 2025.",
    "[9] L. Wang et al., \u201cExpert augmented prediction of circulatory and respiratory instability,\u201d npj Digit. Med., 2026.",
    "[10] M. R. Scheid et al., \u201cWearable deep learning for continuous deterioration prediction,\u201d Nature Commun., 2025.",
]
bullets(s, M, 1.72, 5.6, 4.9, left_refs, size=11, spacing=1.18, gap=7, mark=None)
bullets(s, 6.9, 1.72, 5.6, 4.9, right_refs, size=11, spacing=1.18, gap=7, mark=None)
text(s, M, 6.5, CW, 0.35,
     "Datasets: PhysioNet 2012 (A/B/C), PhysioNet 2019 (A/B), VitalDB. Code and artifacts: ml/deployed_manifest.json, ml/audit_results/.",
     size=9, italic=True, color=MUTED, spacing=1.0)
footer(s, n)
notes(s, "Do not read references aloud. Zheng 2025 is the base paper; the datasets are public PhysioNet corpora.")

# ------------------------------------------------------------------ 16 thank you
s = blank(prs); n += 1
header(s)
s.shapes.add_picture(os.path.join(ASSETS, "syncura-mark.png"), Inches(6.36), Inches(1.68), width=Inches(0.60))
text(s, M, 2.5, CW, 0.9, "Thank You", size=46, bold=True, align=PP_ALIGN.CENTER, spacing=1.0)
text(s, M, 3.5, CW, 0.5, "Questions are welcome on the model, the evidence or the limits.",
     size=15, align=PP_ALIGN.CENTER, spacing=1.0, color=SOFT)
text(s, M, 4.7, CW, 0.9,
     "Abdul Ahad Ikkeri  \u00b7  Fathima Reeha  \u00b7  Fizan Feroz\nGuide: Divya K  \u00b7  Department of CSE, P.A. College of Engineering",
     size=13, align=PP_ALIGN.CENTER, spacing=1.3)
text(s, M, 6.2, CW, 0.3, "Research prototype \u2014 not a medical device, not HIPAA-ready.",
     size=10.5, italic=True, color=MUTED, align=PP_ALIGN.CENTER, spacing=1.0)
footer(s, n)
notes(s, "Close on the caveat and invite the hard question.")

prs.core_properties.title = "SynCura — Project Synopsis V4"
prs.core_properties.author = "Abdul Ahad Ikkeri, Fathima Reeha, Fizan Feroz"
prs.core_properties.subject = "Project synopsis V4 — results, limitations, future scope"
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
