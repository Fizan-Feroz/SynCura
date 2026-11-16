# -*- coding: utf-8 -*-
"""Build SynCura Synopsis V4 (15 slides).

Numbers come from the repo artifacts:
  ml/deployed_manifest.json                     ensemble-v2 (val 0.840 / holdout 0.834)
  ml/training_runs/level3_multiscale_hybrid/*   Level 3 hybrid (holdout 0.843)
  ml/audit_results/clinical_validation_summary  CI, AUPRC, Brier, ECE, SAPS-I baseline
  ml/audit_results/threshold_optimization_results  operating curve (16,944 windows)
"""
import math
import os

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_MARK
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ---------------------------------------------------------------- palette
INK      = RGBColor(0x0B, 0x23, 0x22)
INK_SOFT = RGBColor(0x2C, 0x44, 0x40)
MUTED    = RGBColor(0x5F, 0x7A, 0x73)
BRAND    = RGBColor(0x1C, 0x7C, 0x6A)
MINT     = RGBColor(0x43, 0xD6, 0xC4)
MINT_DIM = RGBColor(0x1F, 0x4E, 0x48)
LINE     = RGBColor(0xD9, 0xE6, 0xE1)
SURF     = RGBColor(0xF3, 0xF8, 0xF6)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
PALE     = RGBColor(0xE6, 0xF4, 0xF0)
CRIT     = RGBColor(0xC4, 0x22, 0x3F)
CRIT_BG  = RGBColor(0xFB, 0xE9, 0xEC)
WARN     = RGBColor(0xB7, 0x7A, 0x0A)
WARN_BG  = RGBColor(0xFA, 0xF1, 0xDF)

HEAD = "Cambria"
BODY = "Calibri"

SW, SH = 13.333, 7.5
M = 0.7                    # side margin
CW = SW - 2 * M            # content width

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SynCura_Synopsis_V4.pptx")


# ---------------------------------------------------------------- helpers
def rgb_hex(c):
    return "%02X%02X%02X" % (c[0], c[1], c[2])


def new_deck():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, fill=None, line=None, radius=None, line_w=0.75):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius is not None else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius is not None:
        shp.adjustments[0] = radius
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    return shp


def text(slide, x, y, w, h, value, size=13, bold=False, color=INK, font=BODY,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.05,
         italic=False, space_after=0, caps=False, spacing=None):
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
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = line
        f = r.font
        f.name = font
        f.size = Pt(size)
        f.bold = bold
        f.italic = italic
        f.color.rgb = color
        if caps:
            rPr = r._r.get_or_add_rPr()
            rPr.set("cap", "all")
        if spacing is not None:
            rPr = r._r.get_or_add_rPr()
            rPr.set("spc", str(int(spacing * 100)))
    return box


def bullets(slide, x, y, w, h, items, size=13, color=INK, gap=7, line_spacing=1.08,
            dot_color=BRAND, font=BODY):
    """items: list of str or (bold_lead, rest)."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = line_spacing
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(0.22 * 914400)))
        pPr.set("indent", str(int(-0.22 * 914400)))
        bu = pPr.makeelement(
            "{http://schemas.openxmlformats.org/drawingml/2006/main}buChar",
            {"char": "\u2022"},
        )
        pPr.append(bu)
        lead, rest = (item if isinstance(item, tuple) else (None, item))
        if lead:
            r1 = p.add_run()
            r1.text = lead
            r1.font.name = font
            r1.font.size = Pt(size)
            r1.font.bold = True
            r1.font.color.rgb = color
        r2 = p.add_run()
        r2.text = rest
        r2.font.name = font
        r2.font.size = Pt(size)
        r2.font.color.rgb = color
    return box


def kicker(slide, value, color=BRAND):
    text(slide, M, 0.42, 8.0, 0.3, value, size=10.5, bold=True, color=color,
         caps=True, spacing=1.4)


def heading(slide, value, size=30, color=INK, y=0.72):
    text(slide, M, y, CW, 0.62, value, size=size, bold=True, color=color, font=HEAD,
         line_spacing=0.95)


def footer(slide, n, dark=False):
    c = MUTED if not dark else RGBColor(0x7F, 0x9C, 0x95)
    text(slide, M, 7.05, 7.0, 0.25, "SynCura  ·  Synopsis V4  ·  November 2026",
         size=8.5, color=c)
    text(slide, SW - M - 1.0, 7.05, 1.0, 0.25, f"{n:02d}", size=8.5, color=c,
         align=PP_ALIGN.RIGHT)


def ecg(slide, x, y, w, h, color, thickness=1.25, baseline=0.55, opacity=1.0):
    n = 260
    pts = []
    for i in range(n):
        t = i / (n - 1)
        v = 0.0
        for c in (0.16, 0.40, 0.64, 0.86):
            d = abs(t - c)
            if d < 0.010:
                v += (1 - d / 0.010) * 1.0
            elif d < 0.026:
                v -= (1 - abs(d - 0.018) / 0.008) * 0.32
        v += 0.035 * math.sin(t * 34) + 0.02 * math.sin(t * 71)
        pts.append((x + t * w, y + h * baseline - v * h * 0.42))
    fb = slide.shapes.build_freeform(Emu(int(pts[0][0])), Emu(int(pts[0][1])))
    fb.add_line_segments([(Emu(int(px)), Emu(int(py))) for px, py in pts[1:]], close=False)
    shp = fb.convert_to_shape()
    shp.fill.background()
    shp.line.color.rgb = color
    shp.line.width = Pt(thickness)
    shp.shadow.inherit = False
    return shp


def stat_card(slide, x, y, w, h, value, label, sub=None, accent=BRAND, bg=SURF):
    rect(slide, x, y, w, h, fill=bg, line=LINE, radius=0.06)
    text(slide, x + 0.22, y + 0.13 * h, w - 0.44, 0.58, value, size=28, bold=True,
         color=accent, font=HEAD, line_spacing=0.9)
    text(slide, x + 0.22, y + 0.59 * h, w - 0.44, 0.28, label, size=9.5, bold=True,
         color=INK_SOFT, caps=True, spacing=0.8)
    if sub:
        text(slide, x + 0.22, y + 0.775 * h, w - 0.44, 0.26, sub, size=9, color=MUTED)


def pill(slide, x, y, w, h, value, fg, bg, size=9):
    rect(slide, x, y, w, h, fill=bg, line=None, radius=0.5)
    text(slide, x, y + (h - 0.18) / 2 - 0.01, w, 0.2, value, size=size, bold=True,
         color=fg, align=PP_ALIGN.CENTER, caps=True, spacing=0.6)


def notes(slide, value):
    slide.notes_slide.notes_text_frame.text = value


# ================================================================ deck
prs = new_deck()
page = 0


# ---------------------------------------------------------------- 1 title
s = blank(prs)
page += 1
rect(s, 0, 0, SW, SH, fill=INK)
ecg(s, 0, 0.30, SW, 0.62, MINT_DIM, thickness=1.1, baseline=0.55)

text(s, M + 0.2, 0.95, 9.0, 0.3, "PROJECT SYNOPSIS  ·  V4  ·  NOVEMBER 2026",
     size=11, bold=True, color=MINT, caps=True, spacing=2.2)
text(s, M + 0.2, 1.35, 10.0, 1.15, "SynCura", size=64, bold=True, color=WHITE, font=HEAD)
text(s, M + 0.2, 2.48, 10.6, 0.95,
     "Predictive ICU monitoring with a multi-scale Attention-LSTM + LightGBM hybrid,\n"
     "real-time risk scoring, and dual explainability",
     size=19, color=RGBColor(0xC9, 0xE4, 0xDD), line_spacing=1.15)

# proof strip
for i, (val, lab) in enumerate((("0.843", "holdout AUROC"), ("638", "holdout patients"), ("16,944", "windows scored"))):
    sx = M + 0.2 + i * 2.85
    text(s, sx, 3.52, 2.6, 0.45, val, size=22, bold=True, color=MINT, font=HEAD)
    text(s, sx, 3.98, 2.6, 0.25, lab, size=8.5, bold=True, color=RGBColor(0x7F, 0x9C, 0x95), caps=True, spacing=1.1)

text(s, M + 0.2, 4.55, 3.4, 0.24, "GUIDE", size=9, bold=True, color=MINT, caps=True, spacing=1.6)
text(s, M + 0.2, 4.78, 3.4, 0.3, "Divya K", size=13.5, bold=True, color=WHITE)

text(s, M + 3.9, 4.55, 5.6, 0.24, "TEAM", size=9, bold=True, color=MINT, caps=True, spacing=1.6)
text(s, M + 3.9, 4.78, 5.8, 0.9,
     "Abdul Ahad Ikkeri · 4PA24CS002\nFathima Reeha · 4PA24CS026\nFizan Feroz · 4PA24CS032",
     size=12.5, color=RGBColor(0xC9, 0xE4, 0xDD), line_spacing=1.28)

text(s, M + 0.2, 6.0, 8.6, 0.3,
     "Department of Computer Science & Engineering  ·  P.A. College of Engineering",
     size=11, color=RGBColor(0x8F, 0xB3, 0xAB))
text(s, M + 0.2, 6.55, 11.5, 0.3,
     "Research prototype — not a medical device, not HIPAA-ready, not prospectively validated.",
     size=9.5, italic=True, color=RGBColor(0x6F, 0x94, 0x8C))
s.shapes.add_picture(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "syncura-mark-mint.png"),
                     Inches(11.95), Inches(6.18), width=Inches(0.72))
notes(s, "Open with the one-liner: SynCura turns 90 minutes of ICU vitals into an explainable 0-100 risk score, "
         "and we ship the whole path - model, API, dashboard and alerts.")


# ---------------------------------------------------------------- 2 abstract
s = blank(prs)
page += 1
kicker(s, "Abstract")
heading(s, "A score clinicians can act on — and argue with")
footer(s, page)

bullets(s, M, 1.62, 7.15, 4.3, [
    ("Problem: ", "ICU deterioration is often visible hours before the crash, but NEWS2, SOFA and APACHE-II "
                  "score single timepoints and miss direction, duration and interaction."),
    ("Method: ", "12 vitals and labs over 90-minute windows, fused with 24-hour trend slopes and 116 engineered "
                 "features, then modelled by an Attention-LSTM + LightGBM hybrid."),
    ("Evidence: ", "locked 638-patient set-B holdout (16,944 windows, 0% patient leakage): 0.843 AUROC "
                   "(hybrid), 0.847 in the threshold-optimised run, versus 0.675 for SAPS-I."),
    ("Product: ", "FastAPI ingestion and scoring, React central station, KernelSHAP + temporal attention, "
                  "ranked beds, replay and demo sources, warning/critical sound cues."),
    ("Caveat: ", "retrospective, single-corpus evaluation with a calibration gap — a research prototype, "
                 "not a validated clinical device."),
], size=12.5, gap=10)

text(s, M, 5.95, 7.0, 0.4,
     "“Six fast signals catch the crash, six slow ones explain the cause.”",
     size=12.5, italic=True, color=BRAND)

stat_card(s, 8.25, 1.62, 4.4, 1.45, "0.843", "Holdout AUROC", "0.804 – 0.877 (95% CI) · SAPS-I 0.675", BRAND)
stat_card(s, 8.25, 3.20, 4.4, 1.45, "86.7%", "Peak accuracy", "LightGBM branch · 96.0% specificity", BRAND)
stat_card(s, 8.25, 4.78, 4.4, 1.45, "16,944", "Holdout windows", "638 unseen patients · 14.2% positive", INK_SOFT)
notes(s, "Five sentences: problem, method, evidence, product, caveat. Never claim clinical readiness.")


# ---------------------------------------------------------------- 3 intro
s = blank(prs)
page += 1
kicker(s, "Introduction")
heading(s, "Why static scores miss the patient who is drifting")
footer(s, page)

text(s, M, 1.6, 5.5, 0.3, "THE CLINICAL PROBLEM", size=10, bold=True, color=MUTED, caps=True, spacing=1.2)
bullets(s, M, 1.98, 5.55, 2.5, [
    "ICU mortality runs 10–29%; most deterioration is preceded by hours of measurable drift.",
    "NEWS2, SOFA and APACHE-II are computed by hand at rounds, from one moment in time.",
    "Thresholds are static, so gradual trends and interacting variables are invisible until late.",
    "The result is late escalation, alarm fatigue, and little explanation for the score.",
], size=12, gap=9)

text(s, 6.65, 1.6, 5.4, 0.3, "OBJECTIVES", size=10, bold=True, color=MUTED, caps=True, spacing=1.2)
objs = [
    ("O1", "Temporal mortality-risk model", "Done"),
    ("O2", "Full-stack real-time system", "Done"),
    ("O3", "Dual explainability (attention + SHAP)", "Done"),
    ("O4", "Operating-curve and baseline comparison", "Done"),
    ("O5", "Simulation, replay and alerting", "Done"),
]
y = 1.98
for code, label, status in objs:
    rect(s, 6.65, y, 5.95, 0.56, fill=SURF, line=LINE, radius=0.08)
    text(s, 6.85, y + 0.13, 0.45, 0.3, code, size=10.5, bold=True, color=BRAND)
    text(s, 7.35, y + 0.11, 3.9, 0.35, label, size=11.5, color=INK)
    pill(s, 11.35, y + 0.14, 1.02, 0.28, status, BRAND, PALE)
    y += 0.64

rect(s, M, 5.45, CW, 1.25, fill=INK, line=None, radius=0.05)
text(s, M + 0.35, 5.6, 2.2, 0.25, "DESIGN CONSTRAINT", size=9, bold=True, color=MINT, caps=True, spacing=1.4)
text(s, M + 0.35, 5.9, CW - 0.7, 0.7,
     "Every design choice was judged by one question: can a bedside clinician see why the number moved, "
     "and act on it in under a minute? That is why explanation and alerts are part of the model contract, "
     "not an afterthought.",
     size=12.5, color=RGBColor(0xC9, 0xE4, 0xDD), line_spacing=1.2)
notes(s, "Frame the gap in current practice, then map the five objectives. All five are demonstrable in the live system.")


# ---------------------------------------------------------------- 4 literature
s = blank(prs)
page += 1
kicker(s, "Literature survey")
heading(s, "Ten years of strong AUROCs, very few deployed wards")
footer(s, page)

rows = [
    ("Dynamic real-time ICU risk model", "Zheng 2025", "Vitals, labs, meds", "Retrospective; drops cross-hospital"),
    ("DL mortality on long ICU sequences", "Yan 2026", "ICD + temporal data", "AUC 0.802; stacking complexity"),
    ("Multimodal deterioration (notes)", "Sadanandan 2026", "Time series + BERT", "Privacy, compute, low AUPRC"),
    ("Vital-signs-only real-time risk", "Wu 2024", "BP, HR, RR, temp", "8× more data; never deployed"),
    ("Attention-LSTM risk (our ancestor)", "Nguyen 2017", "PhysioNet 2012", "Legacy; limited external validity"),
    ("Generative real-time imputation", "Xie 2025", "Recovered missing vitals", "Needs multi-hospital data"),
    ("DEWS interpretable early warning", "Choi 2020", "Vitals + attention", "Attention is not causation"),
    ("Attention residual LSTM-FCN", "Li 2025", "Clinical time series", "Complex; limited external validation"),
    ("Expert-augmented early warning", "Wang 2026", "Rules + ML", "Rule maintenance; single health system"),
    ("Wearable deterioration prediction", "Scheid 2025", "Wearable vitals", "Sensor noise and dropouts"),
]

tbl_shape = s.shapes.add_table(len(rows) + 1, 4, Inches(M), Inches(1.55), Inches(CW), Inches(4.2))
tbl = tbl_shape.table
tbl.first_row = True
tbl.horz_banding = False
for i, wd in enumerate((4.35, 1.75, 2.75, 3.08)):
    tbl.columns[i].width = Inches(wd)

hdr = ("Study", "Authors · Year", "Inputs", "What it leaves open")
for j, htxt in enumerate(hdr):
    cell = tbl.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = INK
    cell.margin_left = Inches(0.12)
    cell.margin_right = Inches(0.08)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = cell.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = htxt
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.name = BODY
    r.font.color.rgb = WHITE
tbl.rows[0].height = Inches(0.36)

for i, row in enumerate(rows, start=1):
    tbl.rows[i].height = Inches(0.38)
    for j, val in enumerate(row):
        cell = tbl.cell(i, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if i % 2 else SURF
        cell.margin_left = Inches(0.12)
        cell.margin_right = Inches(0.08)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        r = p.add_run()
        r.text = val
        r.font.size = Pt(9)
        r.font.name = BODY
        r.font.color.rgb = INK if j < 3 else INK_SOFT
        if j == 0:
            r.font.bold = True

rect(s, M, 6.0, CW, 0.78, fill=PALE, line=None, radius=0.06)
text(s, M + 0.28, 6.14, 0.78, 0.24, "THE GAP", size=9, bold=True, color=BRAND, caps=True, spacing=1.4)
text(s, M + 1.05, 6.12, CW - 1.35, 0.55,
     "The literature optimises AUROC and stops there. Almost none of it reports calibration, alarm burden, "
     "or a working bedside path — which is exactly what this project builds and measures.",
     size=11.5, color=INK_SOFT)
notes(s, "Name the base paper (Zheng 2025) and DEWS (Choi 2020). Point at the last column: deployment and "
         "explanation are the consistent gap.")


# ---------------------------------------------------------------- 5 problem
s = blank(prs)
page += 1
kicker(s, "Problem statement")
heading(s, "The number arrives late, and nobody can explain it")
footer(s, page)

cards = [
    ("01", "Static thresholds", "A single NEWS2 or SOFA cut-off cannot see a patient sliding from 96% to 90% SpO2 over two hours."),
    ("02", "Trajectory blindness", "Hand-computed scores at rounds ignore direction, duration and the interaction between vitals."),
    ("03", "Unexplained alarms", "A bare risk number gives no cause, no timepoint and no next step, so alarms get silenced."),
]
x = M
for code, title, body in cards:
    rect(s, x, 1.62, 3.78, 1.7, fill=WHITE, line=LINE, radius=0.05)
    text(s, x + 0.25, 1.8, 0.7, 0.3, code, size=10, bold=True, color=MINT)
    text(s, x + 0.25, 2.06, 3.3, 0.35, title, size=15, bold=True, color=INK, font=HEAD)
    text(s, x + 0.25, 2.48, 3.3, 0.75, body, size=11, color=INK_SOFT, line_spacing=1.15)
    x += 3.98

text(s, M, 3.75, 6.0, 0.3, "WHAT A USABLE SYSTEM MUST DO", size=10, bold=True, color=MUTED, caps=True, spacing=1.2)
reqs = [
    ("Learn time, not just level", "Sequence model over 90-minute windows, with 24-hour trend context."),
    ("Show its reasons", "Temporal attention for when, KernelSHAP for what, and a bedside action for each alert."),
    ("Survive contact with a ward", "Real-time API, a dashboard that ranks beds, and alerts that escalate without spam."),
]
x = M
for title, body in reqs:
    rect(s, x, 4.12, 3.78, 1.45, fill=SURF, line=None, radius=0.05)
    text(s, x + 0.22, 4.28, 3.3, 0.3, title, size=12, bold=True, color=INK)
    text(s, x + 0.22, 4.63, 3.3, 0.75, body, size=10.5, color=INK_SOFT, line_spacing=1.12)
    x += 3.98

rect(s, M, 5.95, CW, 0.62, fill=INK, line=None, radius=0.05)
text(s, M + 0.28, 6.07, CW - 0.56, 0.4,
     "Scope: in-hospital mortality risk for ICU stays — not sepsis onset, not diagnosis, not an autonomous decision.",
     size=11, color=WHITE)
notes(s, "Three failure modes, three requirements. Close on the scope sentence so nobody assumes a sepsis device.")


# ---------------------------------------------------------------- 6 methodology
s = blank(prs)
page += 1
kicker(s, "Methodology")
heading(s, "A two-branch hybrid over acute and macro time scales")
footer(s, page)

stages = [
    ("01", "Corpus", "PhysioNet 2012\nSet A/B/C · 12,000 stays"),
    ("02", "Windows", "90-min acute window\nstride 15 · causal fill"),
    ("03", "Branches", "Attention-LSTM +\nLightGBM · 116 features"),
    ("04", "Blend", "Meta-stack →\nrisk 0–100"),
    ("05", "Serve", "FastAPI → dashboard\nexplain + alert"),
]
x = M
for code, title, body in stages:
    rect(s, x, 1.62, 2.12, 1.55, fill=WHITE, line=LINE, radius=0.06)
    text(s, x + 0.18, 1.74, 0.6, 0.24, code, size=9, bold=True, color=MINT)
    text(s, x + 0.18, 1.98, 1.8, 0.3, title, size=14, bold=True, color=INK, font=HEAD)
    text(s, x + 0.18, 2.34, 1.8, 0.7, body, size=9.5, color=INK_SOFT, line_spacing=1.12)
    if code != "05":
        text(s, x + 2.14, 2.18, 0.25, 0.3, "→", size=15, color=MINT)
    x += 2.38

# branch detail
rect(s, M, 3.45, 5.8, 1.6, fill=SURF, line=None, radius=0.05)
text(s, M + 0.25, 3.6, 5.2, 0.28, "SEQUENCE BRANCH — WHEN IT CHANGED", size=9, bold=True, color=BRAND, caps=True, spacing=1.2)
bullets(s, M + 0.25, 3.92, 5.3, 1.0, [
    "2-layer LSTM (hidden 96) with additive Bahdanau attention",
    "Context MLP: static demographics + 24-hour polynomial slopes",
    "Class-weighted BCE; patient-level split, no window leakage",
], size=10, gap=5)

rect(s, 6.85, 3.45, 5.78, 1.6, fill=SURF, line=None, radius=0.05)
text(s, 7.1, 3.6, 5.2, 0.28, "TABULAR BRANCH — WHAT IT LOOKS LIKE NOW", size=9, bold=True, color=BRAND, caps=True, spacing=1.2)
bullets(s, 7.1, 3.92, 5.3, 1.0, [
    "LightGBM GBDT on 116 engineered features (moments, volatility, shock indices)",
    "Captures non-linear lab thresholds the LSTM smooths over",
    "Meta-stack blends branch probabilities; α tuned on validation",
], size=10, gap=5)

rect(s, M, 5.42, CW, 0.7, fill=INK, line=None, radius=0.05)
text(s, M + 0.28, 5.55, 1.05, 0.25, "SERVING PATH", size=9, bold=True, color=MINT, caps=True, spacing=1.4)
text(s, M + 1.38, 5.53, CW - 1.68, 0.5,
     "vitals → POST /ingest → SQLite → ensemble scorer (0–100) → KernelSHAP + attention → "
     "React station, bed profile, alerts, sound cues",
     size=10.5, color=RGBColor(0xC9, 0xE4, 0xDD))
notes(s, "Walk the five stages left to right, then explain the two branches: sequence captures trajectory, "
         "LightGBM captures threshold effects, the blend captures both.")


# ---------------------------------------------------------------- 7 data
s = blank(prs)
page += 1
kicker(s, "Data scale")
heading(s, "Three corpora on disk, one honest training scope")
footer(s, page)

data_rows = [
    ("PhysioNet 2012 (A/B/C)", "12,000 stays", "Training + locked holdout", "In scope"),
    ("PhysioNet 2019 (A + B)", "40,336 stays", "Warm-start / scale-up experiments", "In scope"),
    ("VitalDB", "6,388 cases", "Loader + external code-path check", "Not in L3 training"),
    ("SICdb", "access requested", "Pending DUA approval", "Pending"),
]
tbl_shape = s.shapes.add_table(len(data_rows) + 1, 4, Inches(M), Inches(1.55), Inches(CW), Inches(2.4))
tbl = tbl_shape.table
tbl.first_row = True
tbl.horz_banding = False
for i, wd in enumerate((3.45, 2.15, 4.33, 1.85)):
    tbl.columns[i].width = Inches(wd)

for j, htxt in enumerate(("Corpus", "Stays / cases", "Role in the pipeline", "Status")):
    cell = tbl.cell(0, j)
    cell.fill.solid(); cell.fill.fore_color.rgb = INK
    cell.margin_left = Inches(0.12); cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = cell.text_frame.paragraphs[0].add_run(); r.text = htxt
    r.font.size = Pt(9.5); r.font.bold = True; r.font.name = BODY; r.font.color.rgb = WHITE
tbl.rows[0].height = Inches(0.38)

for i, row in enumerate(data_rows, start=1):
    tbl.rows[i].height = Inches(0.52)
    for j, val in enumerate(row):
        cell = tbl.cell(i, j)
        cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if i % 2 else SURF
        cell.margin_left = Inches(0.12); cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        r = cell.text_frame.paragraphs[0].add_run(); r.text = val
        r.font.size = Pt(9.5); r.font.name = BODY
        r.font.color.rgb = INK if j == 0 else INK_SOFT
        if j == 0: r.font.bold = True

stat_card(s, M, 4.4, 3.78, 1.45, "9,802", "Training patients", "Set A + 80% Set B + Set C", BRAND)
stat_card(s, 4.72, 4.4, 3.78, 1.45, "638", "Locked holdout patients", "20% set-B, never trained on", BRAND)
stat_card(s, 8.83, 4.4, 3.8, 1.45, "0%", "Patient overlap", "Group split; no window leakage", INK_SOFT)

text(s, M, 6.15, CW, 0.6,
     "VitalDB and SICdb are wired at the loader level but deliberately excluded from the reported model: "
     "mixing mortality and sepsis labels would inflate the headline without a matched target.",
     size=10, italic=True, color=MUTED)
notes(s, "Be explicit: the headline model is trained on PhysioNet 2012 only. 2019 contributes a warm-start to one "
         "ensemble member; VitalDB is a code-path check. This is the scope we can defend.")


# ---------------------------------------------------------------- 8 results
s = blank(prs)
page += 1
kicker(s, "Results")
heading(s, "The operating curve, not a single accuracy number")
footer(s, page)

stat_card(s, M, 1.52, 2.8, 1.2, "0.843", "Holdout AUROC", "0.804 – 0.877 (95% CI)", BRAND)
stat_card(s, M, 2.84, 2.8, 1.2, "0.469", "Holdout AUPRC", "prevalence 14.2%", BRAND)
stat_card(s, M, 4.16, 2.8, 1.2, "0.193", "Calibration ECE", "Brier 0.152 — the honest gap", CRIT)
stat_card(s, M, 5.48, 2.8, 1.2, "0.675", "SAPS-I AUROC", "same holdout baseline", MUTED)

chart_data = CategoryChartData()
chart_data.categories = ["0.40", "0.50", "0.60", "0.65", "0.70", "0.75", "0.80"]
chart_data.add_series("Accuracy", (0.7116, 0.7658, 0.8007, 0.8229, 0.8433, 0.8579, 0.8665))
chart_data.add_series("Sensitivity", (0.8550, 0.7825, 0.6725, 0.6154, 0.5392, 0.4312, 0.3025))
chart_data.add_series("Specificity", (0.6879, 0.7630, 0.8219, 0.8571, 0.8935, 0.9284, 0.9596))

gframe = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(3.75), Inches(1.5),
                            Inches(8.85), Inches(4.35), chart_data)
chart = gframe.chart
chart.font.size = Pt(10)
chart.font.name = BODY
chart.font.color.rgb = MUTED
chart.has_title = True
chart.chart_title.text_frame.text = "Decision threshold vs. operating characteristics (16,944 holdout windows)"
p = chart.chart_title.text_frame.paragraphs[0]
p.runs[0].font.size = Pt(10.5)
p.runs[0].font.bold = True
p.runs[0].font.color.rgb = INK_SOFT
chart.has_legend = True
chart.legend.position = XL_LEGEND_POSITION.BOTTOM
chart.legend.include_in_layout = False
chart.legend.font.size = Pt(9.5)

plot = chart.plots[0]
plot.has_data_labels = False
colors = [BRAND, CRIT, MUTED]
for series, col in zip(plot.series, colors):
    series.smooth = False
    series.format.line.color.rgb = col
    series.format.line.width = Pt(2.0)
    series.marker_size = 5
    series.marker_style = XL_MARKER_STYLE.CIRCLE

va = chart.value_axis
va.minimum_scale = 0.2
va.maximum_scale = 1.0
va.major_unit = 0.2
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = LINE
va.major_gridlines.format.line.width = Pt(0.75)
va.tick_labels.font.size = Pt(9)
va.format.line.color.rgb = LINE

ca = chart.category_axis
ca.tick_labels.font.size = Pt(9)
ca.format.line.color.rgb = LINE
ca.has_major_gridlines = False
ca.major_tick_mark = XL_TICK_MARK.OUTSIDE

text(s, 3.78, 5.98, 8.8, 0.8,
     "Source: ml/audit_results/threshold_optimization_results.json — 638-patient locked set-B holdout, "
     "16,944 windows, AUROC 0.847 for this run (hybrid blend 0.843). "
     "At 0.50: 76.6% accuracy and 78.3% recall; at 0.80: 86.7% accuracy and 96.0% specificity.",
     size=9, italic=True, color=MUTED, line_spacing=1.15)
notes(s, "Lead with AUROC and its confidence interval, then the curve. If asked why accuracy looks low at 0.5: "
         "14% prevalence means a majority-class model scores 85.8% with zero recall. The threshold is a clinical choice.")


# ---------------------------------------------------------------- 9 product
s = blank(prs)
page += 1
kicker(s, "System implementation")
heading(s, "What is actually running, not just trained")
footer(s, page)

feats = [
    ("Central station", "Beds ranked by risk with 5-second re-ranking, risk rings, SpO2/HR/RR/temp readouts and trend traces."),
    ("Bed profile", "Risk trend, NEWS2 for the bed, an active alert with cause and next action, and a simplified driver view."),
    ("Dual explainability", "Temporal attention shows when; KernelSHAP attribution shows what; the API returns both plus a bedside action."),
    ("Alerting with sound", "Warning and critical cues, 5-second critical siren, red pulse on the bed, per-user mute."),
]
x, y = M, 1.6
for i, (title, body) in enumerate(feats):
    cx = M + (i % 2) * 6.0
    cy = 1.55 + (i // 2) * 1.4
    rect(s, cx, cy, 5.85, 1.28, fill=WHITE, line=LINE, radius=0.05)
    rect(s, cx + 0.22, cy + 0.22, 0.42, 0.42, fill=PALE, line=None, radius=0.5)
    text(s, cx + 0.22, cy + 0.29, 0.42, 0.3, f"{i+1}", size=11, bold=True, color=BRAND, align=PP_ALIGN.CENTER)
    text(s, cx + 0.8, cy + 0.18, 4.8, 0.3, title, size=13.5, bold=True, color=INK, font=HEAD)
    text(s, cx + 0.8, cy + 0.53, 4.85, 0.7, body, size=10.5, color=INK_SOFT, line_spacing=1.1)

rect(s, M, 4.55, CW, 0.95, fill=SURF, line=None, radius=0.05)
stack = [("18", "API endpoints"), ("3", "data sources"), ("5", "scenarios"), ("5", "test modules"), ("1.12.1", "web release")]
x = M + 0.35
for val, lab in stack:
    text(s, x, 4.69, 1.8, 0.4, val, size=17, bold=True, color=BRAND, font=HEAD)
    text(s, x, 5.10, 1.9, 0.25, lab, size=8.5, color=MUTED, caps=True, spacing=0.8)
    x += 2.3

text(s, M, 5.75, CW, 0.35,
     "Live, replay and local demo share one simulation contract; the dashboard never presents demo beds as live patients.",
     size=10, italic=True, color=MUTED)
text(s, M, 6.35, CW, 0.35,
     "Stack: React 18 · Vite · FastAPI · SQLite · PyTorch · Render · Vercel",
     size=9.5, color=MUTED)
notes(s, "Demo this live if possible: central station, open a bed, point at the alert card and the sound toggle. "
         "The point is the path from score to action.")


# ---------------------------------------------------------------- 10 milestones
s = blank(prs)
page += 1
kicker(s, "Work done")
heading(s, "From a leaky 0.844 to a defensible, deployed 0.834")
footer(s, page)

# timeline
ty = 2.55
rect(s, M + 0.15, ty, 10.15, 0.03, fill=LINE, line=None)
milestones = [
    ("SEP 2026", "Leakage found", "v1 pipeline retired; causal interpolation rebuilt"),
    ("SEP 2026", "Ensemble v2", "e12+e13+c53 deployed: val 0.840, holdout 0.834"),
    ("OCT 2026", "Level 3 hybrid", "Attention-LSTM + LightGBM; holdout 0.843"),
    ("OCT 2026", "Clinical validation", "95% CI, AUPRC, Brier, ECE, SAPS-I baseline"),
    ("NOV 2026", "Product release", "v1.12.1, paper/monitor themes, version page"),
    ("NOV 2026", "Sound alerts", "warning + critical cues, 5s siren, red pulse, mute"),
]
x = M + 0.15
for i, (date, title, body) in enumerate(milestones):
    cx = M + i * 2.0
    rect(s, cx + 0.05, ty - 0.07, 0.17, 0.17, fill=BRAND if i < 4 else MINT, line=None, radius=0.5)
    text(s, cx, 2.05, 1.9, 0.22, date, size=8.5, bold=True, color=MUTED, caps=True, spacing=0.8)
    text(s, cx, 2.75, 1.95, 0.3, title, size=11.5, bold=True, color=INK)
    text(s, cx, 3.08, 1.9, 0.95, body, size=9, color=INK_SOFT, line_spacing=1.1)

# achievement list
rect(s, M, 4.12, 5.9, 1.9, fill=SURF, line=None, radius=0.05)
text(s, M + 0.28, 4.26, 5.3, 0.28, "MODEL & VALIDATION", size=9, bold=True, color=BRAND, caps=True, spacing=1.2)
bullets(s, M + 0.28, 4.58, 5.35, 1.35, [
    "Fixed a temporal leak; every reported number now uses causal windows",
    "Locked 638-patient set-B holdout with 0% patient overlap",
    "Reported calibration and AUPRC, not just AUROC",
    "Benchmarked against SAPS-I on the same holdout (+0.167 AUROC)",
    "Level 3 hybrid evaluated on that holdout: 0.843 AUROC",
], size=9.5, gap=4)

rect(s, 6.9, 4.12, 5.73, 1.9, fill=SURF, line=None, radius=0.05)
text(s, 7.15, 4.26, 5.2, 0.28, "ENGINEERING", size=9, bold=True, color=BRAND, caps=True, spacing=1.2)
bullets(s, 7.15, 4.58, 5.2, 1.35, [
    "Causal + BatchNorm inference contract shared by train and serve",
    "Explain endpoint fixed to read latest vitals from SQLite",
    "Accessible paper/monitor themes; reduced-motion paths",
    "Sound alerts, demo guards, CI tests on every push",
    "Live on Render + Vercel; /version reports the served model",
], size=9.5, gap=4)
notes(s, "Two stories: the leak we found and fixed (credibility), and the shipping work that makes the score usable.")


# ---------------------------------------------------------------- 11 limitations I
s = blank(prs)
page += 1
kicker(s, "Limitations — part 1", CRIT)
heading(s, "Where the evidence stops")
footer(s, page)

lims = [
    ("Single-corpus evidence", "The reported holdout is PhysioNet 2012 set-B only. No external hospital has scored this model."),
    ("Small, imbalanced holdout", "638 patients, 14.2% positive; AUROC CI spans 0.804–0.877, so the headline has real uncertainty."),
    ("Selection touched the holdout", "Ensemble members were chosen with val-gating and the fresh holdout was consulted during selection — it is not a sealed test set."),
    ("Mortality ≠ sepsis", "The label is in-hospital death. Challenge-2019 sepsis labels were never pooled; the 2019 schema also lacks GCS."),
    ("Poor calibration", "ECE 0.193 and Brier 0.152: probabilities are not yet reliable enough for absolute risk communication."),
    ("Level 3 is offline", "The hybrid is trained and evaluated, but production serving still runs the ensemble-v2 LSTM trio."),
]
x, y = M, 1.6
for i, (title, body) in enumerate(lims):
    cx = M + (i % 2) * 6.0
    cy = 1.62 + (i // 2) * 1.72
    rect(s, cx, cy, 5.85, 1.22, fill=CRIT_BG if i in (0, 4) else WHITE, line=LINE, radius=0.05)
    rect(s, cx + 0.22, cy + 0.24, 0.12, 0.12, fill=CRIT, line=None, radius=0.5)
    text(s, cx + 0.46, cy + 0.16, 5.1, 0.3, title, size=12, bold=True, color=INK)
    text(s, cx + 0.46, cy + 0.48, 5.15, 0.68, body, size=9.5, color=INK_SOFT, line_spacing=1.1)

text(s, M, 6.72, CW, 0.3,
     "These are the first questions an examiner should ask — and the reason every claim in this deck is scoped to a research prototype.",
     size=9.5, italic=True, color=MUTED)
notes(s, "Do not hide the calibration error. Owning ECE 0.193 and the holdout-selection issue is stronger than being caught on them.")


# ---------------------------------------------------------------- 12 limitations II
s = blank(prs)
page += 1
kicker(s, "Limitations — part 2", CRIT)
heading(s, "Where the product stops")
footer(s, page)

lims2 = [
    ("Not a medical device", "No regulatory clearance, no prospective trial, no IRB protocol, no clinical sign-off. Research use only."),
    ("Not HIPAA-ready", "No authentication or audit trail on patient data; the training UI is hidden by build flag, not secured."),
    ("Alarm burden remains", "Precision is ~35% at the safety threshold; warning cues can repeat every 1.2 s across escalating beds."),
    ("Explanation is correlational", "Attention and SHAP show what the model used, not what caused the deterioration."),
    ("Narrow input surface", "12 features, SpO2 proxied by SaO2, no notes, imaging or medication data; missingness is imputed, not modelled."),
    ("Single-node operations", "SQLite and one API instance; sound preference is per browser; no acknowledgement or escalation workflow."),
]
for i, (title, body) in enumerate(lims2):
    cx = M + (i % 2) * 6.0
    cy = 1.62 + (i // 2) * 1.72
    rect(s, cx, cy, 5.85, 1.22, fill=WARN_BG if i in (0, 2) else WHITE, line=LINE, radius=0.05)
    rect(s, cx + 0.22, cy + 0.24, 0.12, 0.12, fill=WARN, line=None, radius=0.5)
    text(s, cx + 0.46, cy + 0.16, 5.1, 0.3, title, size=12, bold=True, color=INK)
    text(s, cx + 0.46, cy + 0.48, 5.15, 0.68, body, size=9.5, color=INK_SOFT, line_spacing=1.1)

text(s, M, 6.72, CW, 0.3,
     "Everything above is deliberate scope, not an accident — and each item maps to a roadmap phase on the next slide.",
     size=9.5, italic=True, color=MUTED)
notes(s, "Product limitations, not model limitations. If asked 'can we use it tonight?', the answer is no, and here is why.")


# ---------------------------------------------------------------- 13 roadmap
s = blank(prs)
page += 1
kicker(s, "Future roadmap")
heading(s, "What closes each gap")
footer(s, page)

phases = [
    ("0–6 MONTHS", BRAND, [
        ("Calibrate + report", "Temperature scaling / isotonic, decision-curve analysis, ECE target < 0.05."),
        ("Promote Level 3", "Serve the hybrid behind a feature flag with a latency budget and A/B against ensemble-v2."),
        ("Seal a test set", "Lock the holdout before any further selection; report once."),
    ]),
    ("6–12 MONTHS", WARN, [
        ("External validation", "MIMIC-IV / eICU under credentialing; subgroup and fairness analysis."),
        ("Alarm workflow", "Acknowledge, escalate, measure fatigue; tune persistence k and cue cadence."),
        ("Multimodal", "Clinical notes with BioBERT alongside vitals and labs."),
    ]),
    ("12+ MONTHS", MUTED, [
        ("Prospective study", "IRB-approved silent trial, then clinician-in-the-loop evaluation."),
        ("Regulatory path", "Software-as-a-medical-device documentation and clinical evidence."),
        ("Edge + federated", "TinyML gateway and federated training across hospitals."),
    ]),
]
x = M
for label, col, items in phases:
    rect(s, x, 1.6, 3.85, 4.3, fill=WHITE, line=LINE, radius=0.05)
    text(s, x + 0.25, 1.78, 3.3, 0.25, label, size=9, bold=True, color=col, caps=True, spacing=1.2)
    yy = 2.12
    for title, body in items:
        text(s, x + 0.25, yy, 3.35, 0.3, title, size=11.5, bold=True, color=INK)
        text(s, x + 0.25, yy + 0.32, 3.35, 0.7, body, size=9.5, color=INK_SOFT, line_spacing=1.1)
        yy += 1.22
    x += 3.99

text(s, M, 6.2, CW, 0.35,
     "Ordered by what unlocks the next claim: calibration and a sealed test set come before any new model.",
     size=10, italic=True, color=MUTED)
notes(s, "Roadmap is sequenced by evidence, not by novelty. Calibration first, because a risk number nobody can trust is noise.")


# ---------------------------------------------------------------- 14 conclusion
s = blank(prs)
page += 1
kicker(s, "Conclusion")
heading(s, "What we proved, and what we still owe")
footer(s, page)

bullets(s, M, 1.62, 7.3, 3.6, [
    ("Outcome: ", "a working, explainable ICU risk prototype at 0.843 holdout AUROC, against 0.675 for SAPS-I on the same 638 patients."),
    ("System: ", "ingestion API, ensemble scorer, KernelSHAP and attention explanations, dashboard, alerts and audible cues - all running."),
    ("Evidence: ", "a locked split, confidence intervals, AUPRC, calibration, an operating curve and a documented limitations list."),
    ("Honesty: ", "the model is retrospective, single-corpus and uncalibrated; it is not a medical device and must not guide care."),
], size=12.5, gap=9)

stat_card(s, 8.55, 1.62, 4.1, 1.15, "0.843", "holdout AUROC", "hybrid · 16,944 windows")
stat_card(s, 8.55, 2.9, 4.1, 1.15, "638", "holdout patients", "0 patient overlap")
stat_card(s, 8.55, 4.18, 4.1, 1.15, "0.193", "calibration ECE", "the gap we still owe", CRIT)

rect(s, M, 5.6, CW, 1.1, fill=INK, line=None, radius=0.05)
text(s, M + 0.35, 5.74, 2.4, 0.25, "THE ONE-LINE VERDICT", size=9, bold=True, color=MINT, caps=True, spacing=1.4)
text(s, M + 0.35, 6.0, CW - 0.7, 0.6,
     "The papers stop at AUROC; we shipped the path from score to explanation to bedside action - and we can say exactly "
     "where that path is still unproven.",
     size=12.5, color=RGBColor(0xC9, 0xE4, 0xDD), line_spacing=1.2)
notes(s, "Close with the trade: shipped system and honest evidence, not a clinical claim.")

# ---------------------------------------------------------------- 14 references
s = blank(prs)
page += 1
kicker(s, "Appendix")
heading(s, "References")
footer(s, page)

refs_left = [
    "[1] Z. Zheng et al., “Dynamic real-time risk prediction for ICU patients,” J. Med. Internet Res., 2025. (Base paper)",
    "[2] Z. Yan et al., “Deep learning in-hospital mortality using long ICU sequences,” PeerJ, 2026.",
    "[3] B. Sadanandan, “Multimodal ICU deterioration with clinical notes,” arXiv, 2026.",
    "[4] Y. Wu et al., “Vital signs for real-time ICU mortality risk,” J. Big Data, 2024.",
    "[5] M. Nguyen et al., “Deep learning to attend to risk in ICU,” 2017.",
]
refs_right = [
    "[6] P. Xie et al., “RealMIP: real-time risk with continuous data recovery,” npj Digit. Med., 2025.",
    "[7] E. Choi et al., “Deep interpretable early warning (DEWS),” IEEE JBHI, 2020.",
    "[8] Y. Li et al., “Attention residual LSTM-FCN,” IEEE Access, 2025.",
    "[9] PhysioNet/Computing in Cardiology Challenge 2012, “Predicting mortality of ICU patients.”",
    "[10] M. R. Scheid et al., “Wearable deep learning for continuous deterioration,” Nature Commun., 2025.",
]
yy = 1.65
for r in refs_left:
    text(s, M, yy, 5.65, 0.7, r, size=10.5, color=INK_SOFT, line_spacing=1.12)
    yy += 0.82
yy = 1.65
for r in refs_right:
    text(s, 6.7, yy, 5.9, 0.7, r, size=10.5, color=INK_SOFT, line_spacing=1.12)
    yy += 0.82

text(s, M, 6.35, CW, 0.3,
     "Datasets: PhysioNet 2012 (A/B/C), PhysioNet 2019 (A/B), VitalDB. Artifacts: ml/deployed_manifest.json, ml/audit_results/.",
     size=9, italic=True, color=MUTED)
notes(s, "Do not read references aloud. Say: ten sources, Zheng 2025 is the base paper, and the datasets are public PhysioNet corpora.")


# ---------------------------------------------------------------- 15 thank you
s = blank(prs)
page += 1
rect(s, 0, 0, SW, SH, fill=INK)
ecg(s, 0, 6.88, SW, 0.5, MINT_DIM, thickness=1.0, baseline=0.5)
text(s, M + 0.2, 2.15, 9.0, 0.3, "SYNCURA  ·  SYNOPSIS V4", size=10.5, bold=True, color=MINT, caps=True, spacing=2.2)
text(s, M + 0.2, 2.5, 9.5, 1.1, "Thank you", size=54, bold=True, color=WHITE, font=HEAD)
text(s, M + 0.2, 3.75, 10.5, 0.7,
     "Questions on the model, the evidence, or the limits — all three are fair game.",
     size=16, color=RGBColor(0xC9, 0xE4, 0xDD))
text(s, M + 0.2, 5.85, 11.0, 0.3,
     "Team: Abdul Ahad Ikkeri · Fathima Reeha · Fizan Feroz      |      Guide: Divya K",
     size=11, color=RGBColor(0x8F, 0xB3, 0xAB))
text(s, M + 0.2, 6.55, 11.5, 0.3,
     "Research prototype — not a medical device, not HIPAA-ready.",
     size=9, italic=True, color=RGBColor(0x6F, 0x94, 0x8C))
s.shapes.add_picture(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "syncura-mark-mint.png"),
                     Inches(11.95), Inches(6.18), width=Inches(0.72))
notes(s, "Close on the caveat. Invite the hard question; it is the strongest signal that the work is honest.")

prs.core_properties.title = "SynCura — Project Synopsis V4"
prs.core_properties.author = "Abdul Ahad Ikkeri, Fathima Reeha, Fizan Feroz"
prs.core_properties.subject = "Predictive ICU monitoring — Attention-LSTM + LightGBM hybrid, results and limitations"
prs.core_properties.keywords = "ICU, mortality, Attention-LSTM, LightGBM, SHAP, FastAPI, React"
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
