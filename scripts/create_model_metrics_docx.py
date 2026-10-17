"""Generate a professional Word document (.docx) containing the final model information and metrics.

Saves to: PROJ/docs/SynCura_Final_Model_Metrics.docx
"""
import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, 'docs')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, 'SynCura_Final_Model_Metrics.docx')

# Clinical Color Palette
NAVY = RGBColor(27, 54, 93)       # #1B365D - Primary headers
TEAL = RGBColor(0, 128, 128)      # #008080 - Accent
SLATE = RGBColor(92, 118, 141)    # #5C768D - Subtitles/secondary
DARK = RGBColor(33, 37, 41)       # #212529 - Body text
WHITE = RGBColor(255, 255, 255)
HEX_HEADER_BG = "1B365D"
HEX_ROW_ALT = "F8FAFC"
HEX_BORDER = "CBD5E1"


def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_heading_with_spacing(doc, text, level=1, space_before=14, space_after=6):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(space_before)
    h.paragraph_format.space_after = Pt(space_after)
    h.paragraph_format.keep_with_next = True
    for run in h.runs:
        run.font.name = 'Arial'
        if level == 1:
            run.font.color.rgb = NAVY
            run.font.size = Pt(16)
            run.bold = True
        elif level == 2:
            run.font.color.rgb = SLATE
            run.font.size = Pt(13)
            run.bold = True
    return h


def create_document():
    doc = Document()

    # Page setup: US Letter, 0.75-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Base style adjustments
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(10.5)
    font.color.rgb = DARK

    # ==========================================
    # Title & Header
    # ==========================================
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    t_run = title_p.add_run("SynCura — Predictive ICU Monitoring System")
    t_run.font.name = 'Arial'
    t_run.font.size = Pt(22)
    t_run.font.bold = True
    t_run.font.color.rgb = NAVY

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(12)
    sub_run = sub_p.add_run("Final Model Performance, Architecture & Validation Report")
    sub_run.font.name = 'Arial'
    sub_run.font.size = Pt(13)
    sub_run.font.color.rgb = SLATE

    # Metadata Banner Table
    meta_table = doc.add_table(rows=1, cols=3)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    col_widths = [Inches(2.3), Inches(2.3), Inches(2.4)]
    meta_data = [
        ("Final Architecture", "Level 3 Hybrid Ensemble\n(Neural Multi-Scale + LightGBM)"),
        ("Cohort Scope", "PhysioNet 2012 (Sets A+B+C)\nChallenge 2019 (Sets A+B Pretrain)"),
        ("Target Clinical Label", "In-Hospital Mortality Risk\nCausal 90-Min Proximity Windows")
    ]
    
    for i, (k, v) in enumerate(meta_data):
        cell = meta_table.cell(0, i)
        cell.width = col_widths[i]
        set_cell_background(cell, "F1F5F9")
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r_k = p.add_run(f"{k}\n")
        r_k.bold = True
        r_k.font.size = Pt(9.5)
        r_k.font.color.rgb = NAVY
        r_v = p.add_run(v)
        r_v.font.size = Pt(9)
        r_v.font.color.rgb = DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ==========================================
    # 1. Executive Summary & Final Model Definition
    # ==========================================
    add_heading_with_spacing(doc, "1. Final Model Architecture", level=1)
    
    p_desc = doc.add_paragraph()
    p_desc.paragraph_format.space_after = Pt(6)
    p_desc.add_run(
        "The final deployed model is the Level 3 Hybrid Clinical Classifier, combining a deep temporal "
        "multi-scale neural network with an optimized gradient-boosted tree model. The system predicts "
        "in-hospital mortality risk in real time from longitudinal physiological data under strict streaming causality "
        "(zero future information leakage)."
    )

    p_comp = doc.add_paragraph()
    p_comp.paragraph_format.space_after = Pt(10)
    p_comp.add_run(
        "• Neural Multi-Scale Stream: 2-layer LSTM (96 hidden units) with additive temporal attention processing "
        "the 90-minute acute vital waveforms, fused with an MLP branch encoding 8 static demographic/ICU covariates "
        "and 36 longitudinal 24-hour baseline trend features.\n"
        "• LightGBM Clinical Stream: Fast, highly regularized gradient-boosted decision tree model trained over 116 "
        "engineered clinical parameters (static demographics, multi-scale 24h trends, and window summary statistics).\n"
        "• Calibrated Meta-Ensemble: Optimal probability blend of 0.10 × Neural Stream + 0.90 × LightGBM Stream."
    )

    # ==========================================
    # 2. Complete Performance Metrics Table
    # ==========================================
    add_heading_with_spacing(doc, "2. Performance Metrics Evaluation", level=1)

    p_tbl_note = doc.add_paragraph()
    p_tbl_note.paragraph_format.space_after = Pt(6)
    p_tbl_note.add_run(
        "Evaluated on unseen patient records using GroupShuffleSplit patient-level separation. "
        "Fresh Holdout represents unseen PhysioNet 2012 Set-B patients with all locked evaluation slices excluded."
    )

    table = doc.add_table(rows=7, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    t_widths = [Inches(2.0), Inches(1.2), Inches(1.2), Inches(1.3), Inches(1.3)]
    
    headers = [
        "Metric",
        "Level 1 & 2\nNeural Multi-Scale",
        "Level 3\nLightGBM Tree",
        "Level 3 HYBRID\n(Fresh Holdout)",
        "Level 3 HYBRID\n(Validation Set)"
    ]
    
    for col_idx, h_text in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.width = t_widths[col_idx]
        set_cell_background(cell, HEX_HEADER_BG)
        set_cell_margins(cell, top=140, bottom=140, left=120, right=120)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(h_text)
        run.bold = True
        run.font.name = 'Arial'
        run.font.size = Pt(9)
        run.font.color.rgb = WHITE

    rows_data = [
        ("AUROC (Discrimination)", "0.8284", "0.8417", "0.8432", "0.8456"),
        ("Accuracy", "71.84%", "76.33%", "76.01%", "77.50%"),
        ("Sensitivity / Recall (TPR)", "79.72%", "76.16%", "76.48%", "74.79%"),
        ("Specificity (TNR)", "70.54%", "76.35%", "75.94%", "77.98%"),
        ("Precision (PPV)", "30.77%", "34.60%", "34.30%", "37.76%"),
        ("F1-Score", "0.4440", "0.4758", "0.4736", "0.5018"),
    ]

    for row_idx, r_data in enumerate(rows_data):
        row = table.rows[row_idx + 1]
        bg_col = HEX_ROW_ALT if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            cell.width = t_widths[col_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.size = Pt(9.5)
            if col_idx == 0:
                run.bold = True
                run.font.color.rgb = NAVY
            elif col_idx == 3:
                run.bold = True
                run.font.color.rgb = TEAL

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ==========================================
    # 3. Clinical Operating Points & Alerting Thresholds
    # ==========================================
    add_heading_with_spacing(doc, "3. Clinical Operating Points & Decision Thresholds", level=1)

    p_op = doc.add_paragraph()
    p_op.paragraph_format.space_after = Pt(6)
    p_op.add_run(
        "To adapt to varying clinical workflows and mitigate alarm fatigue, the final model provides "
        "three calibrated operational alerting tiers:"
    )

    op_table = doc.add_table(rows=4, cols=5)
    op_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    op_table.autofit = False
    
    op_widths = [Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.9)]
    op_headers = ["Operating Tier", "Alert Threshold", "Sensitivity", "Specificity", "Clinical Action"]

    for col_idx, h_text in enumerate(op_headers):
        cell = op_table.cell(0, col_idx)
        cell.width = op_widths[col_idx]
        set_cell_background(cell, HEX_HEADER_BG)
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if 0 < col_idx < 4 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(h_text)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = WHITE

    op_rows = [
        ("Tier 1: Ward Screening", "0.50", "76.48%", "75.94%", "Routine bedside monitoring; nurse check"),
        ("Tier 2: High Confidence", "0.70", "48.00%", "90.20%", "Rapid Response Team (RRT) assessment"),
        ("Tier 3: Critical Alarm", "0.80", "25.60%", "96.80%", "Urgent ICU physician code escalation"),
    ]

    for row_idx, r_data in enumerate(op_rows):
        row = op_table.rows[row_idx + 1]
        bg_col = HEX_ROW_ALT if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            cell.width = op_widths[col_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if 0 < col_idx < 4 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.size = Pt(9)
            if col_idx == 0:
                run.bold = True
                run.font.color.rgb = NAVY

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ==========================================
    # 4. Training Datasets & Cohort Scale
    # ==========================================
    add_heading_with_spacing(doc, "4. Training & Pretraining Dataset Scale", level=1)

    p_data = doc.add_paragraph()
    p_data.paragraph_format.space_after = Pt(6)
    p_data.add_run(
        "The final model leverages multi-corpus scale-up pretraining followed by expanded supervised training:"
    )

    data_table = doc.add_table(rows=4, cols=4)
    data_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    data_table.autofit = False
    
    d_widths = [Inches(2.2), Inches(1.5), Inches(1.8), Inches(1.5)]
    d_headers = ["Phase", "Datasets Included", "Patient Stays", "Total Windows"]

    for col_idx, h_text in enumerate(d_headers):
        cell = data_table.cell(0, col_idx)
        cell.width = d_widths[col_idx]
        set_cell_background(cell, HEX_HEADER_BG)
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 1 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(h_text)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = WHITE

    d_rows = [
        ("Phase 1: Foundation Pretrain", "Challenge 2019 Set A + Set B", "40,335 stays", "956,732 windows"),
        ("Phase 2: Supervised Fine-Tuning", "PhysioNet 2012 Set A + B + C", "~11,000 stays", "234,868 windows"),
        ("Phase 3: Test & Fresh Holdout", "PhysioNet 2012 Set B (excl locked)", "800 stays (held-out)", "15,312 windows"),
    ]

    for row_idx, r_data in enumerate(d_rows):
        row = data_table.rows[row_idx + 1]
        bg_col = HEX_ROW_ALT if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            cell.width = d_widths[col_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 1 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.size = Pt(9)
            if col_idx == 0:
                run.bold = True
                run.font.color.rgb = NAVY

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ==========================================
    # 5. Input Feature Architecture (116 Parameters)
    # ==========================================
    add_heading_with_spacing(doc, "5. Clinical Feature Architecture (116 Parameters)", level=1)

    p_feat = doc.add_paragraph()
    p_feat.paragraph_format.space_after = Pt(6)
    p_feat.add_run(
        "• Core Dynamic Physiological Vitals & Labs (12 features × 90 min = 1,080 timesteps):\n"
        "   Heart Rate (HR), Respiratory Rate (RespRate), Temperature (Temp), Systolic Blood Pressure (NISysABP),\n"
        "   Diastolic Blood Pressure (NIDiasABP), SpO2, Glasgow Coma Scale (GCS), Blood Urea Nitrogen (BUN),\n"
        "   Serum Creatinine, White Blood Cell Count (WBC), Platelets, and Serum Glucose.\n\n"
        "• Static Patient Demographics & Unit Context (8 features):\n"
        "   Age, Gender, Height, Weight, and 4-unit One-Hot Admission Types (CCU, CSRU, MICU, SICU).\n\n"
        "• Longitudinal 24-Hour Multi-Scale Trend Features (36 features):\n"
        "   Baseline 24-hour mean, acute delta (window mean - baseline mean), and linear trajectory slope per vital.\n\n"
        "• Tabular Window Summary Statistics (72 features):\n"
        "   Mean, minimum, maximum, standard deviation, last observed value, and window delta (last - first) per vital."
    )

    # Footer note
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    p_foot = doc.add_paragraph()
    p_foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_foot = p_foot.add_run("SynCura Clinical Research Prototype — Confidential & Proprietary Medical AI Documentation")
    r_foot.font.size = Pt(8.5)
    r_foot.font.color.rgb = SLATE
    r_foot.font.italic = True

    doc.save(OUT_PATH)
    print(f"Document successfully created at: {OUT_PATH}")
    return OUT_PATH


if __name__ == '__main__':
    create_document()
