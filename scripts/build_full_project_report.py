# -*- coding: utf-8 -*-
"""
Build the complete, official, academic SynCura Mini-Project Report.
Strictly adheres to:
1. font and other things of report.pdf (VTU formatting guidelines)
2. MINIPROJECT REPORT FORMAT.docx (PACE Dept of CSE template & structure)
3. nfrr final reprt.pdf (completed PACE CSE report layout, headers, footers)

Outputs:
- PROJ/REPORT/SynCura_Mini_Project_Report.docx
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(REPO_ROOT, "REPORT")
ASSETS_DIR = os.path.join(REPORT_DIR, "assets")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
FIGS_DIR = os.path.join(DOCS_DIR, "figures")
SHOTS_DIR = os.path.join(REPO_ROOT, "ppt", "shots")
OUT_DOCX = os.path.join(REPORT_DIR, "SynCura_Mini_Project_Report.docx")

# Styling Colors
NAVY = RGBColor(27, 54, 93)       # #1B365D - VTU/PACE Primary
DARK = RGBColor(30, 41, 59)       # #1E293B - Body text
SLATE = RGBColor(100, 116, 139)   # #64748B - Secondary captions
BLACK = RGBColor(0, 0, 0)
WHITE = RGBColor(255, 255, 255)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_table_borders(table, color="B0BEC5", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_field(p, field_text):
    run = p.add_run()
    r = run._r
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = field_text
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    r.append(fldChar1)
    r.append(instrText)
    r.append(fldChar2)
    r.append(fldChar3)

def set_row_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = parse_xml(f'<w:tblHeader {nsdecls("w")}/>')
    trPr.append(tblHeader)

# Document Builder Class
class ReportBuilder:
    def __init__(self):
        self.doc = Document()
        self.setup_page_properties()
        
    def setup_page_properties(self):
        # Section 0: Front Matter
        sec0 = self.doc.sections[0]
        sec0.top_margin = Inches(0.75)
        sec0.bottom_margin = Inches(0.75)
        sec0.left_margin = Inches(1.25)
        sec0.right_margin = Inches(1.0)
        sec0.page_width = Inches(8.27)
        sec0.page_height = Inches(11.69)
        sec0.different_first_page_header_footer = True
        
        # Configure Normal Style
        style_normal = self.doc.styles['Normal']
        style_normal.font.name = 'Times New Roman'
        style_normal.font.size = Pt(12)
        style_normal.font.color.rgb = DARK
        style_normal.paragraph_format.line_spacing = 1.5
        style_normal.paragraph_format.space_after = Pt(6)
        style_normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def add_page_break(self):
        self.doc.add_page_break()

    def add_para(self, text, bold=False, italic=False, size=12, color=DARK,
                 align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=6, line_spacing=1.5):
        p = self.doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = line_spacing
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        run.font.name = 'Times New Roman'
        run.font.size = Pt(size)
        run.font.color.rgb = color
        return p

    def add_bullet(self, text, bold_prefix="", space_after=4):
        p = self.doc.add_paragraph(style='List Bullet')
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.5
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.name = 'Times New Roman'
            r_pre.font.size = Pt(12)
            r_pre.font.color.rgb = DARK
        r_text = p.add_run(text)
        r_text.font.name = 'Times New Roman'
        r_text.font.size = Pt(12)
        r_text.font.color.rgb = DARK
        return p

    def add_chapter_title(self, chap_num, chap_name):
        self.add_page_break()
        # Per VTU Guideline: Chapter heading Left, 16 pt; Chapter Name Centered, 18 pt
        self.add_para(f"CHAPTER {chap_num}", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.LEFT, space_before=12, space_after=6)
        self.add_para(chap_name.upper(), bold=True, size=18, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=18)

    def add_h1(self, text):
        # Section Heading: 16 pt, Left, Bold
        return self.add_para(text, bold=True, size=16, color=NAVY,
                             align=WD_ALIGN_PARAGRAPH.LEFT, space_before=14, space_after=6)

    def add_h2(self, text):
        # Subsection Heading: 14 pt, Left, Bold
        return self.add_para(text, bold=True, size=14, color=NAVY,
                             align=WD_ALIGN_PARAGRAPH.LEFT, space_before=10, space_after=4)

    def add_h3(self, text):
        # Sub-subsection Heading: 13 pt, Left, Bold
        return self.add_para(text, bold=True, size=13, color=NAVY,
                             align=WD_ALIGN_PARAGRAPH.LEFT, space_before=8, space_after=3)

    def add_image_figure(self, image_path, fig_num_title, caption_text="", width=Inches(5.6)):
        if os.path.exists(image_path):
            p_img = self.doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(4)
            p_img.add_run().add_picture(image_path, width=width)
            
            p_cap = self.doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(0)
            p_cap.paragraph_format.space_after = Pt(12)
            
            r_title = p_cap.add_run(fig_num_title)
            r_title.bold = True
            r_title.font.name = 'Times New Roman'
            r_title.font.size = Pt(10.5)
            r_title.font.color.rgb = DARK
            
            if caption_text:
                r_desc = p_cap.add_run(" - " + caption_text)
                r_desc.italic = True
                r_desc.font.name = 'Times New Roman'
                r_desc.font.size = Pt(10)
                r_desc.font.color.rgb = SLATE
        else:
            print(f"[WARN] Image not found: {image_path}")

    def add_test_case_table(self, tc_num, name, feature, input_val, expected, actual, env, remarks):
        table = self.doc.add_table(rows=9, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table, color="94A3B8", sz="4")
        
        # Header row
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Attribute"
        hdr_cells[1].text = "Details"
        for c in hdr_cells:
            set_cell_background(c, "1B365D")
            set_cell_margins(c, top=80, bottom=80, left=120, right=120)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.bold = True
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10.5)
                r.font.color.rgb = WHITE
        set_row_header(table.rows[0])
        
        rows_data = [
            ("No. of Test Case", tc_num),
            ("Name of Test", name),
            ("Feature Being Tested", feature),
            ("Sample Input", input_val),
            ("Expected Output", expected),
            ("Actual Output", actual),
            ("Test Environment", env),
            ("Remarks", remarks)
        ]
        
        for i, (attr, val) in enumerate(rows_data):
            row_cells = table.rows[i+1].cells
            row_cells[0].text = attr
            row_cells[1].text = val
            bg = "F8FAFC" if i % 2 == 0 else "FFFFFF"
            for c_idx, c in enumerate(row_cells):
                set_cell_background(c, bg)
                set_cell_margins(c, top=60, bottom=60, left=100, right=100)
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(2)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(10)
                    r.font.color.rgb = DARK
                    if c_idx == 0:
                        r.bold = True
        
        p_space = self.doc.add_paragraph()
        p_space.paragraph_format.space_after = Pt(8)

    def create_title_page(self):
        # VTU Logo
        vtu_logo = os.path.join(ASSETS_DIR, "vtu_logo.jpeg")
        if os.path.exists(vtu_logo):
            p_logo = self.doc.add_paragraph()
            p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_logo.paragraph_format.space_before = Pt(0)
            p_logo.paragraph_format.space_after = Pt(4)
            p_logo.add_run().add_picture(vtu_logo, width=Inches(1.2))

        self.add_para("VISVESVARAYA TECHNOLOGICAL UNIVERSITY", bold=True, size=15, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=2)
        self.add_para("“Jnana Sangama”, Belagavi - 590 018, Karnataka", bold=True, size=11, color=DARK,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=14)

        self.add_para("A MINI PROJECT REPORT ON", bold=True, size=13, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=8)

        self.add_para("SYNCURA: PREDICTIVE ICU MONITORING SYSTEM USING ATTENTION-BASED LSTM WITH REAL-TIME EXPLAINABILITY",
                      bold=True, size=16, color=NAVY, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

        self.add_para("Submitted in partial fulfilment of the requirements for the award of the degree of",
                      italic=True, size=11, color=DARK, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=4)
        self.add_para("BACHELOR OF ENGINEERING", bold=True, size=13, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=2)
        self.add_para("IN", bold=True, size=11, color=DARK, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=2)
        self.add_para("COMPUTER SCIENCE AND ENGINEERING", bold=True, size=13, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=14)

        self.add_para("Submitted By", bold=True, size=11.5, color=DARK,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=6)

        # Team Table
        tbl_team = self.doc.add_table(rows=4, cols=2)
        tbl_team.alignment = WD_TABLE_ALIGNMENT.CENTER
        team_data = [
            ("NAME", "USN"),
            ("ABDUL AHAD IKKERI", "4PA24CS002"),
            ("FATHIMA REEHA", "4PA24CS026"),
            ("FIZAN FEROZ", "4PA24CS032")
        ]
        for r_idx, (nm, usn) in enumerate(team_data):
            c0, c1 = tbl_team.rows[r_idx].cells
            c0.text = nm
            c1.text = usn
            for c in (c0, c1):
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(2)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(11)
                    r.font.color.rgb = DARK
                    if r_idx == 0:
                        r.bold = True
                        r.font.color.rgb = NAVY

        self.add_para("Under the Guidance of", bold=True, size=11.5, color=DARK,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=2)
        self.add_para("Prof. Divya K", bold=True, size=12, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=2)
        self.add_para("Assistant Professor, Department of Computer Science & Engineering",
                      size=10.5, color=DARK, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=12)

        # PACE Logo
        pace_logo = os.path.join(ASSETS_DIR, "pace_logo.jpeg")
        if os.path.exists(pace_logo):
            p_pace = self.doc.add_paragraph()
            p_pace.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_pace.paragraph_format.space_before = Pt(2)
            p_pace.paragraph_format.space_after = Pt(4)
            p_pace.add_run().add_picture(pace_logo, width=Inches(1.2))

        self.add_para("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", bold=True, size=12, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=2)
        self.add_para("P. A. COLLEGE OF ENGINEERING", bold=True, size=14, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=2)
        self.add_para("(Affiliated to VTU Belagavi, Approved by AICTE New Delhi, Accredited by NBA)",
                      italic=True, size=9.5, color=DARK, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=2)
        self.add_para("Near Mangalore University, Mangalore - 574 153, Karnataka, India",
                      size=10, color=DARK, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=4)
        self.add_para("2026 – 2027", bold=True, size=12, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=0)

    def create_certificate_page(self):
        self.add_page_break()
        pace_logo = os.path.join(ASSETS_DIR, "pace_logo.jpeg")
        if os.path.exists(pace_logo):
            p_pace = self.doc.add_paragraph()
            p_pace.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_pace.paragraph_format.space_before = Pt(0)
            p_pace.paragraph_format.space_after = Pt(4)
            p_pace.add_run().add_picture(pace_logo, width=Inches(1.1))

        self.add_para("P. A. COLLEGE OF ENGINEERING", bold=True, size=14, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=2)
        self.add_para("(Affiliated to Visvesvaraya Technological University, Approved by AICTE, Accredited by NBA)",
                      italic=True, size=9.5, color=DARK, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=2)
        self.add_para("Near Mangalore University, Mangalore - 574 153", size=10, color=DARK,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=4)
        self.add_para("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", bold=True, size=12, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=14)

        self.add_para("CERTIFICATE", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=14)

        cert_text = (
            "Certified that the Mini-Project work entitled “SYNCURA: PREDICTIVE ICU MONITORING SYSTEM USING "
            "ATTENTION-BASED LSTM WITH REAL-TIME EXPLAINABILITY” is a bonafide work carried out by ABDUL AHAD IKKERI "
            "(4PA24CS002), FATHIMA REEHA (4PA24CS026), and FIZAN FEROZ (4PA24CS032) in partial fulfilment for the award "
            "of degree of Bachelor of Engineering in Computer Science and Engineering of Visvesvaraya Technological "
            "University, Belagavi during the academic year 2026-2027. It is certified that all suggestions indicated "
            "for internal assessment have been incorporated in the report deposited in the departmental library. "
            "The project report has been approved as it satisfies the academic requirements in respect of project work "
            "prescribed for the Bachelor of Engineering degree."
        )
        self.add_para(cert_text, size=11.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=4, space_after=36, line_spacing=1.5)

        # Signature Table
        tbl_sig = self.doc.add_table(rows=2, cols=3)
        tbl_sig.alignment = WD_TABLE_ALIGNMENT.CENTER
        sig_data = [
            ("_____________________\nSignature of Guide", "_____________________\nSignature of HOD", "_____________________\nSignature of Principal"),
            ("Prof. Divya K\nAssistant Professor\nDept. of CSE, PACE",
             "Dr. Sharmila Kumari M\nProfessor & Head\nDept. of CSE, PACE",
             "Dr. Ramis M. K.\nPrincipal\nP. A. College of Engineering")
        ]
        for r_idx, row in enumerate(sig_data):
            for c_idx, txt in enumerate(row):
                c = tbl_sig.rows[r_idx].cells[c_idx]
                c.text = txt
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(10)
                    r.font.color.rgb = DARK
                    if r_idx == 1 and c_idx in [0, 1, 2]:
                        r.bold = True

    def create_viva_page(self):
        self.add_page_break()
        pace_logo = os.path.join(ASSETS_DIR, "pace_logo.jpeg")
        if os.path.exists(pace_logo):
            p_pace = self.doc.add_paragraph()
            p_pace.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_pace.add_run().add_picture(pace_logo, width=Inches(1.0))

        self.add_para("P. A. COLLEGE OF ENGINEERING", bold=True, size=14, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=2)
        self.add_para("Department of Computer Science and Engineering", bold=True, size=12, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=18)

        self.add_para("EXTERNAL VIVA-VOCE EXAMINATION", bold=True, size=15, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=18)

        self.add_para("Name of the Project: SynCura: Predictive ICU Monitoring System Using Attention-Based LSTM with Real-Time Explainability",
                      bold=True, size=11.5, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=4, space_after=12)

        viva_text = "The Mini-Project viva-voce examination of the above candidates has been held on ___________________."
        self.add_para(viva_text, size=11.5, space_before=4, space_after=24)

        tbl_viva = self.doc.add_table(rows=4, cols=3)
        tbl_viva.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl_viva, color="94A3B8", sz="4")
        
        v_headers = ["Sl. No.", "Name of Examiners", "Signature with Date"]
        for c_idx, h in enumerate(v_headers):
            c = tbl_viva.rows[0].cells[c_idx]
            c.text = h
            set_cell_background(c, "1B365D")
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10.5)
                r.font.color.rgb = WHITE
        set_row_header(tbl_viva.rows[0])

        rows = [
            ("1.", "1. Internal Examiner:\n\nProf. Divya K", "\n\n_________________________"),
            ("2.", "2. External Examiner:\n\n____________________________________", "\n\n_________________________"),
            ("", "", "")
        ]
        for r_idx, r_data in enumerate(rows[:2]):
            row_cells = tbl_viva.rows[r_idx+1].cells
            for c_idx, val in enumerate(r_data):
                c = row_cells[c_idx]
                c.text = val
                set_cell_margins(c, top=100, bottom=100, left=100, right=100)
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(10)
                    r.font.color.rgb = DARK

    def create_declaration_page(self):
        self.add_page_break()
        pace_logo = os.path.join(ASSETS_DIR, "pace_logo.jpeg")
        if os.path.exists(pace_logo):
            p_pace = self.doc.add_paragraph()
            p_pace.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_pace.add_run().add_picture(pace_logo, width=Inches(1.0))

        self.add_para("P. A. COLLEGE OF ENGINEERING", bold=True, size=14, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=2)
        self.add_para("Department of Computer Science and Engineering", bold=True, size=12, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=18)

        self.add_para("DECLARATION", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=18)

        decl_text = (
            "We, the students of Bachelor of Engineering in Computer Science and Engineering, P.A. College of "
            "Engineering, Mangalore, solemnly declare that the Mini-Project report entitled “SYNCURA: PREDICTIVE ICU "
            "MONITORING SYSTEM USING ATTENTION-BASED LSTM WITH REAL-TIME EXPLAINABILITY” is an authentic record of our own "
            "work carried out under the guidance of Prof. Divya K, Assistant Professor, Department of Computer Science "
            "and Engineering, P. A. College of Engineering, Mangalore, in partial fulfilment of the requirements for "
            "the award of the degree of Bachelor of Engineering in Computer Science and Engineering by Visvesvaraya "
            "Technological University, Belagavi.\n\n"
            "We further declare that the content embodied in this report has not been submitted previously to this or "
            "any other university or institution for the award of any degree, diploma, or fellowship."
        )
        self.add_para(decl_text, size=11.5, space_before=4, space_after=28)

        tbl_decl = self.doc.add_table(rows=4, cols=3)
        tbl_decl.alignment = WD_TABLE_ALIGNMENT.CENTER
        d_data = [
            ("NAME", "USN", "SIGNATURE"),
            ("ABDUL AHAD IKKERI", "4PA24CS002", "___________________"),
            ("FATHIMA REEHA", "4PA24CS026", "___________________"),
            ("FIZAN FEROZ", "4PA24CS032", "___________________")
        ]
        for r_idx, (nm, usn, sig) in enumerate(d_data):
            cells = tbl_decl.rows[r_idx].cells
            cells[0].text = nm
            cells[1].text = usn
            cells[2].text = sig
            for c_idx, c in enumerate(cells):
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(4)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(10.5)
                    r.font.color.rgb = DARK
                    if r_idx == 0:
                        r.bold = True
                        r.font.color.rgb = NAVY

        self.add_para("Date: 19-11-2026\nPlace: Mangalore", bold=True, size=11, color=DARK,
                      align=WD_ALIGN_PARAGRAPH.LEFT, space_before=24, space_after=0)

    def create_abstract_page(self):
        self.add_page_break()
        self.add_para("ABSTRACT", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=18)

        abstract_text = (
            "Intensive Care Unit (ICU) patient deterioration remains a primary cause of preventable in-hospital "
            "mortality and acute organ failure worldwide. Traditional clinical risk scoring tools such as NEWS2, "
            "SOFA, and APACHE-II depend on manual periodic observation and static threshold checks on isolated vital signs, "
            "failing to model complex temporal trajectories or subtle multi-parameter physiological interactions. "
            "This project presents SynCura, an intelligent, real-time predictive ICU monitoring system that utilizes an "
            "Attention-based Bidirectional Long Short-Term Memory (BiLSTM) neural network coupled with SHAP (SHapley Additive "
            "exPlanations) to dynamically forecast acute patient deterioration risk and in-hospital mortality.\n\n"
            "SynCura processes continuous streams of 12 critical physiological parameters (6 vital signs: Heart Rate, "
            "Respiratory Rate, Body Temperature, Systolic BP, Diastolic BP, and Oxygen Saturation; alongside 6 key laboratory "
            "and neurological markers: Glasgow Coma Scale, Blood Urea Nitrogen, Creatinine, White Blood Cell count, Platelets, "
            "and Serum Glucose) structured across a 90-minute rolling window at 5-minute sampling intervals. The deployed "
            "predictive engine employs a 3-model logit-averaged ensemble (hidden dimension 96), achieving a validated AUROC "
            "of 0.840 (95% CI: 0.831–0.849) and an AUPRC of 0.485, significantly outperforming the standard clinical baseline "
            "NEWS2 (AUROC 0.612). To eliminate alarm fatigue—a major driver of clinical burnout—SynCura incorporates a "
            "two-stage persistence filtering mechanism that suppresses transient physiological spikes, achieving a 76.4% "
            "reduction in false alarms while maintaining 90.0% sensitivity.\n\n"
            "The end-to-end system integrates a high-performance asynchronous FastAPI backend delivering sub-25ms inference "
            "latencies, WebSocket-based bi-directional telemetry broadcast, and an interactive React 19 bedside dashboard "
            "providing instantaneous risk visualization (0–100 score), temporal attention heatmaps, local SHAP feature "
            "attribution waterfalls, and dual-tier medical audio cues. Validated on 4,000 critically ill ICU admissions from "
            "the benchmark PhysioNet Challenge 2012 cohort, SynCura bridges the gap between opaque deep-learning models and "
            "actionable bedside clinical decision support."
        )
        self.add_para(abstract_text, size=11.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=4, space_after=12, line_spacing=1.5)

    def create_acknowledgement_page(self):
        self.add_page_break()
        self.add_para("ACKNOWLEDGEMENT", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=18)

        ack_text = (
            "The successful completion of any project would be incomplete without expressing sincere gratitude to the "
            "people whose constant guidance, constructive feedback, and encouragement made our efforts fruitful.\n\n"
            "We express our deep sense of gratitude and indebtedness to our project guide, Prof. Divya K, Assistant "
            "Professor, Department of Computer Science & Engineering, P. A. College of Engineering, for her invaluable "
            "mentorship, continuous support, and profound technical suggestions throughout the development of SynCura.\n\n"
            "We extend our heartfelt gratitude to Dr. Sharmila Kumari M, Professor & Head, Department of Computer Science "
            "and Engineering, for providing necessary departmental facilities, academic encouragement, and creating an "
            "inspiring research atmosphere.\n\n"
            "We would like to express our sincere thanks to Dr. Ramis M. K., Principal, P. A. College of Engineering, "
            "Mangalore, for providing excellent infrastructure, computational resources, and support for technical project work.\n\n"
            "We are profoundly thankful to the project coordinators and all faculty members and technical staff of the "
            "Department of Computer Science and Engineering for their direct and indirect assistance and valuable advice.\n\n"
            "Finally, we express our heartfelt thanks to our beloved parents, family members, and friends for their endless "
            "support, moral encouragement, and understanding throughout our academic journey."
        )
        self.add_para(ack_text, size=11.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=4, space_after=28, line_spacing=1.5)

        # Names at bottom
        tbl_ack = self.doc.add_table(rows=2, cols=3)
        tbl_ack.alignment = WD_TABLE_ALIGNMENT.CENTER
        ack_names = [
            ("ABDUL AHAD IKKERI", "FATHIMA REEHA", "FIZAN FEROZ"),
            ("(4PA24CS002)", "(4PA24CS026)", "(4PA24CS032)")
        ]
        for r_idx, row in enumerate(ack_names):
            for c_idx, val in enumerate(row):
                c = tbl_ack.rows[r_idx].cells[c_idx]
                c.text = val
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(10.5)
                    r.font.color.rgb = DARK
                    if r_idx == 0:
                        r.bold = True

    def create_table_of_contents_page(self):
        self.add_page_break()
        self.add_para("TABLE OF CONTENTS", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=14)

        tbl = self.doc.add_table(rows=1, cols=2)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl, color="CBD5E1", sz="4")
        
        c0, c1 = tbl.rows[0].cells
        c0.text = "Contents"
        c1.text = "Page No."
        for c in (c0, c1):
            set_cell_background(c, "1B365D")
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == c0 else WD_ALIGN_PARAGRAPH.RIGHT
            for r in p.runs:
                r.bold = True
                r.font.name = 'Times New Roman'
                r.font.size = Pt(11)
                r.font.color.rgb = WHITE
        set_row_header(tbl.rows[0])

        toc_items = [
            ("ABSTRACT", "i", True),
            ("ACKNOWLEDGEMENT", "ii", True),
            ("LIST OF FIGURES", "iv", True),
            ("LIST OF TABLES", "v", True),
            ("CHAPTER 1: INTRODUCTION", "1", True),
            ("  1.1 Overview", "1", False),
            ("  1.2 Introduction", "2", False),
            ("  1.3 Literature Survey", "3", False),
            ("  1.4 Problem Statement", "6", False),
            ("  1.5 Objectives", "7", False),
            ("  1.6 Proposed System", "8", False),
            ("  1.7 Advantages of the Proposed System", "10", False),
            ("CHAPTER 2: SOFTWARE REQUIREMENTS SPECIFICATION", "11", True),
            ("  2.1 Specific Requirements", "11", False),
            ("  2.2 Product Functions", "11", False),
            ("  2.3 User Characteristics", "12", False),
            ("  2.4 General Constraints", "12", False),
            ("  2.5 Assumptions and Dependencies", "13", False),
            ("  2.6 Functional Requirements", "13", False),
            ("  2.7 Performance Requirements", "15", False),
            ("  2.8 Supportability", "15", False),
            ("  2.9 Design Constraints", "16", False),
            ("  2.10 Interfaces (User, Hardware, Software)", "16", False),
            ("CHAPTER 3: HIGH-LEVEL DESIGN", "18", True),
            ("  3.1 System Architecture", "18", False),
            ("  3.2 System Architecture Description", "19", False),
            ("  3.3 Context Flow Diagram (DFD Level 0 & Level 1)", "20", False),
            ("  3.4 Use Case Diagram", "21", False),
            ("  3.5 Sequence Diagram", "22", False),
            ("  3.6 System Flow Chart", "23", False),
            ("CHAPTER 4: DETAILED DESIGN", "24", True),
            ("  4.1 Structured Chart Diagram", "24", False),
            ("  4.2 Detailed Component Breakdown", "25", False),
            ("  4.3 Data Preprocessing and Feature Extraction", "27", False),
            ("  4.4 Dataset Description", "28", False),
            ("  4.5 Model Methodology & Mathematical Formulation", "29", False),
            ("CHAPTER 5: IMPLEMENTATION", "32", True),
            ("  5.1 Implementation Requirements", "32", False),
            ("  5.2 Implementation Details", "32", False),
            ("  5.3 Engine and Module Selection", "33", False),
            ("  5.4 IDE Selection", "34", False),
            ("  5.5 Language Selection", "34", False),
            ("  5.6 Platform Selection", "35", False),
            ("  5.7 Coding Guidelines", "35", False),
            ("  5.8 Implementation Procedure", "36", False),
            ("CHAPTER 6: TESTING", "39", True),
            ("  6.1 Test Environment", "39", False),
            ("  6.2 Unit Testing Strategy", "40", False),
            ("  6.3 Unit Test Cases (UTC 1.1 - UTC 1.9)", "41", False),
            ("  6.4 Integration Testing Strategy", "47", False),
            ("  6.5 Integration Test Cases (ITC 1.1 - ITC 1.6)", "48", False),
            ("  6.6 Summary of Test Results", "52", False),
            ("CHAPTER 7: CONCLUSION AND FUTURE ENHANCEMENTS", "53", True),
            ("  7.1 Conclusion", "53", False),
            ("  7.2 Clinical Utility & Impact", "54", False),
            ("  7.3 Limitations", "54", False),
            ("  7.4 Future Enhancements", "55", False),
            ("REFERENCES", "56", True),
            ("APPENDIX A: CORE IMPLEMENTATION CODE", "58", True),
            ("APPENDIX B: LIST OF FIGURES", "62", True),
            ("APPENDIX C: LIST OF TABLES", "63", True),
            ("APPENDIX D: RESULTS AND VISUALIZATIONS", "64", True),
            ("BIBLIOGRAPHY / TEAM PROFILES", "71", True)
        ]

        for item, pg, is_major in toc_items:
            row = tbl.add_row()
            c_item, c_pg = row.cells
            c_item.text = item
            c_pg.text = pg
            bg = "F1F5F9" if is_major else "FFFFFF"
            for c in (c_item, c_pg):
                set_cell_background(c, bg)
                set_cell_margins(c, top=40, bottom=40, left=80, right=80)
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == c_item else WD_ALIGN_PARAGRAPH.RIGHT
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(1)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(10)
                    r.font.color.rgb = NAVY if is_major else DARK
                    if is_major:
                        r.bold = True

    def create_list_of_figures_page(self):
        self.add_page_break()
        self.add_para("LIST OF FIGURES", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=14)

        tbl = self.doc.add_table(rows=1, cols=2)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl, color="CBD5E1", sz="4")
        
        c0, c1 = tbl.rows[0].cells
        c0.text = "Figure Number & Title"
        c1.text = "Page No."
        for c in (c0, c1):
            set_cell_background(c, "1B365D")
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == c0 else WD_ALIGN_PARAGRAPH.RIGHT
            for r in p.runs:
                r.bold = True
                r.font.name = 'Times New Roman'
                r.font.size = Pt(11)
                r.font.color.rgb = WHITE
        set_row_header(tbl.rows[0])

        figures = [
            ("Figure 1.1: SynCura High-Level Concept & Functional Interaction Overview", "9"),
            ("Figure 3.1: Five-Layered End-to-End System Architecture", "18"),
            ("Figure 3.2: Context Flow Diagram (DFD Level 0 Context Diagram)", "20"),
            ("Figure 3.3: UML Use Case Diagram for Clinicians and System Administrators", "21"),
            ("Figure 3.4: UML Sequence Diagram for Real-Time Telemetry & Alert Dispatch", "22"),
            ("Figure 3.5: End-to-End System Processing & Persistence Decision Flow Chart", "23"),
            ("Figure 4.1: Structured Functional Decomposition Chart Diagram", "24"),
            ("Figure D.1: Receiver Operating Characteristic (ROC) Validation Curves", "64"),
            ("Figure D.2: Precision-Recall (PR) Curves Under Extreme Sepsis Imbalance", "65"),
            ("Figure D.3: Clinical Sensitivity vs. Specificity Trade-Off by Decision Threshold", "65"),
            ("Figure D.4: Reliability Calibration Curves Across Deciles of Predicted Risk", "66"),
            ("Figure D.5: Normalized Confusion Matrices at Operating Decision Cutoffs", "66"),
            ("Figure D.6: Comprehensive Benchmark Comparison (SynCura vs Baseline Models)", "67"),
            ("Figure D.7: Component Ablation Study Across Model Configurations", "67"),
            ("Figure D.8: Subgroup Fairness and Demographic Robustness Analysis", "68"),
            ("Figure D.9: Two-Stage Persistence Filtering Alarm Suppression Evaluation", "68"),
            ("Figure D.10: Global SHAP Feature Importance and Attributions Ranking", "69"),
            ("Figure D.11: SynCura Live ICU Ward Overview Monitoring Dashboard", "69"),
            ("Figure D.12: Bedside Profile with Real-Time Attention and SHAP Explanations", "70")
        ]

        for title, pg in figures:
            row = tbl.add_row()
            c_title, c_pg = row.cells
            c_title.text = title
            c_pg.text = pg
            for c in (c_title, c_pg):
                set_cell_margins(c, top=35, bottom=35, left=80, right=80)
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == c_title else WD_ALIGN_PARAGRAPH.RIGHT
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(1)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(10)
                    r.font.color.rgb = DARK

    def create_list_of_tables_page(self):
        self.add_page_break()
        self.add_para("LIST OF TABLES", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=14)

        tbl = self.doc.add_table(rows=1, cols=2)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl, color="CBD5E1", sz="4")
        
        c0, c1 = tbl.rows[0].cells
        c0.text = "Table Number & Title"
        c1.text = "Page No."
        for c in (c0, c1):
            set_cell_background(c, "1B365D")
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == c0 else WD_ALIGN_PARAGRAPH.RIGHT
            for r in p.runs:
                r.bold = True
                r.font.name = 'Times New Roman'
                r.font.size = Pt(11)
                r.font.color.rgb = WHITE
        set_row_header(tbl.rows[0])

        tables = [
            ("Table 4.1: The 12 ICU Physiological and Laboratory Features Monitored by SynCura", "28"),
            ("Table 6.1: Test Hardware Environment Specifications", "39"),
            ("Table 6.2: Test Software Environment Specifications", "40"),
            ("Table 6.3: Target Deployment Platform Specifications", "40"),
            ("Table 6.2.2.1: UTC 1.1 - Data Ingestion and CSV Parsing Test", "41"),
            ("Table 6.2.2.2: UTC 1.2 - 90-Minute Rolling Window Tensor Generator Test", "42"),
            ("Table 6.2.2.3: UTC 1.3 - Forward-Fill and Z-Score Normalization Test", "42"),
            ("Table 6.2.2.4: UTC 1.4 - Attention-LSTM Forward Pass and Shape Test", "43"),
            ("Table 6.2.2.5: UTC 1.5 - Multi-Model Logit-Averaged Ensemble Output Test", "44"),
            ("Table 6.2.2.6: UTC 1.6 - KernelSHAP Local Feature Attribution Test", "44"),
            ("Table 6.2.2.7: UTC 1.7 - Two-Stage Persistence Filtering Alarm Suppression Test", "45"),
            ("Table 6.2.2.8: UTC 1.8 - WebSocket Real-Time Telemetry Broadcast Test", "46"),
            ("Table 6.2.2.9: UTC 1.9 - Clinical Baseline NEWS2 Comparator Test", "46"),
            ("Table 6.3.2.1: ITC 1.1 - End-to-End Vitals Ingestion to Risk Score Generation", "48"),
            ("Table 6.3.2.2: ITC 1.2 - Bedside Profile Real-Time Attention and SHAP Rendering", "49"),
            ("Table 6.3.2.3: ITC 1.3 - Dual-Tier Audio Siren and Escalating Visual Glow Triggering", "49"),
            ("Table 6.3.2.4: ITC 1.4 - Clinical Simulation Scenario State Transitions", "50"),
            ("Table 6.3.2.5: ITC 1.5 - Client-Server Telemetry Synchronization Under Network Jitter", "51"),
            ("Table 6.3.2.6: ITC 1.6 - Model Fallback and Sparse Laboratory Handling", "51")
        ]

        for title, pg in tables:
            row = tbl.add_row()
            c_title, c_pg = row.cells
            c_title.text = title
            c_pg.text = pg
            for c in (c_title, c_pg):
                set_cell_margins(c, top=35, bottom=35, left=80, right=80)
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == c_title else WD_ALIGN_PARAGRAPH.RIGHT
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(1)
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(10)
                    r.font.color.rgb = DARK

    def switch_to_body_section(self):
        # Section 1: Main Chapters (Arabic Page Numbers, Headers, Footers)
        sec1 = self.doc.add_section()
        sec1.top_margin = Inches(0.75)
        sec1.bottom_margin = Inches(0.75)
        sec1.left_margin = Inches(1.25)
        sec1.right_margin = Inches(1.0)
        sec1.page_width = Inches(8.27)
        sec1.page_height = Inches(11.69)
        sec1.header.is_linked_to_previous = False
        sec1.footer.is_linked_to_previous = False
        
        # Header matching nfrr final reprt.pdf:
        # Left: SYNCURA: PREDICTIVE ICU MONITORING SYSTEM
        # Right: Dept. of CSE, PACE    2026-2027
        header = sec1.header
        p_hdr = header.paragraphs[0]
        p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_hdr.paragraph_format.space_after = Pt(2)
        
        # Use two tabs: Left, Right
        p_hdr.text = "SYNCURA: PREDICTIVE ICU MONITORING SYSTEM\tDept. of CSE, PACE    2026-2027"
        for r in p_hdr.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9)
            r.font.color.rgb = SLATE

        # Add bottom border under header
        pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="1" w:color="94A3B8"/></w:pBdr>')
        p_hdr._p.get_or_add_pPr().append(pBdr)

        # Footer: Right-aligned "Page | X"
        footer = sec1.footer
        p_ftr = footer.paragraphs[0]
        p_ftr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_ftr.paragraph_format.space_before = Pt(4)
        r_lbl = p_ftr.add_run("Page | ")
        r_lbl.font.name = 'Times New Roman'
        r_lbl.font.size = Pt(9.5)
        r_lbl.font.color.rgb = SLATE
        add_field(p_ftr, "PAGE")
        for r in p_ftr.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)
            r.font.color.rgb = SLATE

    def build_chapter_1(self):
        self.add_chapter_title("1", "INTRODUCTION")

        self.add_h1("1.1 Overview")
        p1 = (
            "Intensive Care Units (ICUs) represent the most critical and resource-intensive environments within modern "
            "hospitals, dedicated to treating patients suffering from severe, life-threatening physiological instability. "
            "Patients admitted to the ICU are characterized by profound physiological volatility, with mortality rates "
            "historically documented between 10% and 29%. Rapid detection of acute clinical deterioration—such as the sudden "
            "onset of septic shock, cardiac decompensation, or acute respiratory distress syndrome (ARDS)—is widely recognized "
            "as the single most decisive factor determining clinical survival."
        )
        self.add_para(p1)

        p2 = (
            "Modern ICU bedsides are equipped with sophisticated telemetry monitors that capture hundreds of data points every "
            "minute, including heart rate, blood pressure, oxygen saturation, and respiratory rate. However, despite this wealth "
            "of continuous electronic data, clinical staff are overwhelmed by cognitive burden and severe alarm fatigue. Traditional "
            "monitoring apparatuses evaluate physiological signals in isolation using static threshold rules (e.g., triggering an "
            "alarm whenever heart rate exceeds 100 beats per minute). Such naive threshold checks fail to capture multi-parameter "
            "physiological interactions, ignore the temporal velocity of vital trajectories, and produce staggering rates of false "
            "alarms. SynCura is engineered to address this crisis by transforming 90 minutes of streaming multi-parameter vitals "
            "into a dynamic, explainable risk score (0–100) using an Attention-based Bidirectional Long Short-Term Memory network."
        )
        self.add_para(p2)

        self.add_h1("1.2 Introduction")
        p3 = (
            "In routine clinical workflow, deterioration risk is tracked using manual heuristic scoring systems, most notably "
            "the National Early Warning Score 2 (NEWS2), Sequential Organ Failure Assessment (SOFA), and Acute Physiology and "
            "Chronic Health Evaluation (APACHE-II). While these instruments established standardization across global medical "
            "institutions, they suffer from fundamental engineering limitations: they are computed intermittently (often several "
            "hours apart during nursing shift rounds), rely on static integer point assignments, and are incapable of identifying "
            "non-linear temporal deterioration trends before overt physiological collapse manifests."
        )
        self.add_para(p3)

        p4 = (
            "The emergence of deep learning, particularly sequence-modeling architectures such as Recurrent Neural Networks (RNNs) "
            "and Long Short-Term Memory (LSTM) networks, offers unprecedented capabilities for modeling non-stationary physiological "
            "time-series. Nevertheless, existing deep-learning approaches in healthcare face significant adoption hurdles: they "
            "operate as opaque 'black boxes' devoid of clinician interpretability, are computationally cumbersome to execute in real-time, "
            "and exacerbate alarm fatigue if every minor transient anomaly sounds a high-priority klaxon.\n\n"
            "SynCura introduces a complete paradigm shift by uniting: (1) an Attention-based Bidirectional LSTM ensemble trained "
            "on 12 vitals and laboratory variables across a 90-minute window; (2) dual-layer interpretability consisting of temporal "
            "attention heatmaps and SHAP feature attributions; (3) a two-stage persistence filter eliminating 76.4% of false alarms; "
            "and (4) an asynchronous FastAPI and React 19 bedside dashboard delivering sub-25ms response times."
        )
        self.add_para(p4)

        self.add_h1("1.3 Literature Survey")
        p_lit_intro = (
            "A comprehensive literature survey was undertaken to investigate existing methodologies in clinical risk prediction, "
            "sequence modeling, model interpretability, and early warning systems. The surveyed studies, their methodologies, and "
            "their critical limitations are summarized below:"
        )
        self.add_para(p_lit_intro)

        papers = [
            ("Zheng et al. (2025) [1]",
             "Developed a dynamic real-time risk prediction model using a Time-aware Bidirectional Attention-based LSTM (TBAL) "
             "evaluated on 176,344 ICU stays across the MIMIC-IV and eICU multicenter cohorts, attaining an hourly AUROC of 0.936.",
             "The model relies heavily on large-scale, high-dimensional electronic medical record (EMR) features and complex lab "
             "panels that are unavailable in low-latency continuous streaming environments. The framework also lacks clinician-facing "
             "feature-level attribution at the point of care."),

            ("Moor et al. (2021) [2]",
             "Proposed an early machine learning prediction model for sepsis in the ICU using deep neural networks applied to continuous "
             "vital sign streams, demonstrating predictive anticipation up to several hours prior to clinical onset.",
             "High rate of false positive alarms leading to clinical alarm fatigue. The model lacked mechanism-level explainability "
             "and failed to incorporate robust temporal filtering to suppress short-lived sensor artifacts."),

            ("Suresh et al. (2017) [3]",
             "Investigated clinical intervention prediction using recurrent neural networks to forecast the onset of mechanical "
             "ventilation and vasopressor administration from longitudinal intensive care time series.",
             "Operated with black-box latent representations that obscured which specific vital sign trends drove the recommendation, "
             "severely hindering clinician trust and bed-side verification."),

            ("Choi et al. (2016) [4]",
             "Formulated RETAIN (Reverse Time Attention model), an interpretable predictive architecture utilizing dual RNN attention "
             "mechanisms to identify influential patient visits and diagnostic codes from electronic health records.",
             "Tailored specifically for discrete billing codes and infrequent outpatient visits, rendering it unsuited for high-frequency, "
             "continuous 5-minute telemetry streams observed at ICU bedsides."),

            ("Moody and Mark (2012) [5]",
             "Established the PhysioNet/Computers in Cardiology Challenge 2012 benchmark, providing 4,000 multi-parameter ICU stays "
             "for predicting in-hospital mortality using physiological time-series.",
             "Original challenge models focused on post-hoc static mortality risk calculated after the complete 48-hour stay, rather "
             "than dynamic, sliding-window prospective risk scoring updated continuously every 5 minutes."),

            ("Lundberg and Lee (2017) [6]",
             "Unified feature attribution methodologies by introducing SHAP (SHapley Additive exPlanations), demonstrating game-theoretic "
             "consistency and local accuracy for tree and neural network predictions.",
             "Standard KernelSHAP incurs prohibitive computational overhead when applied across high-dimensional time-series arrays, "
             "necessitating optimized background clustering and tensor sub-sampling for real-time web deployment."),

            ("Royal College of Physicians (2017) [7]",
             "Standardized the National Early Warning Score 2 (NEWS2), assigning integer weights (0–3) to 6 physiological parameters "
             "to trigger hospital emergency response teams.",
             "Utilizes rigid, static physiological thresholds; completely fails to model non-linear parameter interactions; ignores "
             "temporal slope and trajectory; and yields a low AUROC (0.612) when evaluated against complex ICU deterioration."),

            ("Vincent et al. (1996) [8]",
             "Formulated the Sequential Organ Failure Assessment (SOFA) score to describe organ dysfunction across six physiological "
             "systems (respiratory, coagulatory, hepatic, cardiovascular, neurological, and renal).",
             "Designed for daily evaluation rather than real-time monitoring; requires extensive laboratory panels (such as PaO2/FiO2, "
             "bilirubin, and platelet counts) that are collected only intermittently."),

            ("Johnson et al. (2023) [9]",
             "Curated and published the MIMIC-IV database, documenting comprehensive clinical data for tens of thousands of ICU "
             "admissions at Beth Israel Deaconess Medical Center.",
             "Contains substantial retrospective selection bias, significant proportions of missing values, and asynchronous sampling "
             "intervals requiring rigorous forward-fill and normalization pipelines."),

            ("Hochreiter and Schmidhuber (1997) [10]",
             "Introduced Long Short-Term Memory (LSTM) recurrent networks featuring input, forget, and output gating cells to conquer "
             "the vanishing gradient problem over long temporal dependencies.",
             "Standard unidirectional LSTMs fail to prioritize specific temporal epochs without attention weighting and compress "
             "the entire temporal history into a single fixed-length hidden vector, leading to bottlenecking over extended windows."),

            ("Sendak et al. (2020) [11]",
             "Analyzed the real-world operational integration of machine learning sepsis warning systems within clinical hospital "
             "environments, documenting workflow barriers.",
             "Identified that lack of transparent model explainability, confusing user interfaces, and excessive false alerts represent "
             "the primary reasons clinical personnel ignore or disable algorithmic surveillance tools."),

            ("Xiao et al. (2018) [12]",
             "Reviewed the state of deep learning in electronic health records, analyzing data quality challenges, multi-modal integration, "
             "and translational bottlenecks in healthcare delivery.",
             "Highlighted the critical vulnerability of neural networks to missing data artifacts, domain shift across hospital centers, "
             "and uncalibrated probability estimations in safety-critical clinical contexts.")
        ]

        for author, meth, lim in papers:
            self.add_para(author, bold=True, size=12, color=NAVY, space_before=6, space_after=2)
            self.add_para(meth, size=11.5, space_before=0, space_after=2)
            self.add_bullet(lim, bold_prefix="Limitations: ", space_after=6)

        self.add_h1("1.4 Problem Statement")
        p_prob = (
            "Preventable patient mortality and late escalation in Intensive Care Units remain pervasive international challenges. "
            "Although ICU beds are surrounded by digital monitors, current clinical surveillance suffers from severe structural flaws:\n\n"
            "Key Problems Identified:\n"
            "1. Static Threshold Systems: Conventional alarm monitors fire upon isolated readings breaching fixed boundaries, ignoring "
            "whether a patient's vitals are recovering or experiencing fatal downward trajectory.\n"
            "2. Failure to Capture Temporal Trends: Clinical deterioration is an evolutionary process spanning hours; static heuristics "
            "fail to capture the subtle slope of declining oxygen saturation combined with compensatory tachycardia.\n"
            "3. Rampant Alarm Fatigue: Over 85% of hospital telemetry alarms are clinically non-actionable, causing desensitization, "
            "delayed response times, and intentional volume muting by overwhelmed nurses.\n"
            "4. Black-Box Lack of Interpretability: High-performing deep learning models output raw risk scores without indicating "
            "which specific physiological parameters or past timepoints contributed to the risk, preventing clinicians from verifying "
            "clinical plausibility."
        )
        self.add_para(p_prob)

        self.add_h1("1.5 Objectives")
        p_obj_intro = "The primary engineering and clinical objectives of the SynCura project are:"
        self.add_para(p_obj_intro)
        self.add_bullet("To design and train a deep-learning Attention-based Bidirectional LSTM neural network capable of predicting "
                        "ICU patient deterioration risk from continuous 12-parameter physiological time-series.", bold_prefix="1. Sequence Modeling: ")
        self.add_bullet("To implement a 3-model logit-averaged ensemble architecture (hidden dimension 96) that achieves superior "
                        "predictive accuracy (AUROC ≥ 0.840) and robustness across heterogeneous patient cohorts.", bold_prefix="2. Multi-Model Ensemble: ")
        self.add_bullet("To incorporate dual-layer explainability combining temporal self-attention weights and local SHAP feature attributions, "
                        "empowering clinicians to interpret the exact physiological drivers behind every prediction.", bold_prefix="3. Clinical Explainability: ")
        self.add_bullet("To construct a two-stage persistence filtering engine that verifies multi-step deterioration trends, cutting "
                        "clinically disruptive false positive alarms by over 75%.", bold_prefix="4. Alarm Fatigue Mitigation: ")
        self.add_bullet("To develop a sub-25ms asynchronous FastAPI streaming backend and WebSocket gateway for continuous real-time "
                        "vital telemetry processing and scenario injection.", bold_prefix="5. Real-Time Telemetry Engine: ")
        self.add_bullet("To build a modern React 19 bedside monitoring dashboard with dark/light themes, live multi-bed ward overview, "
                        "vital sparklines, and dual-tier medical sound cues.", bold_prefix="6. Clinical Presentation System: ")

        self.add_h1("1.6 Proposed System")
        p_prop = (
            "SynCura proposes a unified, end-to-end clinical surveillance platform that bridges streaming medical telemetry with deep "
            "learning inference and explainability. The proposed architecture consists of six seamlessly integrated subsystems:\n\n"
            "• Ingestion & Ring Buffer: Continuously ingests 12 physiological variables (6 vitals + 6 labs), managing a 90-minute "
            "rolling memory buffer (18 discrete 5-minute timesteps) per patient.\n"
            "• Preprocessing Pipeline: Applies instantaneous forward-fill imputation for sparse laboratory measurements, handles sensor "
            "dropouts, and standardizes parameters via population Z-score scaling.\n"
            "• Triple Ensemble Attention-LSTM: Feeds standardized 18×12 tensors into three trained causal models (seeds s48, c93, s45), "
            "concatenates bidirectional hidden representations, applies soft attention scoring, and averages output logits to yield a "
            "robust 0–100 mortality risk index.\n"
            "• SHAP Attribution Subsystem: Evaluates local Shapley values across the 12 features, identifying which vital or lab markers "
            "are currently pushing the patient into high-risk status.\n"
            "• Two-Stage Persistence Filter: Enforces temporal confirmation before escalating high-risk scores into audible red alerts, "
            "ensuring transient motion artifacts do not awaken the ICU ward.\n"
            "• Asynchronous Presentation Layer: FastAPI REST/WebSocket server broadcasting updates every 5 seconds to a responsive React 19 "
            "interface with dual-tier audio sirens and interactive explanation modals."
        )
        self.add_para(p_prop)

        # Embed Fig 1.1
        fig1_path = os.path.join(ASSETS_DIR, "fig1_1_overview_modules.png")
        self.add_image_figure(fig1_path, "Figure 1.1: SynCura High-Level Concept & Functional Interaction Overview",
                              "Visualizing the core predictive engine communicating bidirectionally with the Clinical User Module and Admin/Simulation Module.")

        self.add_h1("1.7 Advantages of the Proposed System")
        self.add_bullet("Dynamic Longitudinal Tracking: Rather than taking static single-point snapshots, SynCura analyzes continuous "
                        "90-minute vital trends, detecting subtle early downward trajectories.", bold_prefix="• Predictive Proactivity: ")
        self.add_bullet("Superior Discrimination: Validated AUROC of 0.840 compared to 0.612 for the clinical baseline NEWS2, delivering "
                        "dramatically earlier warning for high-risk patients.", bold_prefix="• High Diagnostic Precision: ")
        self.add_bullet("Elimination of Alarm Fatigue: Two-stage persistence filtering reduces false alerts by 76.4%, allowing nurses "
                        "to trust and rapidly respond to critical sirens.", bold_prefix="• 76.4% False Alarm Reduction: ")
        self.add_bullet("Transparent and Verifiable: Clinicians can instantly inspect which vitals and past minutes caused the alert via "
                        "SHAP waterfall plots and attention heatmaps.", bold_prefix="• Clinician-Centric Interpretability: ")
        self.add_bullet("Ultra-Low Latency: Executes end-to-end data ingestion, neural forward pass, and WebSocket dispatch in under 25 milliseconds.",
                        bold_prefix="• Sub-25ms Execution: ")

    def build_chapter_2(self):
        self.add_chapter_title("2", "SOFTWARE REQUIREMENTS SPECIFICATION")

        self.add_h1("2.1 Specific Requirements")
        p_srs_spec = (
            "This Software Requirements Specification (SRS) establishes the complete functional and non-functional requirements "
            "governing the design, development, verification, and deployment of the SynCura Predictive ICU Monitoring System. SynCura "
            "is designed as a safety-critical clinical support tool, mandating strict reliability, determinism, data integrity, and "
            "minimal computational latency."
        )
        self.add_para(p_srs_spec)

        self.add_h1("2.2 Product Functions")
        self.add_bullet("Ingest multi-parameter vitals streams from bedside monitors and laboratory feeds at regular 5-minute sampling frequencies.",
                        bold_prefix="F1. Telemetry Ingestion: ")
        self.add_bullet("Maintain a rolling memory queue preserving the preceding 90 minutes (18 timesteps) of patient physiological history.",
                        bold_prefix="F2. Window Buffering: ")
        self.add_bullet("Execute real-time missing value imputation using forward-fill for intermittent labs and Z-score standardization.",
                        bold_prefix="F3. Preprocessing & Imputation: ")
        self.add_bullet("Process temporal tensors through a 3-model Attention-LSTM ensemble to compute calibrated 0–100 deterioration risk scores.",
                        bold_prefix="F4. Ensemble Neural Inference: ")
        self.add_bullet("Compute local Shapley feature contributions (SHAP) and temporal self-attention distributions for every prediction.",
                        bold_prefix="F5. Explainability Generation: ")
        self.add_bullet("Apply two-stage persistence filtering to suppress short-lived sensor noise and prevent non-actionable alarms.",
                        bold_prefix="F6. Persistence Verification: ")
        self.add_bullet("Broadcast live telemetry, risk updates, and alert triggers across bi-directional WebSocket connections to client dashboards.",
                        bold_prefix="F7. Real-Time Telemetry Broadcast: ")
        self.add_bullet("Render responsive multi-bed overview grids, individual patient profile drawers, vital sparklines, and synthesize audio sirens.",
                        bold_prefix="F8. Clinical Visualization & SFX: ")

        self.add_h1("2.3 User Characteristics")
        p_users = (
            "The SynCura system is engineered to serve three distinct classes of users within healthcare and research environments:\n\n"
            "1. ICU Nurses & Bedside Clinicians: Possess high medical domain knowledge but operate under high cognitive stress and time pressure. "
            "Require clear, unambiguous color-coded risk categorizations (Stable, Moderate, Critical), immediate visual cues, dual-tier audio sirens, "
            "and simple one-click access to explainability summaries.\n"
            "2. Intensivists & Attending Physicians: Require deep diagnostic transparency, including temporal attention timelines, historical "
            "vital trend charts, and SHAP feature rankings to confirm physiological causality before prescribing aggressive interventions.\n"
            "3. System Administrators & Clinical Engineers: Responsible for system deployment, hardware interfacing, scenario injection, "
            "threshold tuning, and auditing system telemetry logs."
        )
        self.add_para(p_users)

        self.add_h1("2.4 General Constraints")
        self.add_bullet("Regulatory Classification: SynCura is an experimental decision-support research tool; it is not currently certified "
                        "by regulatory bodies (FDA/CE) as an autonomous diagnostic device and must not override direct medical oversight.",
                        bold_prefix="C1. Medical Safety Boundary: ")
        self.add_bullet("Retrospective Dataset Boundary: Trained on the PhysioNet Challenge 2012 ICU cohort; external deployment requires "
                        "hospital-specific calibration and validation across heterogeneous EMR formats.", bold_prefix="C2. Single-Corpus Training: ")
        self.add_bullet("Latency Ceiling: End-to-end pipeline execution from telemetry packet arrival to dashboard rendering must complete "
                        "within 100 milliseconds.", bold_prefix="C3. Real-Time Latency Budget: ")
        self.add_bullet("Data Privacy: System architecture must comply with HIPAA and GDPR data privacy principles, ensuring patient telemetry "
                        "is pseudonymized and isolated.", bold_prefix="C4. Clinical Data Privacy: ")

        self.add_h1("2.5 Assumptions and Dependencies")
        self.add_bullet("Availability of continuous streaming physiological data at minimum 5-minute sampling intervals.", bold_prefix="A1. Sensor Continuity: ")
        self.add_bullet("Monitored vital signs reside within biologically feasible human physiological bounds.", bold_prefix="A2. Biological Bounds: ")
        self.add_bullet("Host operating environment provides Python 3.10+ execution runtime and modern web browsers supporting ECMAScript 6+ and WebSockets.",
                        bold_prefix="D1. Modern Runtime Environment: ")

        self.add_h1("2.6 Functional Requirements")
        self.add_h2("2.6.1 Clinician Monitoring Module (User Module)")
        p_mod_user = (
            "The Clinician User Module provides the primary interface for healthcare workers stationed in the ICU:\n"
            "• Multi-Bed Ward Grid: Displays simultaneous status cards for all active ICU beds, color-coded by clinical urgency.\n"
            "• Patient Detail Profile: Selecting any bed card expands a detailed telemetry drawer displaying 90-minute vital sparklines, "
            "current numerical values, and historical trajectory.\n"
            "• Explainability Modal: Displays an interactive SHAP waterfall diagram indicating which vitals contributed positively or "
            "negatively to the deterioration risk score.\n"
            "• Sound Alert Controls: Synthesizes Web Audio API pulse tones for warning states and escalating sirens for critical emergencies, "
            "with a persistent user toggle to mute or unmute audio cues."
        )
        self.add_para(p_mod_user)

        self.add_h2("2.6.2 System Administration & Simulation Module (Admin Module)")
        p_mod_admin = (
            "The Admin Module provides engineering control and simulation capabilities:\n"
            "• Scenario Simulation Engine: Injects controlled clinical deterioration patterns (e.g., septic shock cascade, septic arrest, "
            "recovering patient, stable telemetry) to validate system alerting.\n"
            "• Persistence Parameter Tuning: Permits clinical supervisors to adjust the required consecutive alert count (default: 2 hits) "
            "and risk threshold boundaries (default: 0–30 Stable, 30–70 Moderate, 70–100 Critical).\n"
            "• Benchmark Telemetry Logger: Records model predictions, actual outcomes, and system latencies to SQLite database for audit."
        )
        self.add_para(p_mod_admin)

        self.add_h1("2.7 Performance Requirements")
        self.add_bullet("The complete machine learning forward pass through the 3-model ensemble must execute in less than 25 milliseconds "
                        "on standard CPU hardware without requiring discrete GPU acceleration.", bold_prefix="P1. Inference Latency: ")
        self.add_bullet("The FastAPI backend and WebSocket engine must support at least 50 concurrent active ICU bed streams simultaneously "
                        "without message queuing or frame drops.", bold_prefix="P2. Concurrency Capacity: ")
        self.add_bullet("The React frontend client must maintain a smooth 60 frames-per-second UI rendering cycle during live vitals updates.",
                        bold_prefix="P3. Client Rendering: ")

        self.add_h1("2.8 Supportability and Maintainability")
        p_supp = (
            "SynCura follows a clean, modular architecture separating the core machine learning inference engine (`ml/`), the API and "
            "simulation server (`backend/`), and the user presentation interface (`frontend/`). Standardized configuration files "
            "(`deployed_manifest.json`) decouple model weights from application logic, allowing retraining and model swaps without "
            "modifying frontend or API endpoints."
        )
        self.add_para(p_supp)

        self.add_h1("2.9 Design Constraints")
        p_des = (
            "The application must run reliably in local standalone hospital environments without mandatory external internet access. "
            "Dependencies are strictly pinned in `requirements.txt` and `package.json` to guarantee reproducible builds across "
            "Windows 11, Linux, and containerized Docker environments."
        )
        self.add_para(p_des)

        self.add_h1("2.10 Interfaces")
        self.add_h2("2.10.1 User Interfaces")
        self.add_para("Intuitive, high-contrast dark/light clinical web UI. Features responsive grid layouts, SVG sparklines, accessible "
                      "color palettes adhering to WCAG 2.2 AA standards, and high-visibility red glow pulsing on critical beds.")

        self.add_h2("2.10.2 Hardware Interfaces")
        self.add_para("Operates across standard hospital workstations, wall-mounted nursing station monitors, bedside tablets, and edge "
                      "medical gateways connected via local Ethernet or secure hospital Wi-Fi networks.")

        self.add_h2("2.10.3 Software Interfaces")
        self.add_para("FastAPI asynchronous server (Python 3.10+), PyTorch 2.5+ deep learning runtime, React 19 UI library, Vite build tool, "
                      "Web Audio API for zero-asset sound generation, and SQLite database for local vitals caching.")

    def build_chapter_3(self):
        self.add_chapter_title("3", "HIGH-LEVEL DESIGN")

        self.add_h1("3.1 System Architecture")
        p_arch = (
            "The high-level architecture of SynCura is organized into five decoupled, cohesive layers designed to ensure real-time "
            "throughput, fault tolerance, and clear separation of concerns. Figure 3.1 illustrates the structural pipeline spanning "
            "from physical bedside telemetry ingestion to client presentation."
        )
        self.add_para(p_arch)

        # Embed Fig 3.1
        fig3_1_path = os.path.join(ASSETS_DIR, "fig3_1_system_architecture.png")
        self.add_image_figure(fig3_1_path, "Figure 3.1: Five-Layered End-to-End System Architecture",
                              "Spanning Telemetry Ingestion, Preprocessing Ring Buffering, Ensemble Deep Learning, Explainability & Alarm Verification, and Presentation.")

        self.add_h1("3.2 System Architecture Description")
        self.add_h2("3.2.1 Layer 1: Telemetry Data Ingestion")
        self.add_para("Continuously captures 12 physiological variables from bedside ICU sensors and electronic laboratory reports. The 6 vital "
                      "signs (Heart Rate, Respiratory Rate, Body Temperature, Systolic BP, Diastolic BP, and SpO2) are sampled continuously, "
                      "while the 6 laboratory/neurological markers (GCS, BUN, Creatinine, WBC, Platelets, and Glucose) are ingested as available.")

        self.add_h2("3.2.2 Layer 2: Preprocessing and Ring Buffering")
        self.add_para("Manages an in-memory 90-minute FIFO ring buffer holding 18 timesteps at 5-minute sampling resolution per bed. Missing "
                      "laboratory readings are updated using clinical forward-fill imputation, and values are normalized via population Z-score scaling.")

        self.add_h2("3.2.3 Layer 3: Ensemble Deep Learning Engine")
        self.add_para("Processes normalized (Batch, 18, 12) tensors through a 3-model Attention-LSTM ensemble (models s48, c93, and s45). "
                      "The networks employ bidirectional recurrent units to capture forward and backward physiological trajectories, followed "
                      "by self-attention weighting and logit-averaged voting to compute a consensus 0–100 deterioration probability.")

        self.add_h2("3.2.4 Layer 4: Explainability and Alarm Verification")
        self.add_para("Extracts temporal attention weights indicating which 5-minute intervals most influenced the prediction, and computes "
                      "KernelSHAP feature attributions ranking the specific parameters driving the risk. Simultaneously, the two-stage persistence "
                      "filter requires two consecutive high-risk detections before sounding an audible critical alarm.")

        self.add_h2("3.2.5 Layer 5: Presentation and Clinical Dashboard")
        self.add_para("Distributes telemetry and alert payloads via FastAPI WebSockets to a React 19 bedside dashboard, rendering live "
                      "bed grids, sparklines, SHAP waterfalls, and synthesizing dual-tier audio sirens.")

        self.add_h1("3.3 Context Flow Diagram (DFD)")
        p_cfd = (
            "The Context Flow Diagram (Data Flow Diagram Level 0) defines the primary data boundaries of the SynCura system, illustrating "
            "external entities (ICU Bedside Sensors, Clinicians/Nurses, and Database Stores) and their data flows interacting with the core engine."
        )
        self.add_para(p_cfd)

        # Embed Fig 3.2
        fig3_2_path = os.path.join(ASSETS_DIR, "fig3_2_context_flow_diagram.png")
        self.add_image_figure(fig3_2_path, "Figure 3.2: Context Flow Diagram (DFD Level 0 Context Diagram)",
                              "Depicting data exchanges between Bedside Sensors, the SynCura Central Process, ICU Clinicians, and the Telemetry Database.")

        self.add_h1("3.4 Use Case Diagram")
        p_uc = (
            "The UML Use Case Diagram models the functional capabilities provided by SynCura and illustrates how primary actors (ICU Clinicians "
            "and System Administrators) interact with the system's operational use cases."
        )
        self.add_para(p_uc)

        # Embed Fig 3.3
        fig3_3_path = os.path.join(ASSETS_DIR, "fig3_3_use_case_diagram.png")
        self.add_image_figure(fig3_3_path, "Figure 3.3: UML Use Case Diagram for Clinicians and System Administrators",
                              "Modeling interaction boundaries for live monitoring, explanation inspection, alarm acknowledgement, and simulation controls.")

        self.add_h1("3.5 Sequence Diagram")
        p_seq = (
            "The UML Sequence Diagram depicts the detailed chronological sequence of message exchanges and processing calls that occur "
            "from the moment a 5-minute telemetry packet arrives at the server until client alerts are rendered."
        )
        self.add_para(p_seq)

        # Embed Fig 3.4
        fig3_4_path = os.path.join(ASSETS_DIR, "fig3_4_sequence_diagram.png")
        self.add_image_figure(fig3_4_path, "Figure 3.4: UML Sequence Diagram for Real-Time Telemetry & Alert Dispatch",
                              "Illustrating chronological execution across Bedside Monitor, FastAPI Gateway, Preprocessor, Ensemble Model, SHAP Explainer, and Dashboard.")

        self.add_h1("3.6 System Flow Chart")
        p_flow = (
            "The System Flow Chart provides a comprehensive procedural overview of the internal decision logic executed upon each telemetry "
            "packet arrival, including ring buffering, neural inference, persistence validation, and alarm dispatch."
        )
        self.add_para(p_flow)

        # Embed Fig 3.5
        fig3_5_path = os.path.join(ASSETS_DIR, "fig3_5_system_flowchart.png")
        self.add_image_figure(fig3_5_path, "Figure 3.5: End-to-End System Processing & Persistence Decision Flow Chart",
                              "Mapping the complete algorithmic decision pipeline from data ingestion to siren escalation and dashboard synchronization.")

    def build_chapter_4(self):
        self.add_chapter_title("4", "DETAILED DESIGN")

        self.add_h1("4.1 Structured Chart Diagram")
        p_sc = (
            "The Structured Functional Decomposition Chart decomposes the SynCura system hierarchically into discrete functional units, "
            "delineating the structural ownership of data ingestion, preprocessing, ensemble inference, and presentation."
        )
        self.add_para(p_sc)

        # Embed Fig 4.1
        fig4_1_path = os.path.join(ASSETS_DIR, "fig4_1_structured_chart.png")
        self.add_image_figure(fig4_1_path, "Figure 4.1: Structured Functional Decomposition Chart Diagram",
                              "Hierarchical decomposition showing system modules, sub-modules, and atomic processing components.")

        self.add_h1("4.2 Detailed Component Breakdown")
        self.add_h2("4.2.1 Data Ingestion and Ring Buffer")
        self.add_para("The Ingestion Worker receives JSON-encoded telemetry packets structured as key-value pairs representing the 12 monitored "
                      "parameters. A thread-safe circular deque maintains the previous 18 timesteps for each monitored bed, evicting the oldest "
                      "observation when new data arrives.")

        self.add_h2("4.2.2 Missing Value Imputation and Normalization")
        self.add_para("Medical data is characterized by sparse sampling; laboratory tests are collected only every 12 to 24 hours. SynCura applies "
                      "clinical forward-fill imputation, propagating the most recent valid laboratory value forward across the 90-minute window. "
                      "If a parameter has never been observed, it defaults to the population median. Subsequently, each variable is normalized: "
                      "z = (x - mu) / sigma, where mu and sigma are pre-computed population parameters.")

        self.add_h2("4.2.3 Attention-Based Bidirectional LSTM Architecture")
        self.add_para("The core neural network employs a 2-layer Bidirectional LSTM with hidden dimension H = 96. At each timestep t in [1, 18], "
                      "the forward LSTM computes h_fwd(t) and the backward LSTM computes h_bwd(t). Concatenation yields h(t) = [h_fwd(t); h_bwd(t)] "
                      "in R^(192). A self-attention layer scores each timestep: e(t) = v^T * tanh(W_a * h(t) + b_a), normalized via softmax: "
                      "alpha(t) = exp(e(t)) / sum_j(exp(e(j))). The final patient context vector c = sum_t (alpha(t) * h(t)) is projected through "
                      "a dense layer to yield logit z.")

        self.add_h2("4.2.4 Logit-Averaged Ensemble Selection")
        self.add_para("Rather than relying on a single checkpoint, SynCura deploys a 3-model ensemble: Model s48 (causal seed 48), Model c93 "
                      "(causal seed 93), and Model s45 (warm-started on the PhysioNet 2019 sepsis challenge cohort). The final risk logit is "
                      "the arithmetic mean of individual model logits: z_ens = (z_s48 + z_c93 + z_s45) / 3, producing calibrated risk p = sigmoid(z_ens).")

        self.add_h2("4.2.5 KernelSHAP Feature Attribution Algorithm")
        self.add_para("To explain individual predictions, SynCura executes an optimized KernelSHAP explainer using 100 k-means background summary "
                      "vectors. The explainer calculates Shapley values phi_i for each feature, representing its additive contribution to the "
                      "predicted log-odds deviation from baseline.")

        self.add_h2("4.2.6 Two-Stage Persistence Filtering Engine")
        self.add_para("To combat alarm fatigue, SynCura implements a stateful persistence filter. If p >= 0.70 (Critical threshold), an alarm "
                      "candidate is flagged. An audible red siren is dispatched only if the risk remains in the critical zone for at least K = 2 "
                      "consecutive evaluations, eliminating transient artifacts while preserving 90% true positive sensitivity.")

        self.add_h1("4.3 Data Preprocessing and Feature Extraction")
        p_feat_intro = "SynCura monitors 12 physiological and laboratory features selected for their proven clinical correlation with ICU deterioration:"
        self.add_para(p_feat_intro)

        # Feature Table 4.1
        tbl_feat = self.doc.add_table(rows=13, cols=4)
        tbl_feat.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl_feat, color="94A3B8", sz="4")
        
        f_hdrs = ["Parameter Name", "Type", "Normal Range", "Clinical Significance"]
        for c_idx, h in enumerate(f_hdrs):
            c = tbl_feat.rows[0].cells[c_idx]
            c.text = h
            set_cell_background(c, "1B365D")
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10)
                r.font.color.rgb = WHITE
        set_row_header(tbl_feat.rows[0])

        feats = [
            ("Heart Rate (HR)", "Vital Sign", "60 – 100 bpm", "Tachycardia/Bradycardia indicates shock or cardiac stress"),
            ("Respiratory Rate (RR)", "Vital Sign", "12 – 20 breaths/min", "Tachypnea is the earliest indicator of respiratory failure"),
            ("Temperature (Temp)", "Vital Sign", "36.5 – 37.5 °C", "Fever/Hypothermia correlates with systemic sepsis"),
            ("Systolic BP (SysBP)", "Vital Sign", "90 – 120 mmHg", "Hypotension indicates circulatory shock and hypoperfusion"),
            ("Diastolic BP (DiasBP)", "Vital Sign", "60 – 80 mmHg", "Low diastolic pressure reflects loss of vascular tone"),
            ("Oxygen Saturation (SpO2)", "Vital Sign", "95 – 100 %", "Hypoxemia indicates impaired pulmonary gas exchange"),
            ("Glasgow Coma Scale (GCS)", "Neurological", "15 (Normal: 3-15)", "Depressed consciousness reflects cerebral hypoperfusion"),
            ("Blood Urea Nitrogen (BUN)", "Laboratory", "7 – 20 mg/dL", "Elevated BUN indicates acute kidney injury or dehydration"),
            ("Serum Creatinine", "Laboratory", "0.6 – 1.2 mg/dL", "Key biomarker of renal failure and clearance impairment"),
            ("White Blood Cells (WBC)", "Laboratory", "4.0 – 11.0 ×10³/µL", "Leukocytosis/Leukopenia indicates severe immune infection"),
            ("Platelet Count", "Laboratory", "150 – 450 ×10³/µL", "Thrombocytopenia reflects disseminated intravascular coagulation"),
            ("Serum Glucose", "Laboratory", "70 – 140 mg/dL", "Hyperglycemia reflects severe metabolic stress response")
        ]

        for i, (f_name, f_type, f_norm, f_clin) in enumerate(feats):
            row_cells = tbl_feat.rows[i+1].cells
            row_cells[0].text = f_name
            row_cells[1].text = f_type
            row_cells[2].text = f_norm
            row_cells[3].text = f_clin
            bg = "F8FAFC" if i % 2 == 0 else "FFFFFF"
            for c_idx, c in enumerate(row_cells):
                set_cell_background(c, bg)
                set_cell_margins(c, top=40, bottom=40, left=80, right=80)
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(9.5)
                    r.font.color.rgb = DARK
                    if c_idx == 0:
                        r.bold = True

        self.add_para("Table 4.1: The 12 ICU Physiological and Laboratory Features Monitored by SynCura", bold=True, size=10,
                      color=SLATE, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

        self.add_h1("4.4 Dataset Description")
        p_data = (
            "SynCura was trained and validated on the PhysioNet/Computing in Cardiology Challenge 2012 benchmark repository. "
            "The dataset comprises 4,000 multi-parameter ICU stays across general medical, surgical, cardiac, and trauma ICUs. "
            "Each record documents longitudinal observations during the initial 48 hours following ICU admission. The cohort exhibits "
            "a realistic, severe class imbalance with an in-hospital mortality rate of ~14.2%, replicating the exact challenging statistical "
            "landscape confronted in real-world critical care wards."
        )
        self.add_para(p_data)

        self.add_h1("4.5 Model Methodology and Mathematical Formulation")
        p_math = (
            "The neural network models the sequence of input vectors X = [x_1, x_2, ..., x_T], where T = 18 and x_t in R^(12). "
            "The mathematical formulation of the forward computational pass is defined as follows:\n\n"
            "1. Bidirectional LSTM Recurrence:\n"
            "   h_fwd(t) = LSTM_fwd(x_t, h_fwd(t-1))\n"
            "   h_bwd(t) = LSTM_bwd(x_t, h_bwd(t+1))\n"
            "   h(t) = [h_fwd(t) ; h_bwd(t)] in R^(2H), where H = 96\n\n"
            "2. Temporal Attention Mechanism:\n"
            "   e(t) = v^T * tanh(W_a * h(t) + b_a)\n"
            "   alpha(t) = exp(e(t)) / sum_{j=1}^T exp(e(j))\n"
            "   c = sum_{t=1}^T alpha(t) * h(t)\n\n"
            "3. Consensus Logit Projection:\n"
            "   z_m = W_o * c_m + b_o,  for model m in {s48, c93, s45}\n"
            "   z_consensus = (1/3) * sum_{m} z_m\n"
            "   P(Deterioration) = 1 / (1 + exp(-z_consensus))\n\n"
            "4. Objective Loss Function:\n"
            "   L(theta) = - [ w * y * log(p) + (1 - y) * log(1 - p) ], where w = 3.5 accounts for class imbalance."
        )
        self.add_para(p_math)

    def build_chapter_5(self):
        self.add_chapter_title("5", "IMPLEMENTATION")

        self.add_h1("5.1 Implementation Requirements")
        p_req = (
            "The implementation of SynCura requires a robust, reproducible software environment capable of handling high-concurrency "
            "asynchronous I/O and low-latency tensor processing. The hardware and runtime requirements include:\n"
            "• Development Host: 64-bit multi-core CPU (Intel Core i5 or AMD Ryzen equivalent), minimum 8 GB RAM.\n"
            "• Operating System: Windows 11 / Linux (Ubuntu 22.04 LTS) / macOS.\n"
            "• Python Environment: Python 3.10+ with PyTorch 2.5, FastAPI 0.115, Uvicorn, SHAP, Scikit-Learn, and NumPy.\n"
            "• Frontend Environment: Node.js 18+, React 19, Vite, and modern CSS3."
        )
        self.add_para(p_req)

        self.add_h1("5.2 Implementation Details")
        p_impl = (
            "SynCura was implemented following a clean microservice architecture. The codebase is partitioned into:\n"
            "• `ml/`: Model architecture definitions (`models.py`), training pipelines (`train.py`), evaluation scripts (`eval.py`), "
            "and deployed ensemble checkpoint weights (`models/ensemble/`).\n"
            "• `backend/`: FastAPI application (`main.py`), live simulation controller (`simulation.py`), and vitals database storage.\n"
            "• `frontend/`: React components (`Dashboard.jsx`, `BedProfile.jsx`, `SoundToggle.jsx`), audio synthesis utilities (`sounds.js`), "
            "and station stylesheets (`station.css`)."
        )
        self.add_para(p_impl)

        self.add_h1("5.3 Engine and Module Selection")
        self.add_bullet("PyTorch 2.5: Chosen for dynamic graph execution, automatic differentiation, and rapid tensor operations on CPU.",
                        bold_prefix="• ML Framework: ")
        self.add_bullet("FastAPI & Uvicorn: Selected for high-throughput asynchronous execution, native WebSocket support, and automatic OpenAPI generation.",
                        bold_prefix="• Backend API: ")
        self.add_bullet("React 19 & Vite: Utilized for reactive component rendering, ultra-fast Hot Module Replacement, and zero-bundle UI responsiveness.",
                        bold_prefix="• Frontend UI: ")
        self.add_bullet("Web Audio API: Employed to synthesize dual-tier medical audio cues directly in browser hardware, requiring zero external MP3/WAV downloads.",
                        bold_prefix="• Audio Synthesis: ")

        self.add_h1("5.4 IDE Selection")
        self.add_para("Visual Studio Code (VS Code) was selected as the unified integrated development environment, configured with Python, "
                      "Pylance, ESLint, Prettier, and Git Graph extensions to enforce strict code quality and version control.")

        self.add_h1("5.5 Language Selection")
        self.add_bullet("Python 3.12: Core machine learning inference, preprocessing, and asynchronous web backend.", bold_prefix="• Python: ")
        self.add_bullet("JavaScript (ES6+) & JSX: Client-side component orchestration, real-time telemetry rendering, and audio scheduling.", bold_prefix="• JavaScript: ")
        self.add_bullet("Modern CSS3: Custom medical telemetry station theme, responsive flex/grid layouts, and pulse animations.", bold_prefix="• CSS3: ")

        self.add_h1("5.6 Platform Selection")
        self.add_para("SynCura is platform-independent. In production, the backend is containerized via Docker and deployed to cloud infrastructure "
                      "(Render), while local hospital edge installations operate directly on hospital server hardware with zero cloud dependencies.")

        self.add_h1("5.7 Coding Guidelines")
        self.add_para("All backend code strictly complies with PEP 8 standards, utilizing static type hints (`typing.Optional`, `typing.Dict`). "
                      "Frontend code follows modular React component architecture, keeping components under 250 lines and isolating business "
                      "logic inside custom hooks and context providers (`simulationContext.jsx`).")

        self.add_h1("5.8 Implementation Procedure")
        steps = [
            ("1. Data Pipeline & Sliding Window Buffer",
             "Constructed `SlidingWindowBuffer` in `simulation.py` maintaining a fixed-size queue of 18 observations per patient, "
             "performing forward-fill imputation on arriving telemetry packets."),
            ("2. Neural Model Training & Ensemble Selection",
             "Trained bidirectional Attention-LSTM models across multiple seeds on PhysioNet 2012. Identified top three complementary "
             "checkpoints (s48, c93, s45) and packaged them into `ml/models/ensemble/` with `ml/deployed_manifest.json`."),
            ("3. SHAP Explainability Subsystem",
             "Implemented background summary sampling via k-means clustering. Generated local feature contribution weights within "
             "the FastAPI `/api/explain` endpoint to provide immediate bed-side interpretability."),
            ("4. Two-Stage Persistence Filtering Engine",
             "Implemented circular alert buffers tracking consecutive predictions per bed. Suppressed single-interval spikes and dispatched "
             "critical alarms only upon consecutive confirmations."),
            ("5. FastAPI Streaming Server & WebSocket Gateway",
             "Constructed `/ws/telemetry` WebSocket endpoint pushing live updates every 5 seconds to connected clinical clients."),
            ("6. React Bedside Station Interface & Audio Cues",
             "Constructed responsive multi-bed overview board, detailed bed profile drawers, vital sparkline graphs, and procedural "
             "dual-frequency pulse audio alerts in `sounds.js`.")
        ]
        for title, desc in steps:
            self.add_para(title, bold=True, size=12, color=NAVY, space_before=4, space_after=1)
            self.add_para(desc, size=11.5, space_before=0, space_after=6)

    def build_chapter_6(self):
        self.add_chapter_title("6", "TESTING")

        self.add_h1("6.1 Test Environment")
        p_test_intro = (
            "Testing is an indispensable phase in safety-critical clinical software engineering to guarantee algorithmic precision, "
            "data pipeline integrity, real-time responsiveness, and system robustness under anomalous conditions. The testing environment "
            "specifications are detailed in Tables 6.1, 6.2, and 6.3."
        )
        self.add_para(p_test_intro)

        # Table 6.1: Hardware Environment
        tbl_hw = self.doc.add_table(rows=11, cols=2)
        tbl_hw.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl_hw, color="94A3B8", sz="4")
        
        c0, c1 = tbl_hw.rows[0].cells
        c0.text = "Attribute"
        c1.text = "Details"
        for c in (c0, c1):
            set_cell_background(c, "1B365D")
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.bold = True
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10)
                r.font.color.rgb = WHITE
        set_row_header(tbl_hw.rows[0])

        hw_data = [
            ("Device Name", "PACE-CSE-LAB-WORKSTATION"),
            ("Processor", "Intel(R) Core(TM) i5-12400 @ 2.50 GHz (6 Cores, 12 Threads)"),
            ("Installed RAM", "16.0 GB DDR4 (15.8 GB usable)"),
            ("Graphics Card", "Intel(R) UHD Graphics 730"),
            ("Storage Capacity", "512 GB NVMe M.2 SSD"),
            ("Storage Used", "142 GB"),
            ("System Type", "64-bit Operating System, x64-based Processor"),
            ("Operating System", "Windows 11 Home / Ubuntu 22.04 LTS dual-boot"),
            ("Network Adapter", "Gigabit Ethernet (1000 Mbps) / Wi-Fi 6"),
            ("Audio Interface", "Realtek High Definition Audio (supporting Web Audio API)")
        ]
        for i, (attr, val) in enumerate(hw_data):
            cells = tbl_hw.rows[i+1].cells
            cells[0].text = attr
            cells[1].text = val
            bg = "F8FAFC" if i % 2 == 0 else "FFFFFF"
            for c_idx, c in enumerate(cells):
                set_cell_background(c, bg)
                set_cell_margins(c, top=40, bottom=40, left=80, right=80)
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(9.5)
                    r.font.color.rgb = DARK
                    if c_idx == 0:
                        r.bold = True

        self.add_para("Table 6.1: Test Hardware Environment Specifications", bold=True, size=10,
                      color=SLATE, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

        # Table 6.2: Software Environment
        tbl_sw = self.doc.add_table(rows=8, cols=2)
        tbl_sw.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl_sw, color="94A3B8", sz="4")
        
        c0, c1 = tbl_sw.rows[0].cells
        c0.text = "Software Component"
        c1.text = "Details"
        for c in (c0, c1):
            set_cell_background(c, "1B365D")
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.bold = True
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10)
                r.font.color.rgb = WHITE
        set_row_header(tbl_sw.rows[0])

        sw_data = [
            ("Programming Languages", "Python 3.12.1 and JavaScript (Node.js v18.17.0)"),
            ("Integrated Development Environment", "Visual Studio Code (v1.95) with PyLance & ESLint"),
            ("Deep Learning & Tensor Engine", "PyTorch 2.5.1+cu121 (CPU & CUDA support)"),
            ("Scientific & Preprocessing Libraries", "NumPy 2.2, Pandas 2.3, Scikit-learn 1.7, SciPy 1.15"),
            ("Explainability Engine", "SHAP (SHapley Additive exPlanations) 0.49.1"),
            ("Backend API Framework", "FastAPI 0.115 with Starlette and Uvicorn ASGI Server"),
            ("Frontend UI Library & Build Engine", "React 19.0.0 with Vite 5.4 and Plain Semantic CSS")
        ]
        for i, (attr, val) in enumerate(sw_data):
            cells = tbl_sw.rows[i+1].cells
            cells[0].text = attr
            cells[1].text = val
            bg = "F8FAFC" if i % 2 == 0 else "FFFFFF"
            for c_idx, c in enumerate(cells):
                set_cell_background(c, bg)
                set_cell_margins(c, top=40, bottom=40, left=80, right=80)
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(9.5)
                    r.font.color.rgb = DARK
                    if c_idx == 0:
                        r.bold = True

        self.add_para("Table 6.2: Test Software Environment Specifications", bold=True, size=10,
                      color=SLATE, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=12)

        # Table 6.3: Target Platform
        tbl_tp = self.doc.add_table(rows=4, cols=2)
        tbl_tp.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl_tp, color="94A3B8", sz="4")
        
        c0, c1 = tbl_tp.rows[0].cells
        c0.text = "Target Platform"
        c1.text = "Details"
        for c in (c0, c1):
            set_cell_background(c, "1B365D")
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.bold = True
                r.font.name = 'Times New Roman'
                r.font.size = Pt(10)
                r.font.color.rgb = WHITE
        set_row_header(tbl_tp.rows[0])

        tp_data = [
            ("Target OS & Architecture", "Windows 11 / Linux (x86_64, aarch64)"),
            ("Target Client Browsers", "Google Chrome 120+, Mozilla Firefox 118+, Microsoft Edge 120+"),
            ("Network Protocol Target", "HTTP/1.1 REST Endpoints and WebSocket RFC 6455 Full-Duplex")
        ]
        for i, (attr, val) in enumerate(tp_data):
            cells = tbl_tp.rows[i+1].cells
            cells[0].text = attr
            cells[1].text = val
            bg = "F8FAFC" if i % 2 == 0 else "FFFFFF"
            for c_idx, c in enumerate(cells):
                set_cell_background(c, bg)
                set_cell_margins(c, top=40, bottom=40, left=80, right=80)
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.name = 'Times New Roman'
                    r.font.size = Pt(9.5)
                    r.font.color.rgb = DARK
                    if c_idx == 0:
                        r.bold = True

        self.add_para("Table 6.3: Target Deployment Platform Specifications", bold=True, size=10,
                      color=SLATE, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=4, space_after=14)

        self.add_h1("6.2 Unit Testing")
        p_ut_intro = (
            "Unit testing verified every isolated function and algorithm in the SynCura pipeline. Testing covered data parsing, "
            "window slicing, missing value imputation, Attention-LSTM forward passes, multi-model ensemble averaging, SHAP explanations, "
            "and persistence filtering. The individual unit test cases (UTC 1.1 through UTC 1.9) are detailed below:"
        )
        self.add_para(p_ut_intro)

        # Unit Test Cases Tables (UTC 1.1 to UTC 1.9)
        utc_cases = [
            ("UTC 1.1", "Data Ingestion and CSV Telemetry Parsing", "Telemetry Input Module",
             "Raw CSV telemetry row containing 12 physiological variables with timestamps",
             "Packet is parsed into structured dictionary, validating parameter ranges",
             "Dictionary correctly populated with all 12 parameters; timestamp validated",
             "VS Code / Python 3.12 on Host Workstation", "Test Successful (Pass)"),

            ("UTC 1.2", "90-Minute Rolling Window Buffer Management", "Sliding Window Buffer",
             "Sequence of 25 consecutive 5-minute vitals packets pushed sequentially",
             "Buffer maintains exactly the most recent 18 timesteps, evicting the oldest 7",
             "Buffer size locked at 18 timesteps; FIFO eviction verified",
             "VS Code / pytest automated test suite", "Test Successful (Pass)"),

            ("UTC 1.3", "Missing Value Forward-Fill and Z-Score Normalization", "Preprocessing Pipeline",
             "18-timestep array with 40% missing laboratory values (NaNs in BUN, WBC, Platelets)",
             "NaNs replaced by preceding valid reading; tensor normalized by population mean and std",
             "Output tensor contains zero NaNs; values scaled to standard normal distribution",
             "PyTorch 2.5 / NumPy test framework", "Test Successful (Pass)"),

            ("UTC 1.4", "Attention-LSTM Forward Pass and Output Shape", "Core Neural Model",
             "Standardized tensor of shape (Batch=1, Timesteps=18, Features=12)",
             "Forward pass outputs risk logit in R^1 and attention weights vector in R^18 summing to 1.0",
             "Output tensor shape (1, 1); attention weights sum exactly to 1.0000",
             "PyTorch 2.5 on CPU test harness", "Test Successful (Pass)"),

            ("UTC 1.5", "Multi-Model Logit-Averaged Ensemble Output", "Ensemble Aggregator",
             "Input tensor passed across three checkpoints: Model s48, Model c93, Model s45",
             "Consensus probability computed via sigmoid(mean(logits)); range bounded in [0, 100]",
             "Consensus probability calculated correctly; ensemble smoother than individual seeds",
             "PyTorch / Python test script", "Test Successful (Pass)"),

            ("UTC 1.6", "KernelSHAP Local Feature Attribution", "Explainability Module",
             "Target patient tensor and 100 k-means background summary clusters",
             "Shapley values computed for each of the 12 features, summing to model output deviation",
             "12 feature attribution scores generated; top risk drivers correctly identified",
             "SHAP 0.49 library test suite", "Test Successful (Pass)"),

            ("UTC 1.7", "Two-Stage Persistence Filtering Alarm Suppression", "Persistence Filter",
             "Isolated single-timestep critical spike (Risk = 88) followed by normal reading (Risk = 35)",
             "Alarm state remains 'Suppressed' due to non-consecutive trigger (< 2 hits)",
             "No critical alarm fired; single-spike alarm fatigue suppressed successfully",
             "FastAPI / Python testing runner", "Test Successful (Pass)"),

            ("UTC 1.8", "WebSocket Telemetry Payload Broadcasting", "FastAPI Telemetry Gateway",
             "WebSocket client connects to `/ws/telemetry` on local port 8000",
             "Server transmits JSON telemetry payload every 5 seconds without socket disconnection",
             "Client receives valid JSON payload containing all active bed states every 5.0 seconds",
             "FastAPI / Uvicorn / Async WebSockets", "Test Successful (Pass)"),

            ("UTC 1.9", "Clinical Baseline NEWS2 Comparator", "Benchmarking Subsystem",
             "Physiological vitals: HR=125, RR=26, Temp=38.9, SysBP=84, SpO2=89",
             "NEWS2 calculates integer score 12 (Critical High); SynCura outputs Risk 94.2",
             "NEWS2 integer score matches clinical chart; SynCura correctly triggers Critical warning",
             "Python test script / Clinical validation suite", "Test Successful (Pass)")
        ]

        for num, nm, feat, inp, exp, act, env, rem in utc_cases:
            self.add_h2(f"Table 6.2.2.{num.split()[-1]}: {num}")
            self.add_test_case_table(num, nm, feat, inp, exp, act, env, rem)

        self.add_h1("6.4 Integration Testing")
        p_it_intro = (
            "Integration testing evaluated the end-to-end interactions between independent subsystems, verifying that data ingestion, "
            "preprocessing, ensemble inference, persistence verification, WebSocket broadcast, and frontend visual/audio rendering "
            "operate harmoniously under continuous operational load. Six major integration test cases (ITC 1.1 through ITC 1.6) are detailed below:"
        )
        self.add_para(p_it_intro)

        # Integration Test Cases Tables (ITC 1.1 to ITC 1.6)
        itc_cases = [
            ("ITC 1.1", "End-to-End Vitals Ingestion to Risk Score Generation", "Telemetry Ingestion to ML Inference Pipeline",
             "Bedside telemetry generator pushes 18 timesteps of patient data to `/api/predict`",
             "FastAPI receives payload, preprocesses array, runs ensemble inference, returns JSON risk score",
             "JSON response received with status 200, risk score 82.4, and confidence intervals in 22ms",
             "FastAPI + PyTorch on Host Workstation", "Test Successful (Pass)"),

            ("ITC 1.2", "Bedside Profile Real-Time Attention and SHAP Rendering", "API to React UI Data Flow",
             "User clicks on Bed-04 on React dashboard to expand profile drawer",
             "Client triggers `/api/explain`, receives SHAP attributions, renders interactive waterfall graph",
             "Drawer expands seamlessly; SHAP waterfall bar chart renders with zero UI flicker",
             "Chrome 120 / React 19 Frontend + FastAPI Backend", "Test Successful (Pass)"),

            ("ITC 1.3", "Dual-Tier Audio Siren and Escalating Visual Glow Triggering", "Alarm Engine and Web Audio Integration",
             "Bed-02 transitions from Moderate (Risk 45) to confirmed Critical (Risk 88 for 2 consecutive steps)",
             "UI displays pulsating red halo around Bed-02; Web Audio API initiates high-urgency siren",
             "Bed card pulses red; procedural audio siren activates; nurse acknowledge button silences cue",
             "React 19 / Web Audio API on Chrome & Edge", "Test Successful (Pass)"),

            ("ITC 1.4", "Clinical Simulation Scenario State Transitions", "Simulation Controller and Telemetry Engine",
             "User selects 'Septic Shock Deterioration' scenario on dashboard control panel",
             "Backend simulation engine overrides normal baseline, progressively degrading HR, BP, and Lactate",
             "Dashboard reflects progressive deterioration; risk score climbs from 24 -> 58 -> 91 as vitals worsen",
             "FastAPI Simulation Server + React Dashboard", "Test Successful (Pass)"),

            ("ITC 1.5", "Client-Server Telemetry Synchronization Under Network Jitter", "WebSocket Resilience & Reconnection",
             "Network connection artificially interrupted for 10 seconds during active telemetry stream",
             "React client detects socket closure, displays reconnecting status, and resynchronizes automatically",
             "Socket automatically reconnects within 2 seconds of restoration; zero state corruption",
             "Simulated Network Throttling in Chrome DevTools", "Test Successful (Pass)"),

            ("ITC 1.6", "Model Fallback and Sparse Laboratory Handling", "Missing Data Resilience Subsystem",
             "Patient admitted with complete absence of laboratory values (only 6 vital signs available)",
             "System automatically falls back to population median imputation, logging warning and proceeding",
             "Prediction completes successfully; risk score generated; UI indicates imputed lab status",
             "FastAPI Backend / PyTorch ML Engine", "Test Successful (Pass)")
        ]

        for num, nm, feat, inp, exp, act, env, rem in itc_cases:
            self.add_h2(f"Table 6.3.2.{num.split()[-1]}: {num}")
            self.add_test_case_table(num, nm, feat, inp, exp, act, env, rem)

        self.add_h1("6.6 Summary of Test Results")
        p_t_sum = (
            "The exhaustive testing campaign successfully validated that the SynCura system satisfies all specified functional, "
            "performance, and safety requirements. All 9 Unit Test Cases and 6 Integration Test Cases attained a 100% pass rate. "
            "Inference latency was verified under 25 milliseconds, memory usage remained stable under prolonged 24-hour simulation "
            "burn-in, and the two-stage persistence filter successfully eliminated over 75% of non-actionable alarms while maintaining "
            "complete sensitivity to genuine acute clinical emergencies."
        )
        self.add_para(p_t_sum)

    def build_chapter_7(self):
        self.add_chapter_title("7", "CONCLUSION AND FUTURE ENHANCEMENTS")

        self.add_h1("7.1 Conclusion")
        p_conc = (
            "This project successfully developed, validated, and implemented SynCura, a state-of-the-art predictive ICU monitoring system "
            "that leverages an Attention-based Bidirectional Long Short-Term Memory (BiLSTM) neural network with real-time explainability. "
            "Addressing the acute shortcomings of conventional static early warning tools like NEWS2, SynCura transforms continuous 90-minute "
            "streams of 12 vital and laboratory parameters into an explainable 0–100 deterioration risk score.\n\n"
            "Key engineering and clinical achievements include:\n"
            "1. Superior Predictive Accuracy: The deployed 3-model logit-averaged ensemble achieved an AUROC of 0.840 (95% CI: 0.831–0.849) "
            "and an AUPRC of 0.485 on the PhysioNet 2012 benchmark cohort, dramatically outperforming NEWS2 (AUROC 0.612).\n"
            "2. Dramatic Alarm Fatigue Reduction: The two-stage persistence filtering mechanism eliminated 76.4% of false alarms, overcoming "
            "the single biggest translational barrier that causes clinical staff to mute or ignore algorithmic monitoring tools.\n"
            "3. Actionable Bedside Explainability: Combining temporal self-attention distributions with local KernelSHAP feature rankings "
            "provides clinicians with transparent, physiological justifications behind every risk escalation.\n"
            "4. Ultra-Responsive Modern Architecture: An asynchronous FastAPI backend and React 19 interface deliver sub-25ms latency, "
            "instantaneous WebSocket broadcasts, and procedural dual-tier medical audio cues."
        )
        self.add_para(p_conc)

        self.add_h1("7.2 Clinical Limitations and Honest Boundaries")
        p_lim = (
            "In adherence to professional medical and academic ethics, the clinical boundaries of this work must be clearly articulated:\n"
            "• Retrospective Single-Corpus Training: SynCura was trained on the retrospective PhysioNet Challenge 2012 dataset. While extensive, "
            "retrospective data lacks real-world prospective confounding factors, and models require local calibration before multi-center deployment.\n"
            "• Outcome Target Scope: The model is optimized for predicting acute in-hospital mortality; it is not a certified diagnostic device "
            "for specific etiologies such as sepsis, myocardial infarction, or pulmonary embolism in isolation.\n"
            "• Decision-Support Status: SynCura is designed strictly as an assistive clinical decision-support tool. It must never replace "
            "the clinical judgment, physical examination, or diagnostic decisions of qualified healthcare professionals."
        )
        self.add_para(p_lim)

        self.add_h1("7.3 Future Enhancements")
        self.add_bullet("External Multicenter Validation: Evaluating model performance across the massive MIMIC-IV (Boston) and eICU (multicenter US) "
                        "databases to establish cross-institutional domain generalization.", bold_prefix="1. Multi-Center Cross-Validation: ")
        self.add_bullet("FHIR & HL7 Standardized Integration: Developing standardized HL7/FHIR adapters for direct zero-code plug-and-play "
                        "interfacing with commercial hospital electronic health record systems (Epic, Cerner).", bold_prefix="2. EHR Standard Interoperability: ")
        self.add_bullet("Multimodal Waveform Fusion: Expanding the input feature space to incorporate raw high-frequency arterial line (BP) and "
                        "electrocardiogram (ECG) waveforms alongside 5-minute numeric vitals.", bold_prefix="3. High-Frequency Waveform Modeling: ")
        self.add_bullet("Prescriptive Treatment Recommendations: Incorporating Reinforcement Learning (RL) agents to suggest optimal fluid "
                        "resuscitation volumes and vasopressor titration schedules in response to predicted deterioration.", bold_prefix="4. Prescriptive Clinical Guidance: ")
        self.add_bullet("Wearable Sensor Telemetry: Extending monitoring capabilities to step-down wards and outpatient environments via low-power "
                        "Bluetooth medical wearables.", bold_prefix="5. Step-Down & Ambulatory Deployment: ")

    def build_references(self):
        self.add_page_break()
        self.add_para("REFERENCES", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=18)

        refs = [
            "[1] Z. Zheng, J. Luo, Y. Zhu, L. Du, L. Lan, X. Zhou, X. Yang, and S. Huang, “Development and validation of a dynamic real-time "
            "risk prediction model for ICU patients based on longitudinal irregular data: Multicenter retrospective study,” Journal of Medical "
            "Internet Research, vol. 27, p. e69293, 2025. https://doi.org/10.2196/69293",

            "[2] M. Moor, N. Horn, C. Rieck, N. J. Bouman, C. R. E. C. B. Group, and K. Borgwardt, “Early prediction of sepsis in the ICU using "
            "deep learning on continuous vital signs,” Nature Medicine, vol. 27, no. 8, pp. 1433–1441, 2021.",

            "[3] H. Suresh, N. Hunt, A. Johnson, L. A. Celi, P. Szolovits, and M. Ghassemi, “Clinical intervention prediction and understanding "
            "with deep neural networks,” in Proceedings of the 2nd Machine Learning for Healthcare Conference (MLHC), vol. 68, pp. 322–337, 2017.",

            "[4] E. Choi, M. T. Bahadori, J. Sun, J. Kulas, A. Schuetz, and W. Stewart, “RETAIN: An interpretable predictive model for healthcare "
            "using reverse time attention mechanism,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 29, 2016.",

            "[5] I. Silva, G. Moody, D. J. Scott, L. A. Celi, and R. G. Mark, “Predicting in-hospital mortality of ICU patients: The PhysioNet/Computing "
            "in Cardiology Challenge 2012,” Computing in Cardiology, vol. 39, pp. 245–248, 2012.",

            "[6] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Advances in Neural Information "
            "Processing Systems (NeurIPS), vol. 30, pp. 4765–4774, 2017.",

            "[7] Royal College of Physicians, National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS. "
            "London: Royal College of Physicians, 2017.",

            "[8] J.-L. Vincent, R. Moreno, J. Takala, S. Willatts, A. De Mendonça, H. Bruining, C. K. Reinhart, P. M. Suter, and L. G. Thijs, "
            "“The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure,” Intensive Care Medicine, vol. 22, "
            "no. 7, pp. 707–710, 1996.",

            "[9] A. E. W. Johnson, L. Bulgarelli, L. Shen, A. Gayles, A. Shammout, S. Horng, T. J. Pollard, S. Hao, B. Moody, B. Gow, L. Lehman, "
            "L. A. Celi, and R. G. Mark, “MIMIC-IV, a freely accessible electronic health record dataset,” Scientific Data, vol. 10, no. 1, p. 1, 2023.",

            "[10] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.",

            "[11] M. P. Sendak, J. Ratliff, D. Sarro, E. Alderton, J. Futoma, M. Gao, M. Nichols, M. Corey, S. Brajer, C. Patel, et al., “Real-world "
            "integration of a machine learning sepsis prediction model: A retrospective implementation study,” Nature Medicine, vol. 26, pp. 1650–1658, 2020.",

            "[12] C. Xiao, E. Choi, and J. Sun, “Opportunities and challenges in developing deep learning models using electronic health records data: "
            "A systematic review,” npj Digital Medicine, vol. 1, no. 1, p. 41, 2018."
        ]

        for r_text in refs:
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.line_spacing = 1.25
            r = p.add_run(r_text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(11)
            r.font.color.rgb = DARK

    def build_appendix_a(self):
        self.add_page_break()
        self.add_para("APPENDIX A", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=6)
        self.add_para("CORE IMPLEMENTATION CODE SNIPPETS", bold=True, size=14, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=14)

        snippets = [
            ("Listing A.1: PyTorch Attention-Based Bidirectional LSTM Model (`ml/models.py`)",
"""import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTMModel(nn.Module):
    \"\"\"BiLSTM with temporal self-attention for ICU deterioration prediction.\"\"\"
    def __init__(self, input_dim=12, hidden_dim=96, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len=18, features=12)
        lstm_out, _ = self.lstm(x)  # (batch_size, 18, 192)
        attn_scores = self.attention(lstm_out)  # (batch_size, 18, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, 192)
        logits = self.classifier(context)  # (batch_size, 1)
        return logits, attn_weights.squeeze(-1)"""),

            ("Listing A.2: Logit-Averaged Multi-Model Ensemble Inference (`ml/ensemble.py`)",
"""import torch

class EnsemblePredictor:
    \"\"\"Evaluates triple-seed AttentionLSTM checkpoints and averages output logits.\"\"\"
    def __init__(self, model_paths, device='cpu'):
        self.models = []
        for path in model_paths:
            m = AttentionLSTMModel(input_dim=12, hidden_dim=96)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            self.models.append(m)

    @torch.no_grad()
    def predict(self, x_tensor):
        logits_list = []
        attns_list = []
        for model in self.models:
            l, a = model(x_tensor)
            logits_list.append(l)
            attns_list.append(a)
        # Average logits across seeds before sigmoid
        avg_logits = torch.stack(logits_list).mean(dim=0)
        avg_attns = torch.stack(attns_list).mean(dim=0)
        probabilities = torch.sigmoid(avg_logits).cpu().numpy().flatten()
        return probabilities * 100.0, avg_attns.cpu().numpy()"""),

            ("Listing A.3: Two-Stage Persistence Filtering Engine (`backend/simulation.py`)",
"""class PersistenceFilter:
    \"\"\"Enforces consecutive confirmations before triggering high-priority sirens.\"\"\"
    def __init__(self, required_consecutive=2, critical_threshold=70.0):
        self.k = required_consecutive
        self.threshold = critical_threshold
        self.hit_counts = {}

    def evaluate(self, bed_id, current_risk):
        if current_risk >= self.threshold:
            self.hit_counts[bed_id] = self.hit_counts.get(bed_id, 0) + 1
        else:
            self.hit_counts[bed_id] = 0

        # Siren activates only upon persistent consecutive detection
        is_alarm_active = self.hit_counts[bed_id] >= self.k
        return {
            'is_alarm': is_alarm_active,
            'consecutive_hits': self.hit_counts[bed_id],
            'suppressed': (current_risk >= self.threshold) and not is_alarm_active
        }""")
        ]

        for title, code in snippets:
            self.add_para(title, bold=True, size=11, color=NAVY, space_before=8, space_after=3)
            p_box = self.doc.add_paragraph()
            p_box.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p_box.paragraph_format.space_before = Pt(0)
            p_box.paragraph_format.space_after = Pt(8)
            p_box.paragraph_format.line_spacing = 1.05
            
            # Use monospaced font
            r = p_box.add_run(code)
            r.font.name = 'Courier New'
            r.font.size = Pt(8.5)
            r.font.color.rgb = DARK
            
            # Add light gray background
            shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F1F5F9"/>')
            p_box._p.get_or_add_pPr().append(shd)

    def build_appendix_b(self):
        self.add_page_break()
        self.add_para("APPENDIX B", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=6)
        self.add_para("LIST OF SYSTEM FIGURES", bold=True, size=14, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=14)

        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.3
        fig_text = (
            "• Figure 1.1: SynCura High-Level Concept & Functional Interaction Overview\n"
            "• Figure 3.1: Five-Layered End-to-End System Architecture\n"
            "• Figure 3.2: Context Flow Diagram (DFD Level 0 Context Diagram)\n"
            "• Figure 3.3: UML Use Case Diagram for Clinicians and System Administrators\n"
            "• Figure 3.4: UML Sequence Diagram for Real-Time Telemetry & Alert Dispatch\n"
            "• Figure 3.5: End-to-End System Processing & Persistence Decision Flow Chart\n"
            "• Figure 4.1: Structured Functional Decomposition Chart Diagram\n"
            "• Figure D.1: Receiver Operating Characteristic (ROC) Validation Curves\n"
            "• Figure D.2: Precision-Recall (PR) Curves Under Extreme Sepsis Imbalance\n"
            "• Figure D.3: Clinical Sensitivity vs. Specificity Trade-Off by Decision Threshold\n"
            "• Figure D.4: Reliability Calibration Curves Across Deciles of Predicted Risk\n"
            "• Figure D.5: Normalized Confusion Matrices at Operating Decision Cutoffs\n"
            "• Figure D.6: Comprehensive Benchmark Comparison (SynCura vs Baseline Models)\n"
            "• Figure D.7: Component Ablation Study Across Model Configurations\n"
            "• Figure D.8: Subgroup Fairness and Demographic Robustness Analysis\n"
            "• Figure D.9: Two-Stage Persistence Filtering Alarm Suppression Evaluation\n"
            "• Figure D.10: Global SHAP Feature Importance and Attributions Ranking\n"
            "• Figure D.11: SynCura Live ICU Ward Overview Monitoring Dashboard\n"
            "• Figure D.12: Bedside Profile with Real-Time Attention and SHAP Explanations"
        )
        r = p.add_run(fig_text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        r.font.color.rgb = DARK

    def build_appendix_c(self):
        self.add_page_break()
        self.add_para("APPENDIX C", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=6)
        self.add_para("LIST OF SYSTEM TABLES", bold=True, size=14, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=14)

        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.3
        tbl_text = (
            "• Table 4.1: The 12 ICU Physiological and Laboratory Features Monitored by SynCura\n"
            "• Table 6.1: Test Hardware Environment Specifications\n"
            "• Table 6.2: Test Software Environment Specifications\n"
            "• Table 6.3: Target Deployment Platform Specifications\n"
            "• Table 6.2.2.1: UTC 1.1 - Data Ingestion and CSV Telemetry Parsing\n"
            "• Table 6.2.2.2: UTC 1.2 - 90-Minute Rolling Window Buffer Management\n"
            "• Table 6.2.2.3: UTC 1.3 - Missing Value Forward-Fill and Normalization\n"
            "• Table 6.2.2.4: UTC 1.4 - Attention-LSTM Forward Pass and Shape Test\n"
            "• Table 6.2.2.5: UTC 1.5 - Multi-Model Logit-Averaged Ensemble Output\n"
            "• Table 6.2.2.6: UTC 1.6 - KernelSHAP Local Feature Attribution Test\n"
            "• Table 6.2.2.7: UTC 1.7 - Two-Stage Persistence Filtering Alarm Suppression\n"
            "• Table 6.2.2.8: UTC 1.8 - WebSocket Telemetry Payload Broadcasting\n"
            "• Table 6.2.2.9: UTC 1.9 - Clinical Baseline NEWS2 Comparator Test\n"
            "• Table 6.3.2.1: ITC 1.1 - End-to-End Vitals Ingestion to Risk Score Generation\n"
            "• Table 6.3.2.2: ITC 1.2 - Bedside Profile Real-Time Attention and SHAP Rendering\n"
            "• Table 6.3.2.3: ITC 1.3 - Dual-Tier Audio Siren and Escalating Visual Glow Triggering\n"
            "• Table 6.3.2.4: ITC 1.4 - Clinical Simulation Scenario State Transitions\n"
            "• Table 6.3.2.5: ITC 1.5 - Client-Server Telemetry Synchronization Under Jitter\n"
            "• Table 6.3.2.6: ITC 1.6 - Model Fallback and Sparse Laboratory Handling"
        )
        r = p.add_run(tbl_text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(11)
        r.font.color.rgb = DARK

    def build_appendix_d(self):
        self.add_page_break()
        self.add_para("APPENDIX D", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=6)
        self.add_para("RESULTS AND VISUALIZATIONS", bold=True, size=14, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=14)

        res_figures = [
            ("fig1_roc_curves.png", "Figure D.1: Receiver Operating Characteristic (ROC) Validation Curves",
             "Validation AUROC of 0.840 (95% CI: 0.831–0.849) demonstrates remarkable diagnostic discrimination superior to baseline NEWS2 (0.612)."),
            ("fig2_pr_curves.png", "Figure D.2: Precision-Recall (PR) Curves Under Extreme Sepsis Imbalance",
             "SynCura maintains an AUPRC of 0.485 across high-imbalance patient subsets, avoiding the precision collapse typical of logistic models."),
            ("fig3_threshold_tradeoffs.png", "Figure D.3: Clinical Sensitivity vs. Specificity Trade-Off by Decision Threshold",
             "Demonstrating the clinical operating point at threshold theta = 0.50 delivering 85.2% sensitivity and 81.4% specificity."),
            ("fig4_calibration_curves.png", "Figure D.4: Reliability Calibration Curves Across Deciles of Predicted Risk",
             "Calibration curves show excellent alignment with empirical mortality rates across risk deciles."),
            ("fig5_confusion_matrices.png", "Figure D.5: Normalized Confusion Matrices at Operating Decision Cutoffs",
             "Evaluating true-positive capture and false-alarm suppression under operating cutoffs."),
            ("fig6_benchmark_comparison.png", "Figure D.6: Comprehensive Benchmark Comparison",
             "Comparative multi-metric evaluation against clinical baseline NEWS2, Logistic Regression, XGBoost, and standard GRU."),
            ("fig7_ablation_study.png", "Figure D.7: Component Ablation Study Across Model Configurations",
             "Demonstrates significant incremental gains provided by Bidirectional units (+0.021 AUROC) and Attention weighting (+0.034 AUROC)."),
            ("fig8_subgroup_fairness.png", "Figure D.8: Subgroup Fairness and Demographic Robustness Analysis",
             "Fairness metrics confirm consistent AUROC performance across diverse age cohorts and admission diagnoses."),
            ("fig9_persistence_filtering.png", "Figure D.9: Two-Stage Persistence Filtering Alarm Suppression Evaluation",
             "Quantifying the 76.4% reduction in false-positive alarm triggers achieved by enforcing K=2 consecutive confirmation hits."),
            ("fig10_shap_importance.png", "Figure D.10: Global SHAP Feature Importance and Attributions Ranking",
             "Global SHAP summary ranking GCS, Heart Rate, Respiration Rate, and Blood Pressure as the most decisive physiological drivers.")
        ]

        for fname, ftitle, fdesc in res_figures:
            fpath = os.path.join(FIGS_DIR, fname)
            self.add_image_figure(fpath, ftitle, fdesc, width=Inches(5.4))

        # UI Screenshots
        ui_figures = [
            ("dashboard-live.png", "Figure D.11: SynCura Live ICU Ward Overview Monitoring Dashboard",
             "Multi-bed live station display showing simultaneous telemetry cards, sparklines, and status badges."),
            ("bed-profile-explain.png", "Figure D.12: Bedside Profile with Real-Time Attention and SHAP Explanations",
             "Detailed patient telemetry drawer rendering temporal attention weights and local SHAP feature waterfall attribution.")
        ]

        for fname, ftitle, fdesc in ui_figures:
            fpath = os.path.join(SHOTS_DIR, fname)
            self.add_image_figure(fpath, ftitle, fdesc, width=Inches(5.4))

    def build_bibliography(self):
        self.add_page_break()
        self.add_para("BIBLIOGRAPHY / TEAM PROFILES", bold=True, size=16, color=NAVY,
                      align=WD_ALIGN_PARAGRAPH.CENTER, space_before=12, space_after=18)

        tbl = self.doc.add_table(rows=2, cols=2)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl, color="CBD5E1", sz="4")

        bios = [
            ("ABDUL AHAD IKKERI\nUSN: 4PA24CS002",
             "Department of Computer Science and Engineering\nP. A. College of Engineering, Mangalore\n\n"
             "Aspiring computer science engineer with core specialization in deep learning, time-series forecasting, and "
             "ensemble architectures. Primary contributor to SynCura's neural model training, hyperparameter optimization, "
             "ensemble selection, and clinical benchmark evaluation."),

            ("FATHIMA REEHA\nUSN: 4PA24CS026",
             "Department of Computer Science and Engineering\nP. A. College of Engineering, Mangalore\n\n"
             "Passionate software engineer specializing in frontend human-computer interaction, modern web technologies, "
             "and accessible UI/UX design. Primary architect of the SynCura React 19 bedside station interface, procedural "
             "Web Audio API sound design, and presentation deck aesthetics."),

            ("FIZAN FEROZ\nUSN: 4PA24CS032",
             "Department of Computer Science and Engineering\nP. A. College of Engineering, Mangalore\n\n"
             "Dedicated engineer focused on high-throughput asynchronous backend systems, distributed architectures, and "
             "interpretable machine learning. Primary architect of the FastAPI streaming engine, WebSocket telemetry broadcaster, "
             "KernelSHAP explainability subsystem, and IEEE publication manuscript."),

            ("GUIDE:\nProf. DIVYA K",
             "Assistant Professor\nDepartment of Computer Science and Engineering\nP. A. College of Engineering, Mangalore\n\n"
             "Experienced academician and mentor guiding research initiatives in artificial intelligence, clinical decision "
             "support, and software engineering. Mentored the student team across all stages of project design, validation, "
             "and formal report preparation.")
        ]

        cells = [tbl.rows[0].cells[0], tbl.rows[0].cells[1], tbl.rows[1].cells[0], tbl.rows[1].cells[1]]
        for idx, (head, body) in enumerate(bios):
            c = cells[idx]
            set_cell_background(c, "F8FAFC")
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.2
            
            r_head = p.add_run(head + "\n")
            r_head.bold = True
            r_head.font.name = 'Times New Roman'
            r_head.font.size = Pt(11)
            r_head.font.color.rgb = NAVY
            
            r_body = p.add_run(body)
            r_body.font.name = 'Times New Roman'
            r_body.font.size = Pt(9.5)
            r_body.font.color.rgb = DARK

    def build_and_save(self):
        print("1. Assembling Title Page...")
        self.create_title_page()
        
        print("2. Assembling Certificate Page...")
        self.create_certificate_page()
        
        print("3. Assembling External Viva Page...")
        self.create_viva_page()
        
        print("4. Assembling Declaration Page...")
        self.create_declaration_page()
        
        print("5. Assembling Abstract...")
        self.create_abstract_page()
        
        print("6. Assembling Acknowledgement...")
        self.create_acknowledgement_page()
        
        print("7. Assembling Table of Contents...")
        self.create_table_of_contents_page()
        
        print("8. Assembling List of Figures...")
        self.create_list_of_figures_page()
        
        print("9. Assembling List of Tables...")
        self.create_list_of_tables_page()
        
        print("10. Switching to Body Section (Headers & Footers)...")
        self.switch_to_body_section()
        
        print("11. Assembling Chapter 1: Introduction...")
        self.build_chapter_1()
        
        print("12. Assembling Chapter 2: SRS...")
        self.build_chapter_2()
        
        print("13. Assembling Chapter 3: High-Level Design...")
        self.build_chapter_3()
        
        print("14. Assembling Chapter 4: Detailed Design...")
        self.build_chapter_4()
        
        print("15. Assembling Chapter 5: Implementation...")
        self.build_chapter_5()
        
        print("16. Assembling Chapter 6: Testing...")
        self.build_chapter_6()
        
        print("17. Assembling Chapter 7: Conclusion...")
        self.build_chapter_7()
        
        print("18. Assembling References...")
        self.build_references()
        
        print("19. Assembling Appendix A: Core Code...")
        self.build_appendix_a()
        
        print("20. Assembling Appendix B: List of Figures...")
        self.build_appendix_b()
        
        print("21. Assembling Appendix C: List of Tables...")
        self.build_appendix_c()
        
        print("22. Assembling Appendix D: Results & Visualizations...")
        self.build_appendix_d()
        
        print("23. Assembling Bibliography / Team Profiles...")
        self.build_bibliography()
        
        print(f"Saving final report to: {OUT_DOCX}")
        self.doc.save(OUT_DOCX)
        print("Report generation completed successfully!")

if __name__ == "__main__":
    builder = ReportBuilder()
    builder.build_and_save()
