# -*- coding: utf-8 -*-
"""Programmatic QA for the generated deck: bounds + text-fit estimates."""
import os
import sys

from PIL import ImageFont
from pptx import Presentation
from pptx.util import Emu

DECK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "SynCura_Synopsis_V4.pptx")
EMU_IN = 914400
DPI = 96

FONTS = {
    ("Calibri", False): r"C:\Windows\Fonts\calibri.ttf",
    ("Calibri", True): r"C:\Windows\Fonts\calibrib.ttf",
    ("Cambria", False): r"C:\Windows\Fonts\cambria.ttc",
    ("Cambria", True): r"C:\Windows\Fonts\cambriab.ttf",
}
_cache = {}


def font_for(name, bold, size_pt):
    path = FONTS.get((name, bool(bold)))
    if not path or not os.path.exists(path):
        path = FONTS[("Calibri", bool(bold))]
    key = (path, round(size_pt, 2))
    if key not in _cache:
        _cache[key] = ImageFont.truetype(path, max(6, int(round(size_pt * DPI / 72.0))))
    return _cache[key]


def wrap_lines(text, font, max_px):
    words = text.split()
    if not words:
        return 1
    lines, cur = 1, ""
    for w in words:
        cand = (cur + " " + w).strip()
        if font.getlength(cand) <= max_px:
            cur = cand
        else:
            if cur:
                lines += 1
            cur = w
            while font.getlength(cur) > max_px and len(cur) > 1:
                # hard break very long token
                cut = len(cur)
                while cut > 1 and font.getlength(cur[:cut]) > max_px:
                    cut -= 1
                cur = cur[cut:]
                lines += 1
    return lines


def shape_text_height(shape):
    """Return (required_px, box_px, worst_paragraph_text)."""
    tf = shape.text_frame
    box_w_px = (shape.width / EMU_IN) * DPI
    left = tf.margin_left or 0
    right = tf.margin_right or 0
    top = tf.margin_top or 0
    bottom = tf.margin_bottom or 0
    avail_w = box_w_px - (left + right) / EMU_IN * DPI
    required = (top + bottom) / EMU_IN * DPI
    worst = ""
    worst_over = 0
    for p in tf.paragraphs:
        text = "".join(r.text for r in p.runs)
        if not text.strip():
            size = 12
            required += size * DPI / 72 * 1.4
            continue
        # size/bold from the first run that has one
        size = None
        bold = False
        name = "Calibri"
        for r in p.runs:
            if r.font.size:
                size = r.font.size.pt
            bold = bool(r.font.bold)
            if r.font.name:
                name = r.font.name
            if size:
                break
        size = size or 12
        f = font_for(name, bold, size)
        lines = wrap_lines(text, f, avail_w)
        asc, desc = f.getmetrics()
        line_px = (asc + desc) * (p.line_spacing or 1.0)
        sa = (p.space_after.pt if p.space_after else 0) * DPI / 72
        required += lines * line_px + sa
        over = required - (shape.height / EMU_IN * DPI)
        if over > worst_over:
            worst_over = over
            worst = text[:60]
    return required, (shape.height / EMU_IN) * DPI, worst


def main():
    prs = Presentation(DECK)
    sw = prs.slide_width / EMU_IN
    sh = prs.slide_height / EMU_IN
    issues = 0
    for idx, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if shape.width is None or shape.height is None:
                continue
            l = shape.left / EMU_IN
            t = shape.top / EMU_IN
            r = l + shape.width / EMU_IN
            b = t + shape.height / EMU_IN
            if l < -0.01 or t < -0.01 or r > sw + 0.01 or b > sh + 0.01:
                print(f"[BOUNDS] slide {idx}: {shape.shape_type} at ({l:.2f},{t:.2f})-({r:.2f},{b:.2f})")
                issues += 1
            if shape.has_text_frame and shape.text_frame.text.strip():
                req, box, worst = shape_text_height(shape)
                if req > box * 1.04:
                    print(f"[OVERFLOW?] slide {idx}: need {req/DPI:.2f}in, box {box/DPI:.2f}in :: {worst}")
                    issues += 1
    print("slides:", len(prs.slides._sldIdLst), "issues:", issues)


if __name__ == "__main__":
    main()
