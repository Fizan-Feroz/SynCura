# -*- coding: utf-8 -*-
"""Approximate raster preview of the deck (no PowerPoint / LibreOffice needed).

This is a QA aid, not a renderer: it draws autoshape fills, text with real font
metrics, tables and line charts so layout collisions and overflow are visible.
"""
import io
import os
import sys

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "SynCura_Synopsis_V4.pptx")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "preview")
DPI = 120
E = 914400

FONTS = {
    ("Calibri", False): r"C:\Windows\Fonts\calibri.ttf",
    ("Calibri", True): r"C:\Windows\Fonts\calibrib.ttf",
    ("Cambria", False): r"C:\Windows\Fonts\cambria.ttc",
    ("Cambria", True): r"C:\Windows\Fonts\cambriab.ttf",
    ("Times New Roman", False): r"C:\Windows\Fonts\times.ttf",
    ("Times New Roman", True): r"C:\Windows\Fonts\timesbd.ttf",
}
_fc = {}


def font(name, bold, size_pt):
    path = FONTS.get((name, bool(bold))) or FONTS[("Calibri", bool(bold))]
    if not os.path.exists(path):
        path = FONTS[("Calibri", bool(bold))]
    key = (path, round(size_pt, 1))
    if key not in _fc:
        _fc[key] = ImageFont.truetype(path, max(7, int(round(size_pt * DPI / 72))))
    return _fc[key]


def px(emu):
    return int(round(emu / E * DPI))


def color(fill):
    try:
        if fill.type is None:
            return None
        rgb = fill.fore_color.rgb
        return (rgb[0], rgb[1], rgb[2])
    except Exception:
        return None


def wrap(draw, text, fnt, max_w):
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if draw.textlength(cand, font=fnt) <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_text_frame(draw, shape):
    tf = shape.text_frame
    x0 = px(shape.left + (tf.margin_left or 0))
    y0 = px(shape.top + (tf.margin_top or 0))
    w = px(shape.width - (tf.margin_left or 0) - (tf.margin_right or 0))
    h = px(shape.height - (tf.margin_top or 0) - (tf.margin_bottom or 0))

    # measure block for vertical centering
    blocks = []
    total_h = 0
    for p in tf.paragraphs:
        txt = "".join(r.text for r in p.runs)
        size = 12
        bold = False
        name = "Calibri"
        col = (0, 0, 0)
        for r in p.runs:
            if r.font.size:
                size = r.font.size.pt
            bold = bool(r.font.bold)
            if r.font.name:
                name = r.font.name
            try:
                c = r.font.color.rgb
                col = (c[0], c[1], c[2])
            except Exception:
                pass
        fnt = font(name, bold, size)
        lines = wrap(draw, txt, fnt, w)
        lh = int((fnt.getmetrics()[0] + fnt.getmetrics()[1]) * (p.line_spacing or 1.0))
        sa = int((p.space_after.pt if p.space_after else 0) * DPI / 72)
        blocks.append((lines, fnt, lh, sa, p.alignment, col))
        total_h += lh * len(lines) + sa

    y = y0
    if tf.vertical_anchor == MSO_ANCHOR.MIDDLE:
        y = y0 + max(0, (h - total_h) // 2)
    elif tf.vertical_anchor == MSO_ANCHOR.BOTTOM:
        y = y0 + max(0, h - total_h)

    for lines, fnt, lh, sa, align, col in blocks:
        for ln in lines:
            tw = draw.textlength(ln, font=fnt)
            tx = x0
            if align == PP_ALIGN.CENTER:
                tx = x0 + (w - tw) / 2
            elif align == PP_ALIGN.RIGHT:
                tx = x0 + w - tw
            draw.text((tx, y), ln, font=fnt, fill=col)
            y += lh
        y += sa


def draw_table(draw, shape, scale):
    tbl = shape.table
    x = px(shape.left)
    y = px(shape.top)
    widths = [px(c.width) for c in tbl.columns]
    for ri, row in enumerate(tbl.rows):
        rh = px(row.height)
        cx = x
        for ci, cell in enumerate(row.cells):
            cw = widths[ci]
            fill = color(cell.fill)
            draw.rectangle([cx, y, cx + cw, y + rh], fill=fill, outline=(216, 230, 224))
            ml = px(cell.margin_left or 0)
            ty = y + 4
            for p in cell.text_frame.paragraphs:
                txt = "".join(r.text for r in p.runs)
                if not txt:
                    continue
                size, bold, name = 9.5, False, "Calibri"
                col = (30, 30, 30)
                for r in p.runs:
                    if r.font.size:
                        size = r.font.size.pt
                    bold = bool(r.font.bold)
                    if r.font.name:
                        name = r.font.name
                    try:
                        c = r.font.color.rgb
                        col = (c[0], c[1], c[2])
                    except Exception:
                        pass
                fnt = font(name, bold, size)
                for ln in wrap(draw, txt, fnt, cw - 2 * ml):
                    draw.text((cx + ml, ty), ln, font=fnt, fill=col)
                    ty += int((fnt.getmetrics()[0] + fnt.getmetrics()[1]) * 1.05)
            cx += cw
        y += rh


def draw_chart(draw, shape, prs_slide, scale):
    """Approximate the results line chart from its known series."""
    x0, y0 = px(shape.left), px(shape.top)
    w, h = px(shape.width), px(shape.height)
    plot_l, plot_t = x0 + 55, y0 + 34
    plot_r, plot_b = x0 + w - 20, y0 + h - 46
    cats = ["0.40", "0.50", "0.60", "0.65", "0.70", "0.75", "0.80"]
    series = [
        ("Accuracy", (0.7116, 0.7658, 0.8007, 0.8229, 0.8433, 0.8579, 0.8665), (28, 124, 106)),
        ("Sensitivity", (0.8550, 0.7825, 0.6725, 0.6154, 0.5392, 0.4312, 0.3025), (196, 34, 63)),
        ("Specificity", (0.6879, 0.7630, 0.8219, 0.8571, 0.8935, 0.9284, 0.9596), (95, 122, 115)),
    ]
    fnt = font("Calibri", False, 9)

    # frame + grid
    draw.rectangle([plot_l, plot_t, plot_r, plot_b], outline=(216, 230, 224))
    for i in range(5):
        val = 0.2 + i * 0.2
        yy = plot_b - (val - 0.2) / 0.8 * (plot_b - plot_t)
        draw.line([plot_l, yy, plot_r, yy], fill=(232, 240, 236))
        draw.text((plot_l - 34, yy - 6), f"{val:.1f}", font=fnt, fill=(95, 122, 115))

    step = (plot_r - plot_l) / (len(cats) - 1) if len(cats) > 1 else 0
    for i, c in enumerate(cats):
        xx = plot_l + i * step
        draw.text((xx - 8, plot_b + 6), c, font=fnt, fill=(95, 122, 115))

    for name, vals, col in series:
        pts = []
        for i, v in enumerate(vals):
            xx = plot_l + i * step
            yy = plot_b - (v - 0.2) / 0.8 * (plot_b - plot_t)
            pts.append((xx, yy))
        draw.line(pts, fill=col, width=3)
        for (xx, yy) in pts:
            draw.ellipse([xx - 3, yy - 3, xx + 3, yy + 3], fill=col)

    lx = plot_l
    for name, _, col in series:
        draw.rectangle([lx, plot_b + 26, lx + 16, plot_b + 30], fill=col)
        draw.text((lx + 22, plot_b + 18), name, font=fnt, fill=(44, 68, 64))
        lx += 110


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    prs = Presentation(DECK)
    W, H = px(prs.slide_width), px(prs.slide_height)
    for i, slide in enumerate(prs.slides, start=1):
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        bg = color(slide.background.fill)
        if bg:
            draw.rectangle([0, 0, W, H], fill=bg)
        for sh in slide.shapes:
            if sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
                fill = color(sh.fill)
                line = None
                try:
                    if sh.line.color and sh.line.color.type is not None:
                        c = sh.line.color.rgb
                        line = (c[0], c[1], c[2])
                except Exception:
                    line = None
                box = [px(sh.left), px(sh.top), px(sh.left + sh.width), px(sh.top + sh.height)]
                name = str(sh.auto_shape_type)
                if "ROUNDED" in name:
                    draw.rounded_rectangle(box, radius=int(0.06 * DPI), fill=fill, outline=line)
                else:
                    draw.rectangle(box, fill=fill, outline=line)
                if sh.has_text_frame and sh.text_frame.text.strip():
                    draw_text_frame(draw, sh)
            elif sh.has_table:
                draw_table(draw, sh, 1)
            elif sh.has_chart:
                draw_chart(draw, sh, slide, 1)
            elif sh.shape_type == MSO_SHAPE_TYPE.LINE:
                try:
                    x1, y1 = px(sh.begin_x), px(sh.begin_y)
                    x2, y2 = px(sh.end_x), px(sh.end_y)
                    draw.line([x1, y1, x2, y2], fill=(31, 56, 100), width=2)
                    draw.polygon([(x2, y2), (x2 - 7, y2 - 4), (x2 - 7, y2 + 4)], fill=(31, 56, 100))
                except Exception:
                    pass
            elif sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    im = Image.open(io.BytesIO(sh.image.blob)).convert("RGBA")
                    box = (px(sh.left), px(sh.top), px(sh.left + sh.width), px(sh.top + sh.height))
                    im = im.resize((max(1, box[2] - box[0]), max(1, box[3] - box[1])))
                    base = img.convert("RGBA")
                    base.alpha_composite(im, (max(0, box[0]), max(0, box[1])))
                    img.paste(base.convert("RGB"))
                except Exception as exc:
                    print("picture skip", exc)
            elif sh.has_text_frame and sh.text_frame.text.strip():
                draw_text_frame(draw, sh)
        img.save(os.path.join(OUTDIR, f"slide-{i:02d}.png"))
    print("wrote", len(prs.slides._sldIdLst), "images to", OUTDIR)


if __name__ == "__main__":
    main()
