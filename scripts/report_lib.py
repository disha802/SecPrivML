"""
Small python-docx helper layer used by 05_build_reports.py.

Gives every report the same look: A4, Calibri body, numbered headings,
auto-numbered table/figure captions, shaded header rows, code blocks,
a Word TOC field (refreshed by Word on open / by 06_export_pdf.py) and
page-numbered footers.

Inline markup in ``para``/``bullets``: **bold**, *italic*, `code`.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ACCENT = RGBColor(0x1C, 0x5C, 0xAB)
INK2 = RGBColor(0x52, 0x51, 0x4E)
HEADER_FILL = "DCE8F7"
CODE_FILL = "F4F4F2"


def _shade(cell_or_par, fill: str):
    el = cell_or_par._tc if hasattr(cell_or_par, "_tc") else cell_or_par._p
    pr = el.get_or_add_tcPr() if hasattr(cell_or_par, "_tc") else el.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pr.append(shd)


def _field(paragraph, instr: str, placeholder: str = ""):
    run = paragraph.add_run()
    for tag, text in (("begin", None), ("instr", instr), ("separate", None),
                      ("text", placeholder), ("end", None)):
        if tag == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        elif tag == "text":
            el = OxmlElement("w:t")
            el.text = text
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        run._r.append(el)
    return run


_INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)")


def add_inline(paragraph, text: str, size: float | None = None, color=None):
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = paragraph.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = paragraph.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = paragraph.add_run(part[1:-1])
            r.italic = True
        else:
            r = paragraph.add_run(part)
        if size and not part.startswith("`"):
            r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = color
    return paragraph


class Report:
    def __init__(self, header_text: str):
        self.doc = Document()
        self.n_table = 0
        self.n_fig = 0
        sec = self.doc.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        sec.orientation = WD_ORIENT.PORTRAIT
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2.3))
        sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.2)
        self.text_width = Cm(21.0 - 4.6)

        st = self.doc.styles
        st["Normal"].font.name = "Calibri"
        st["Normal"].font.size = Pt(11)
        st["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        pf = st["Normal"].paragraph_format
        pf.space_after = Pt(6)
        pf.line_spacing = 1.12
        for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
            h = st[f"Heading {lvl}"]
            h.font.name = "Calibri"
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = ACCENT
            h._element.rPr.rFonts.set(qn("w:asciiTheme"), "")
            h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
            h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.keep_with_next = True
        cap = st["Caption"]
        cap.font.size = Pt(9.5)
        cap.font.italic = False
        cap.font.bold = False
        cap.font.color.rgb = INK2

        # header + page-number footer
        hp = sec.header.paragraphs[0]
        hp.text = header_text
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for r in hp.runs:
            r.font.size = Pt(8.5)
            r.font.color.rgb = INK2
        fp = sec.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = fp.add_run("Page ")
        r.font.size = Pt(9)
        _field(fp, "PAGE", "1").font.size = Pt(9)
        r = fp.add_run(" of ")
        r.font.size = Pt(9)
        _field(fp, "NUMPAGES", "1").font.size = Pt(9)
        sec.different_first_page_header_footer = True

    # ------------------------------------------------------------- structure
    def title_page(self, course: str, kind: str, title: str, subtitle: str, fields: list[tuple]):
        d = self.doc
        for _ in range(3):
            d.add_paragraph()
        for text, size, bold, color in ((course, 14, True, INK2), (kind, 12, False, INK2)):
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(text)
            r.font.size, r.bold, r.font.color.rgb = Pt(size), bold, color
        d.add_paragraph()
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title)
        r.font.size, r.bold, r.font.color.rgb = Pt(22), True, ACCENT
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(subtitle)
        r.font.size, r.italic = Pt(12.5), True
        for _ in range(3):
            d.add_paragraph()
        t = d.add_table(rows=0, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for k, v in fields:
            row = t.add_row().cells
            row[0].width, row[1].width = Cm(5), Cm(8.5)
            add_inline(row[0].paragraphs[0], f"**{k}**")
            add_inline(row[1].paragraphs[0], v)
        self.page_break()

    def toc(self):
        p = self.doc.add_paragraph()
        r = p.add_run("Contents")
        r.font.size, r.bold, r.font.color.rgb = Pt(16), True, ACCENT
        p = self.doc.add_paragraph()
        _field(p, 'TOC \\o "1-2" \\h \\z \\u',
               "Right-click and choose 'Update Field' to build the table of contents.")
        self.page_break()

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def h1(self, text):
        return self.doc.add_heading(text, level=1)

    def h2(self, text):
        return self.doc.add_heading(text, level=2)

    def h3(self, text):
        return self.doc.add_heading(text, level=3)

    # --------------------------------------------------------------- content
    def para(self, text: str, align=None, size=None, italic=False):
        p = self.doc.add_paragraph()
        add_inline(p, text, size)
        if italic:
            for r in p.runs:
                r.italic = True
        if align == "center":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align is None:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return p

    def bullets(self, items, numbered: bool = False):
        style = "List Number" if numbered else "List Bullet"
        for it in items:
            p = self.doc.add_paragraph(style=style)
            add_inline(p, it)
            p.paragraph_format.space_after = Pt(3)

    def equation(self, text: str):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.font.name = "Cambria Math"
        r.font.size = Pt(11)
        r.italic = True

    def code(self, text: str, size: float = 8.5):
        for line in text.strip("\n").split("\n"):
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_after = Pt(0)
            pf.space_before = Pt(0)
            pf.line_spacing = 1.0
            pf.left_indent = Cm(0.3)
            r = p.add_run(line if line else " ")
            r.font.name = "Consolas"
            r.font.size = Pt(size)
            _shade(p, CODE_FILL)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def caption(self, kind: str, text: str):
        if kind == "Table":
            self.n_table += 1
            n = self.n_table
        else:
            self.n_fig += 1
            n = self.n_fig
        p = self.doc.add_paragraph(style="Caption")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if kind == "Figure" else WD_ALIGN_PARAGRAPH.LEFT
        add_inline(p, f"**{kind} {n}.** {text}")
        if kind == "Table":
            p.paragraph_format.keep_with_next = True
        return n

    def table(self, header: list[str], rows: list[list], caption: str | None = None,
              widths: list[float] | None = None, size: float = 9, bold_first_col=False):
        if caption:
            self.caption("Table", caption)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        total = self.text_width.cm
        widths = widths or [total / len(header)] * len(header)
        scale = total / sum(widths)
        widths = [w * scale for w in widths]
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            c.paragraphs[0].text = ""
            add_inline(c.paragraphs[0], f"**{h}**", size)
            _shade(c, HEADER_FILL)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                txt = str(v)
                if bold_first_col and i == 0 and not txt.startswith("**"):
                    txt = f"**{txt}**"
                add_inline(cells[i].paragraphs[0], txt, size)
        for row in t.rows:
            for i, c in enumerate(row.cells):
                c.width = Cm(widths[i])
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if len(t.rows) <= 16:                     # keep short tables on one page
            for row in t.rows[:-1]:
                for c in row.cells:
                    for p in c.paragraphs:
                        p.paragraph_format.keep_with_next = True
        # repeat header row across pages
        trPr = t.rows[0]._tr.get_or_add_trPr()
        el = OxmlElement("w:tblHeader")
        el.set(qn("w:val"), "true")
        trPr.append(el)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def figure(self, path: Path, caption: str, width_cm: float = 16.0):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(min(width_cm, self.text_width.cm)))
        self.caption("Figure", caption)

    def callout(self, text: str, fill: str = "EAF2FC"):
        p = self.doc.add_paragraph()
        add_inline(p, text, 10)
        p.paragraph_format.left_indent = Cm(0.3)
        p.paragraph_format.right_indent = Cm(0.3)
        _shade(p, fill)
        return p

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        print("wrote", path)
