"""
Step 6 (optional, Windows + Microsoft Word only) — refresh the table of
contents / page fields in each report and export a PDF copy.

Usage:  python scripts/06_export_pdf.py [reports/IA1_Report.docx ...]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(paths):
    import win32com.client as win32
    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        for p in paths:
            p = Path(p).resolve()
            doc = word.Documents.Open(str(p))
            for toc in doc.TablesOfContents:
                toc.Update()
            doc.Fields.Update()
            doc.Save()
            doc.SaveAs2(str(p.with_suffix(".pdf")), FileFormat=17)   # wdFormatPDF
            doc.Close(False)
            print("exported", p.with_suffix(".pdf").relative_to(ROOT))
    finally:
        word.Quit()


if __name__ == "__main__":
    targets = sys.argv[1:] or sorted((ROOT / "reports").glob("*.docx"))
    export([t for t in targets if not Path(t).name.startswith("~$")])
