"""
Step 5 — Build the Word reports from the result CSVs and figures.

Usage:  python scripts/05_build_reports.py [ia1 ia2 manuscript]

Outputs: reports/IA1_Report.docx, reports/IA2_Report.docx,
         reports/Research_Manuscript_Draft.docx
All numbers in the reports are read from results/ — rerunning the experiments
and this script keeps text, tables and figures consistent.
"""

import sys

from _common import ROOT

import report_ia1

OUT = ROOT / "reports"


def main(which):
    if "ia1" in which:
        report_ia1.build(OUT / "IA1_Report.docx")
    if "ia2" in which:
        import report_ia2
        report_ia2.build(OUT / "IA2_Report.docx")
    if "manuscript" in which:
        import report_manuscript
        report_manuscript.build(OUT / "Research_Manuscript_Draft.docx")


if __name__ == "__main__":
    main(sys.argv[1:] or ["ia1", "ia2", "manuscript"])
