# PrivBench-Adult

**A Unified Privacy–Utility Evaluation of Differential, Local, Syntactic and Compressive Privacy for Tabular Machine Learning**

Security and Privacy for Machine Learning — Individual Research-Based Assessment (IA-1 + IA-2)

---

## What this project does

The course's privacy mechanisms protect *different stages* of an ML pipeline. Each is normally evaluated with its own metric (ε, k, reconstruction error), so they are hard to compare. This project applies one mechanism per module at its natural stage, trains **the same model** on every protected output, and attacks every result with **the same attack suite**:

| Module | Mechanism (implemented from scratch) | Pipeline stage |
|---|---|---|
| M1 Privacy considerations | Membership inference, attribute inference, re-identification | evaluation of everything |
| M2 Differential privacy | DP-SGD + Rényi-DP accountant (matches Opacus), Laplace/Gaussian mechanisms | training |
| M3 Local DP | GRR, OUE, Piecewise mechanism | collection |
| M4 PPDM | Mondrian k-anonymity, l-diversity, t-closeness, NCP | publication |
| M5 Compressive privacy | PCA, DCA, **Desensitised DCA (proposed)**, hybrid D-DCA → DP-SGD | feature release |

Dataset: **UCI Adult** (CC BY 4.0, DOI 10.24432/C5XW20), 45,222 records after cleaning. Provenance and SHA-256 hashes are in [data/provenance.json](data/provenance.json).

## Deliverables

| File | Contents |
|---|---|
| [reports/IA1_Report.docx](reports/IA1_Report.docx) (+ PDF) | IA-1: problem, module mapping, concepts, dataset, initial implementation, preliminary results, IA-2 plan |
| [reports/IA2_Report.docx](reports/IA2_Report.docx) (+ PDF) | IA-2: refined problem, final mapping, provenance, full implementation, experiments E1–E8, critical analysis |
| [reports/Research_Manuscript_Draft.docx](reports/Research_Manuscript_Draft.docx) (+ PDF) | 5-page conference-style paper draft |
| `figures/`, `results/` | All figures (300 dpi) and raw result CSVs with run logs |

**Before submitting:** fill in the placeholders on each title page and in the manuscript author line (`[Your Name]`, `[Your Roll Number]`, `[Faculty Name]`, `[Institute Name]`, date). Also complete the bracketed tools/assistance declaration under each individual-contribution statement, following your institute's policy.

## Reproduce everything

Requires Python ≥ 3.10. Runs on a laptop CPU; no GPU or deep-learning framework is needed.

```bash
pip install -r requirements.txt
```

```bash
python scripts/01_prepare_data.py
```

```bash
python scripts/02_run_ia1.py
```

```bash
python scripts/03_run_ia2.py
```

```bash
python scripts/04_make_figures.py
```

```bash
python scripts/05_build_reports.py
```

```bash
python -m pytest
```

| Step | Time | Output |
|---|---|---|
| 01 prepare data | ~10 s | `data/raw`, `data/processed`, `data/provenance.json` |
| 02 IA-1 experiments | ~1 min | `results/ia1/` |
| 03 IA-2 experiments | ~13 min (`--quick` ≈ 4 min) | `results/ia2/` |
| 04 figures | ~20 s | `figures/ia1`, `figures/ia2` |
| 05 reports | ~10 s | `reports/*.docx` (every number is read from `results/`) |
| 06 PDF export (optional, needs MS Word) | ~30 s | `reports/*.pdf`, refreshes the table of contents |

If you edit a `.docx` by hand, keep a copy. Re-running step 05 regenerates the reports from scratch.

## Key results (IA-2, mean over 3–5 seeds)

| Configuration | Accuracy | MIA AUC | Sex inference (bal. acc.) | Re-id risk |
|---|---|---|---|---|
| Non-private MLP | 0.850 | 0.503 | 1.000 | 0.381 |
| DP-SGD ε = 1 | 0.851 | 0.499 | n/a | n/a |
| DP-SGD ε = 0.1 | 0.815 | 0.499 | n/a | n/a |
| Mondrian k = 10, l = 2 | 0.849 | 0.505 | 1.000 | 0.039 |
| LDP ε_total = 64 | 0.831 | 0.506 | 0.994 | — |
| D-DCA m = 5 (proposed) | 0.849 | 0.500 | 0.706 | — |
| Hybrid D-DCA-5 + DP-SGD ε = 0.1 | 0.816 | 0.497 | 0.706 | — |

Findings:

- DP-SGD costs almost nothing above ε ≈ 0.5. Its empirical MIA reduction comes mostly from clipping.
- k-anonymity protects identity, but without l-diversity it discloses income for 30% of records at k = 10, and it never hides sex.
- Record-level LDP needs a very large ε for utility.
- D-DCA removes linear attribute leakage at negligible cost.
- The hybrid does **not** beat DP-SGD on accuracy (a negative result, reported honestly). Its value is that it combines a formal model guarantee with attribute protection.

## Repository layout

```
src/privml/        library: data, attacks (M1), dp (M2), ldp (M3), ppdm (M4), compressive (M5),
                   models (NumPy MLP with closed-form per-example gradients), experiments
scripts/           01–06 pipeline steps + report generators
tests/             30 unit tests (accountant vs Opacus, estimator unbiasedness, k-anonymity, gradients, hashes)
data/              raw UCI files, cleaned CSVs, provenance.json
results/           CSV outputs, run logs, environment records
figures/           IA-1 and IA-2 figures
reports/           Word + PDF deliverables
assignment_brief/  assignment and rubric PDFs
archive/           the earlier "Unauthorized Data Use in LLMs" prototype (superseded)
```
