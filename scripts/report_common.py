"""Text shared by the IA-1 report, IA-2 report and manuscript draft."""

from __future__ import annotations

import json

from _common import ROOT

TITLE = "PrivBench-Adult"
LONG_TITLE = ("A Unified Privacy–Utility Evaluation of Differential, Local, Syntactic and "
              "Compressive Privacy for Tabular Machine Learning")
COURSE = "Security and Privacy for Machine Learning"

STUDENT_FIELDS = [
    ("Student name", "Disha Kataria"),
    ("Roll number", "16014223034"),
    ("Programme", "B.Tech Honours — Semester VII"),
    ("Course", COURSE),
    ("Course faculty", "Ashwini Dalvi"),
    ("Department", "Department of Information Technology"),
    ("Institute", "K. J. Somaiya School of Engineering"),
]
SUBMISSION_DATE = "22/10/2026"


def provenance() -> dict:
    return json.loads((ROOT / "data" / "provenance.json").read_text())


# Module mapping used (and refined) in both reports ---------------------------------
MODULE_MAP_HEADER = ["Module", "Concepts used", "Role in this project", "Treatment"]
MODULE_MAP_IA1 = [
    ["M1  Privacy considerations in ML",
     "Threat models; membership inference; attribute inference; re-identification via "
     "quasi-identifiers; proxy attributes; privacy metrics",
     "Defines *what* is being protected and supplies the attack suite used to measure leakage "
     "of every mechanism", "Implemented + evaluated"],
    ["M2  Differential privacy for ML",
     "ε/δ-DP, sensitivity, Laplace & Gaussian mechanisms, composition, DP-SGD, Rényi-DP "
     "accounting", "Protects the *training* stage; provides a formal, worst-case guarantee",
     "Implemented + evaluated"],
    ["M3  Local differential privacy",
     "Randomised response, GRR, OUE, Piecewise mechanism, unbiased frequency estimation",
     "Protects the *collection* stage when no trusted curator exists",
     "Implemented + evaluated"],
    ["M4  Privacy-preserving data mining",
     "Quasi-identifiers, k-anonymity (Mondrian), l-diversity, t-closeness, information loss",
     "Protects the *publishing* stage (release of a sanitised microdata table)",
     "Implemented + evaluated"],
    ["M5  Compressive privacy",
     "Lossy dimensionality reduction (PCA), utility-supervised DCA, privacy-aware projection",
     "Protects the *feature-release* stage: only a low-dimensional projection leaves the owner",
     "Implemented (PCA) + planned (DCA variants)"],
]

REFERENCES = [
    "C. Dwork and A. Roth, “The Algorithmic Foundations of Differential Privacy,” *Foundations and Trends in Theoretical Computer Science*, vol. 9, no. 3–4, pp. 211–407, 2014.",
    "M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar and L. Zhang, “Deep learning with differential privacy,” in *Proc. ACM CCS*, 2016, pp. 308–318.",
    "I. Mironov, “Rényi differential privacy,” in *Proc. IEEE CSF*, 2017, pp. 263–275.",
    "I. Mironov, K. Talwar and L. Zhang, “Rényi differential privacy of the sampled Gaussian mechanism,” arXiv:1908.10530, 2019.",
    "B. Balle, G. Barthe, M. Gaboardi, J. Hsu and T. Sato, “Hypothesis testing interpretations and Rényi differential privacy,” in *Proc. AISTATS*, 2020.",
    "S. L. Warner, “Randomized response: A survey technique for eliminating evasive answer bias,” *J. American Statistical Association*, vol. 60, no. 309, pp. 63–69, 1965.",
    "P. Kairouz, K. Bonawitz and D. Ramage, “Discrete distribution estimation under local privacy,” in *Proc. ICML*, 2016.",
    "T. Wang, J. Blocki, N. Li and S. Jha, “Locally differentially private protocols for frequency estimation,” in *Proc. USENIX Security*, 2017, pp. 729–745.",
    "N. Wang, X. Xiao, Y. Yang, J. Zhao, S. C. Hui, H. Shin, J. Shin and G. Yu, “Collecting and analyzing multidimensional data with local differential privacy,” in *Proc. IEEE ICDE*, 2019, pp. 638–649.",
    "Ú. Erlingsson, V. Pihur and A. Korolova, “RAPPOR: Randomized aggregatable privacy-preserving ordinal response,” in *Proc. ACM CCS*, 2014.",
    "L. Sweeney, “k-anonymity: A model for protecting privacy,” *Int. J. Uncertainty, Fuzziness and Knowledge-Based Systems*, vol. 10, no. 5, pp. 557–570, 2002.",
    "K. LeFevre, D. J. DeWitt and R. Ramakrishnan, “Mondrian multidimensional k-anonymity,” in *Proc. IEEE ICDE*, 2006.",
    "A. Machanavajjhala, D. Kifer, J. Gehrke and M. Venkitasubramaniam, “l-diversity: Privacy beyond k-anonymity,” *ACM TKDD*, vol. 1, no. 1, 2007.",
    "N. Li, T. Li and S. Venkatasubramanian, “t-closeness: Privacy beyond k-anonymity and l-diversity,” in *Proc. IEEE ICDE*, 2007.",
    "J. Xu, W. Wang, J. Pei, X. Wang, B. Shi and A. W.-C. Fu, “Utility-based anonymization using local recoding,” in *Proc. ACM KDD*, 2006.",
    "R. Agrawal and R. Srikant, “Privacy-preserving data mining,” in *Proc. ACM SIGMOD*, 2000.",
    "S. Y. Kung, “Compressive privacy: From information/estimation theory to machine learning,” *IEEE Signal Processing Magazine*, vol. 34, no. 1, pp. 94–112, 2017.",
    "S. Y. Kung, “A compressive privacy approach to generalized information bottleneck and privacy funnel problems,” *Journal of the Franklin Institute*, vol. 355, no. 4, pp. 1846–1872, 2018.",
    "R. Shokri, M. Stronati, C. Song and V. Shmatikov, “Membership inference attacks against machine learning models,” in *Proc. IEEE S&P*, 2017.",
    "S. Yeom, I. Giacomelli, M. Fredrikson and S. Jha, “Privacy risk in machine learning: Analyzing the connection to overfitting,” in *Proc. IEEE CSF*, 2018.",
    "N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis and F. Tramèr, “Membership inference attacks from first principles,” in *Proc. IEEE S&P*, 2022.",
    "A. Narayanan and V. Shmatikov, “Robust de-anonymization of large sparse datasets,” in *Proc. IEEE S&P*, 2008.",
    "B. Becker and R. Kohavi, “Adult” [Dataset], UCI Machine Learning Repository, 1996. doi:10.24432/C5XW20.",
    "R. Kohavi, “Scaling up the accuracy of naive-Bayes classifiers: A decision-tree hybrid,” in *Proc. KDD*, 1996.",
    "F. Ding, M. Hardt, J. Miller and L. Schmidt, “Retiring Adult: New datasets for fair machine learning,” in *Proc. NeurIPS*, 2021.",
    "A. Yousefpour et al., “Opacus: User-friendly differential privacy library in PyTorch,” arXiv:2109.12298, 2021.",
    "N. Papernot and T. Steinke, “Hyperparameter tuning with Rényi differential privacy,” in *Proc. ICLR*, 2022.",
    "E. Bagdasaryan, O. Poursaeed and V. Shmatikov, “Differential privacy has disparate impact on model accuracy,” in *Proc. NeurIPS*, 2019.",
    "S. Ravfogel, Y. Elazar, H. Gonen, M. Twiton and Y. Goldberg, “Null it out: Guarding protected attributes by iterative nullspace projection,” in *Proc. ACL*, 2020.",
    "N. Natarajan, I. S. Dhillon, P. Ravikumar and A. Tewari, “Learning with noisy labels,” in *Proc. NeurIPS*, 2013.",
    "Government of India, *The Digital Personal Data Protection Act, 2023* (Act No. 22 of 2023).",
    "European Parliament and Council, *Regulation (EU) 2016/679 (General Data Protection Regulation)*, 2016.",
]


def references(rep, keys: list[int] | None = None):
    rep.h1("References")
    items = REFERENCES if keys is None else [REFERENCES[i - 1] for i in keys]
    for i, r in enumerate(items, 1):
        rep.para(f"[{i}] {r}", align="left", size=9.5)


def code_structure_rows():
    return [
        ["`scripts/01_prepare_data.py`", "Download UCI archive, SHA-256 hashing, cleaning, provenance record"],
        ["`scripts/02_run_ia1.py`", "IA-1 preliminary experiments P1–P6"],
        ["`scripts/03_run_ia2.py`", "IA-2 full experiments E1–E8 (`--quick` for smoke test)"],
        ["`scripts/04_make_figures.py`", "All figures from the result CSVs"],
        ["`scripts/05_build_reports.py`", "Regenerates these Word reports from results"],
        ["`src/privml/data.py`", "Schema, cleaning, public/private/test splits, encoder fitted on public split"],
        ["`src/privml/attacks.py`", "M1: membership inference, attribute inference, re-identification risk"],
        ["`src/privml/dp.py`", "M2: Laplace/Gaussian mechanisms, DP histogram, RDP accountant, DP-SGD"],
        ["`src/privml/ldp.py`", "M3: GRR, OUE, Piecewise mechanism, LDP record perturbation"],
        ["`src/privml/ppdm.py`", "M4: Mondrian k-anonymity, l-diversity, NCP, homogeneity, t-closeness"],
        ["`src/privml/compressive.py`", "M5: PCA, DCA, Desensitised DCA (D-DCA), reconstruction error"],
        ["`src/privml/models.py`", "NumPy MLP with closed-form per-example gradients; LR/RF baselines"],
        ["`src/privml/experiments.py`", "Experiment definitions shared by IA-1 and IA-2 runners"],
        ["`tests/`", "30 unit tests (accountant vs Opacus, estimator unbiasedness, k-anonymity, gradients)"],
    ]
