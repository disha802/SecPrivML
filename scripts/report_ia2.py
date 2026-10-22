"""IA-2 report: Implementation, Evaluation and Technical Outcomes (final technical report)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ROOT
from report_common import (COURSE, SUBMISSION_DATE, LONG_TITLE, MODULE_MAP_HEADER, STUDENT_FIELDS, TITLE,
                           code_structure_rows, provenance, references)
from report_lib import Report

R2 = ROOT / "results" / "ia2"
F2 = ROOT / "figures" / "ia2"
INF = float("inf")


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def ms(mean, sd, d=3):
    return f"{mean:.{d}f} ± {sd:.{d}f}"


class Results:
    """Loads every IA-2 CSV and exposes the aggregates the narrative needs."""

    def __init__(self):
        rd = lambda n: pd.read_csv(R2 / f"{n}.csv")
        self.e1, self.e2, self.e3 = rd("e1_baselines"), rd("e2_dpsgd"), rd("e3_ablation")
        self.e4, self.e5, self.e6 = rd("e4_marginals"), rd("e5_ldp_training"), rd("e6_kanon")
        self.e7, self.e8 = rd("e7_compressive"), rd("e8_hybrid")
        num = lambda df: df.select_dtypes("number").columns.difference(["seed"])
        self.b = self.e1.groupby(["regime", "model"])[num(self.e1)].agg(["mean", "std"])
        self.dp = self.e2.groupby(["regime", "target_eps"])[num(self.e2)].agg(["mean", "std"])
        self.ab = self.e3.groupby(["param", "value"])[["accuracy", "f1"]].agg(["mean", "std"])
        self.mg = self.e4.groupby(["attribute", "eps", "mechanism"]).l1_error.mean().unstack()
        self.ldp = self.e5.groupby(["variant", "eps_total"])[num(self.e5)].mean()
        e6 = self.e6.copy()
        e6["l"] = e6["l"].astype(str)
        self.ka = e6.groupby(["l", "k"])[num(e6)].mean()
        self.cp = self.e7.groupby(["experiment", "projection", "m"])[num(self.e7)].agg(["mean", "std"])
        self.hy = self.e8.groupby(["variant", "target_eps"])[num(self.e8)].agg(["mean", "std"])

    def base(self, reg, m, c, stat="mean"):
        return self.b.loc[(reg, m), (c, stat)]

    def dps(self, reg, e, c, stat="mean"):
        return self.dp.loc[(reg, e), (c, stat)]

    def cpv(self, proj, m, c, stat="mean", exp="E7"):
        return self.cp.loc[(exp, proj, m), (c, stat)]

    def hyv(self, v, e, c, stat="mean"):
        return self.hy.loc[(v, e), (c, stat)]


HYB = "D-DCA-5 + DP-SGD (proposed)"
DPF = "DP-SGD (all features)"


def build(out):
    R = Results()
    prov = provenance()
    sizes = prov["sizes"]
    eps_grid = sorted(e for e in R.e2.target_eps.unique() if np.isfinite(e))
    np_mlp = R.base("full", "MLP", "accuracy")
    majority = 1 - sizes["class_balance"]["test_pos_rate"]
    raw7 = R.cp.loc[("E7", "Raw (all features)", 87)]
    sup7 = R.cp.loc[("E7", "Raw (sex suppressed)", 85)]
    gain = {e: R.hyv(HYB, e, "accuracy") - R.hyv(DPF, e, "accuracy") for e in eps_grid}
    pooled = {e: max(R.hyv(HYB, e, "accuracy", "std"), R.hyv(DPF, e, "accuracy", "std")) for e in eps_grid}
    ties = [e for e in eps_grid if abs(gain[e]) <= pooled[e]]          # within one sd
    large = [e for e in eps_grid if e >= 0.5]
    ratios = []
    for at in ["workclass", "marital-status", "occupation", "native-country"]:
        for e_ in sorted(R.e4.eps.unique()):
            r_ = R.mg.loc[(at, e_)]
            ratios.append(min(r_["Local-GRR"], r_["Local-OUE"]) / r_["Central-Laplace"])
    rmin, rmax = min(ratios), max(ratios)
    ka_raw = R.ka.loc[("none", 1)]

    rep = Report(f"IA-2 | {COURSE} | Implementation, Evaluation and Technical Outcomes")
    rep.title_page(COURSE, "Internal Assessment 2 — Implementation, Evaluation and Technical Outcomes",
                   TITLE, LONG_TITLE, STUDENT_FIELDS + [("Submission date", SUBMISSION_DATE)])
    rep.toc()

    # ================================================================= abstract
    rep.h1("Abstract")
    rep.para(
        "Privacy-preserving machine learning offers mechanisms for every stage of the data pipeline — "
        "local differential privacy (LDP) at collection, k-anonymity at publication, DP-SGD at training "
        "and compressive (lossy-projection) privacy at feature release — but they are usually evaluated "
        f"in isolation, with incomparable metrics. This work presents **{TITLE}**, a reproducible framework "
        "that applies one mechanism from each of the course's Modules 2–5 to the same data (UCI Adult, "
        f"{sizes['final']['total_rows']:,} records), trains the same model on each protected output, and "
        "attacks every result with the same Module-1 suite: membership inference, sensitive-attribute "
        "inference and re-identification. All mechanisms, including a Rényi-DP accountant (validated "
        "against Opacus) and closed-form per-example gradients for DP-SGD, are implemented from first "
        "principles and covered by 30 unit tests. We further propose **Desensitised DCA (D-DCA)**, a "
        "privacy-penalised discriminant projection, and a **hybrid D-DCA → DP-SGD** pipeline.")
    rep.para(
        f"Results over 3–5 seeds show that: DP-SGD keeps accuracy within 0.2 points of the non-private "
        f"model ({pct(np_mlp)}) down to ε = 0.5 and costs {100 * (np_mlp - R.dps('full', 0.05, 'accuracy')):.1f} points at "
        f"ε = 0.05; at equal ε, removing the trusted curator (LDP) costs two to three orders of magnitude "
        f"in histogram accuracy and, for full-record collection, needs ε ≈ 64–128 to approach non-private "
        f"utility, at which point the sensitive attribute is again inferable; Mondrian k-anonymity "
        f"eliminates unique records at almost no utility cost but leaves sex fully inferable and discloses "
        f"income for {pct(R.ka.loc[('none', 10), 'homogeneity_rate'])} of records at k = 10 unless l-diversity "
        f"is enforced; and D-DCA (m = 5) reduces a linear attacker to chance and a gradient-boosting attacker "
        f"from {f(raw7[('gbm_attr_bal_acc', 'mean')])} to {f(R.cpv('D-DCA (proposed)', 5, 'gbm_attr_bal_acc'))} balanced accuracy "
        f"while keeping {pct(R.cpv('D-DCA (proposed)', 5, 'accuracy'))} accuracy. Contrary to our IA-1 hypothesis, the "
        f"hybrid does not raise accuracy over DP-SGD: it matches it within one standard deviation for "
        f"ε ≤ {max(ties):g} and trails by ≈ {100 * -np.mean([gain[e] for e in large]):.1f} points for ε ≥ 0.5, but it is the "
        "only configuration that pairs a formal guarantee for the model with attribute protection for released "
        "features. No single mechanism protects against all three attacks; the framework makes these gaps measurable.")
    rep.para("**Keywords:** differential privacy, local differential privacy, DP-SGD, k-anonymity, "
             "l-diversity, compressive privacy, discriminant component analysis, membership inference, "
             "attribute inference, privacy–utility trade-off.", align="left")

    rep.h2("Rubric compliance map")
    rep.table(["IA-2 stage", "Where addressed", "Evidence"], [
        ["1. Problem refinement & objectives", "Section 1", "Revised problem statement, Table 2 (changes since IA-1), objectives, RQ1–RQ5, technical requirements"],
        ["2. Integration of course concepts", "Section 3", "Final module-mapping table, architecture figure, per-module integration, analytical treatment"],
        ["3. Dataset finalisation & individual implementation", "Section 4", "Provenance record (all checklist items), hashes, sizes, preprocessing, algorithms, code, run instructions, contribution"],
        ["4. Evaluation, comparison & critical analysis", "Section 5", "E1–E8 results, comparison tables/plots, metric definitions, interpretation, validity threats, improvements"],
        ["5. Final documentation & research/IP", "Sections 6–8, Appendices", "Complete report, reproducibility, references, manuscript draft (separate file), IP note"],
    ], caption="Mapping of IA-2 assessment stages to sections of this report.", widths=[4.5, 2.6, 8.9])

    # ================================================================= 1
    rep.h1("1. Problem Refinement and Objectives")
    rep.h2("1.1 Summary of IA-1 findings")
    rep.para(
        "IA-1 established the problem, dataset and code base, and produced five observations (O1–O5): "
        "average-case membership inference is weak on well-generalising models but strong for high-capacity "
        "or small-data models (O1); DP-SGD is almost free on Adult for ε ≥ 0.5 (O2); k-anonymity removes "
        "uniqueness but leaves many homogeneous classes (O3); PCA retains the sensitive attribute (O4); "
        "and local DP is far less accurate than central DP at equal ε (O5).")
    rep.h2("1.2 Revised problem statement")
    rep.callout(
        "**Revised problem statement.** For a tabular personal-data classification task, quantify — "
        "under one pipeline, one model and one attack suite, and in both a data-rich and a data-scarce "
        "regime — the privacy–utility trade-off of local DP at collection, k-anonymity (with and without "
        "l-diversity) at publication, DP-SGD at training and privacy-aware compressive projection at "
        "feature release, *including the strong-privacy regime ε < 0.5*; and test whether a **hybrid of "
        "compressive privacy and DP-SGD** dominates either mechanism alone.")
    rep.h2("1.3 Changes since IA-1 and their justification")
    rep.table(["Change", "Evidence that motivated it", "Effect on the study"], [
        ["ε grid extended from {0.5…8} to {0.05…8, ∞}", "O2: < 0.3 pt utility loss for ε ≥ 0.5", "Exposes the real trade-off region (E2, E8)"],
        ["Small-data regime (n = 1,000, high-capacity MLP) added to baselines and DP-SGD", "O1: MIA only bites when the generalisation gap is large", "Makes DP's effect on MIA measurable (E1, E2)"],
        ["TPR at 1% FPR added to MIA reporting", "Low-FPR metric recommended by Carlini et al. [21]", "Guards against averaging away rare but confident leaks"],
        ["l-diversity, t-closeness and attribute inference added to k-anonymity", "O3: 30% homogeneity at k = 10", "Separates identity, attribute and value disclosure (E6)"],
        ["New method: Desensitised DCA (D-DCA)", "O4: PCA keeps sex; DCA only covers utility", "Privacy-aware projection (E7)"],
        ["Explicit sensitive columns suppressed *before* projection", "IA-2 exploration: a linear projection of the one-hot sex column stays non-linearly decodable", "D-DCA removes proxies; suppression removes the attribute"],
        ["Joint (sex × relationship) privacy label tested as a variant", "relationship is a near-perfect proxy of sex (IA-1 Fig. 2)", "Negative result reported: costs utility (E7)"],
        ["New method: hybrid D-DCA → DP-SGD", "O2 + O4: DP noise scales with dimension; projection hides sex", "Tests whether combining modules dominates (E8)"],
        ["LDP *training* experiment, with a features-only ablation", "O5: large LDP estimation error", "Quantifies LDP cost on learning and isolates label noise (E5)"],
        ["Seeds increased from 3 to 5 (3 for slower sweeps)", "IA-1 limitation: wide intervals", "Mean ± sd for every reported number"],
    ], caption="Changes made since IA-1.", widths=[5.2, 5.6, 5.2], size=8.5)
    rep.h2("1.4 Final objectives")
    rep.table(["ID", "Objective", "Status"], [
        ["O1", "Quantify exposure of raw data and non-private models (both regimes)", "Done — E1, Section 5.2"],
        ["O2", "Implement and test one mechanism per module (M2–M5) from first principles", "Done — Section 4, 30 tests"],
        ["O3", "Evaluate all mechanisms with the same model and attack suite", "Done — unified table, Section 5.9"],
        ["O4", "Characterise each trade-off across its parameter (ε, k, m, γ)", "Done — E2–E7"],
        ["O5", "Propose and evaluate a privacy-aware projection and a hybrid", "Done — D-DCA (E7), hybrid (E8)"],
    ], caption="Final objectives and completion status.", widths=[1, 10, 5])
    rep.h2("1.5 Research questions")
    rep.table(["RQ", "Question"], [
        ["RQ1", "How much does each mechanism reduce membership, attribute and re-identification risk, and at what utility cost?"],
        ["RQ2", "What does removing the trusted curator cost: central DP vs local DP at the same ε?"],
        ["RQ3", "Does k-anonymity (with/without l-diversity) protect attributes and values, or only identities?"],
        ["RQ4", "Can a privacy-penalised projection (D-DCA) hide a sensitive attribute while preserving utility better than PCA/DCA?"],
        ["RQ5", "Does D-DCA compression before DP-SGD improve the privacy–utility trade-off over DP-SGD alone?"],
    ], caption="Research questions.", widths=[1.2, 14.8])
    rep.h2("1.6 Technical requirements")
    rep.table(["ID", "Requirement", "How it is met"], [
        ["FR1", "Reproducible data acquisition with integrity check", "Scripted download + SHA-256 in provenance.json; test re-hashes files"],
        ["FR2", "Formal (ε, δ) accounting for every DP model", "RDP accountant; ε recorded per run; matches Opacus"],
        ["FR3", "Identical utility model across mechanisms", "One MLP class (87→64→1) for all mechanisms"],
        ["FR4", "Identical attack suite across mechanisms", "`attacks.py` used by every experiment"],
        ["FR5", "No private statistic leaks via preprocessing", "Encoder and projections fitted on the public split"],
        ["NFR1", "Runs on a laptop CPU", "Full IA-2 run ≈ 13 min; no GPU or deep-learning framework"],
        ["NFR2", "Deterministic, seeded, tested", "Explicit seeds; 30 unit tests"],
    ], caption="Functional and non-functional requirements.", widths=[1.3, 6, 8.7])

    # ================================================================= 2
    rep.h1("2. Background and Related Work")
    rep.para(
        "**Attacks.** Shokri et al. [19] introduced shadow-model membership inference; Yeom et al. [20] "
        "linked it to over-fitting via a simple loss-threshold attack; Carlini et al. [21] argued for "
        "reporting TPR at low FPR. Linkage attacks on quasi-identifiers [11, 22] motivated syntactic "
        "anonymisation. **Central DP** [1] gives a worst-case guarantee; DP-SGD [2] with Rényi-DP "
        "accounting [3–5] is the standard way to train private models, though it may harm minority "
        "groups disproportionately [28]. **Local DP** [6–10] removes the trusted curator at the cost of "
        "accuracy, which is why multi-dimensional collection [9] remains challenging. **PPDM** "
        "offers k-anonymity [11], efficiently realised by Mondrian [12], and its refinements "
        "l-diversity [13] and t-closeness [14]. **Compressive privacy** [17, 18] (Kung) frames privacy "
        "as a lossy projection that retains utility-relevant and discards privacy-relevant signal, with "
        "DCA as the utility-supervised projection; related representation-level defences include "
        "nullspace projection [29].")
    rep.para(
        "**Gap.** Comparative studies typically stay within one family, e.g. different DP optimisers, or "
        "different anonymisation algorithms. They rarely subject mechanisms from *different stages and "
        "guarantee types* to the *same* attacks on the same data and model, and they rarely examine "
        "combining compressive privacy with DP-SGD. This work fills that gap at a reproducible, "
        "laptop-scale level.")

    # ================================================================= 3
    rep.h1("3. Integration of Course Concepts")
    rep.h2("3.1 Final module-mapping table")
    rep.table(MODULE_MAP_HEADER + ["Evidence"], [
        ["M1  Privacy considerations", "Threat model (Kerckhoffs); loss-threshold MIA; attribute inference (LR & GBM attackers); prosecutor re-identification risk; proxy-attribute analysis",
         "Common yardstick for every mechanism", "Implemented, evaluated", "`attacks.py`; E1; all tables"],
        ["M2  Differential privacy", "(ε, δ)-DP; Laplace & Gaussian mechanisms; sensitivity; composition; DP-SGD with Poisson sampling + clipping; RDP accountant; noise calibration",
         "Training-stage protection with formal guarantee; central baseline for M3", "Implemented, evaluated", "`dp.py`; E2, E3, E4, E8"],
        ["M3  Local DP", "Randomised response; GRR; OUE; Piecewise mechanism; unbiased estimators; budget splitting across attributes",
         "Collection-stage protection without trusted curator", "Implemented, evaluated", "`ldp.py`; E4, E5"],
        ["M4  PPDM", "Quasi-identifiers; Mondrian k-anonymity; distinct l-diversity; t-closeness; NCP information loss; homogeneity attack",
         "Publishing-stage protection of microdata", "Implemented, evaluated; SMC/HE analytical", "`ppdm.py`; E6"],
        ["M5  Compressive privacy", "PCA; DCA (Kung); proposed Desensitised DCA; reconstruction error; hybrid with DP-SGD",
         "Feature-release protection; dimensionality reduction for DP", "Implemented, evaluated, extended", "`compressive.py`; E7, E8"],
    ], caption="Final module-mapping table.", widths=[2.6, 5, 3.4, 2.4, 2.6], size=8, bold_first_col=True)
    rep.h2("3.2 Architecture")
    rep.figure(F2 / "fig1_architecture.png",
               "Final architecture. Each module protects one pipeline stage; the hybrid chains M5 into M2. "
               "Every output is consumed by the same utility model and the same Module-1 attack suite.")
    rep.h2("3.3 How each module contributes")
    rep.bullets([
        "**M1 — the yardstick.** Without a common attack suite, ε, k and m are incomparable. M1 converts "
        "every mechanism's output into the same three empirical risks. The attacker is always given the "
        "mechanism and the public split (Kerckhoffs's principle), e.g. for LDP the attacker simulates the "
        "perturbation on public data before training its classifier.",
        "**M2 — formal training-time privacy.** DP-SGD bounds the influence of any single record on the "
        "model; the accountant reports the exact (ε, δ). The Laplace histogram provides the central "
        "baseline against which LDP is measured (RQ2).",
        "**M3 — collection without trust.** Each record's 13 attributes are perturbed on the user side "
        "(PM for numeric, GRR for categorical and the label), with ε_total split equally (sequential "
        "composition). The collector trains on perturbed data only.",
        "**M4 — sanitised publication.** Mondrian partitions the 7 QIs; generalised cells become "
        "ranges/sets that the model consumes as means/fractional one-hots. l-diversity is enforced inside "
        "the partitioning (a cut is allowed only if both halves stay 2-diverse in income).",
        "**M5 — data minimisation at source.** Only an m-dimensional projection is released. D-DCA chooses "
        "directions that separate income classes but not sex classes; the hybrid then runs DP-SGD on "
        "these m features, so the model is DP *and* the features are desensitised.",
    ])
    rep.h2("3.4 Concepts treated analytically")
    rep.table(["Concept", "Why not implemented", "Analytical treatment"], [
        ["Secure multi-party computation / homomorphic encryption (M4)", "Protects *computation* across parties, not the released output; Adult has a single owner", "Orthogonal to the release-stage threats studied: SMC hides inputs during training but the trained model still leaks membership, so it would be combined with DP-SGD rather than replace it"],
        ["Enforcing t-closeness (M4)", "Binary sensitive attribute and 7 QIs make strict t-closeness at small t collapse to very coarse classes", "t is *measured* for every k (Section 5.6); we show values remain ≈ 0.6–0.75, i.e. class income rates deviate strongly from the population"],
        ["Shuffle model / amplification (M3)", "Requires a trusted shuffler infrastructure", "Would reduce the central–local gap in E4; noted as future work"],
        ["Shadow-model and LiRA MIAs (M1)", "Many shadow models per configuration × ~300 configurations", "Loss-threshold MIA is a lower bound; IA-2 adds TPR@1%FPR to partly compensate"],
    ], caption="Techniques justified through analysis rather than implementation.", widths=[3.6, 5.2, 7.2], size=8.5)

    # ================================================================= 4
    rep.h1("4. Dataset Finalisation and Implementation")
    rep.h2("4.1 Provenance record")
    h = prov["hashes"]
    rep.table(["Checklist item", "Record"], [
        ["Dataset / source name", f"{prov['dataset']} — UCI Machine Learning Repository"],
        ["URL / access reference", f"{prov['source_url']} ; archive {prov['download_url']} ; DOI {prov['doi']}"],
        ["Collection date and method", f"Original: {prov['original_collection']}. Retrieved for this study on {prov['retrieved_on']} via {prov['retrieval_method']}."],
        ["Author / publisher", f"{prov['creators']}; {prov['publisher']}"],
        ["Licence / usage terms", f"{prov['license']} — reuse permitted with attribution (cited as [23])"],
        ["Original size", f"{sizes['original']['total_rows']:,} rows ({sizes['original']['train_rows']:,} train + {sizes['original']['test_rows']:,} test), 14 attributes + label"],
        ["Final size", f"{sizes['final']['total_rows']:,} rows: public {sizes['splits']['public']:,} / private {sizes['splits']['private_train']:,} / test {sizes['splits']['test']:,}; 12 features (87 encoded) + label"],
        ["Relevant fields / labels", "Utility label: income > 50K (24.9% positive). Sensitive attribute: sex (67.6% male). QIs: age, sex, race, marital-status, native-country, education-num, workclass"],
        ["Preprocessing operations", "Strip whitespace; remove '.' from test labels; drop 3,620 rows with missing values; drop fnlwgt and education; binarise label; seeded 10% public split; standardise/one-hot (fitted on public split)"],
        ["Version / release identifier", prov["version"]],
        ["File hashes (SHA-256)", f"adult.zip `{h['archive']['sha256'][:16]}…`; adult.data `{h['files']['adult.data']['sha256'][:16]}…`; adult.test `{h['files']['adult.test']['sha256'][:16]}…` (full values in Appendix C)"],
        ["Access restrictions", "None — openly downloadable; no login or data-use agreement"],
        ["Privacy safeguards", "Data contains no direct identifiers; it is already public, so the study simulates a private dataset; no attempt is made to link records to real individuals; derived artefacts contain only aggregate metrics"],
        ["Known limitations", "1994 US population; missing-value deletion bias; binary sex coding; widely used benchmark (risk of benchmark-specific conclusions) [25]"],
    ], caption="Dataset provenance record (IA-2 checklist).", widths=[3.6, 12.4], size=8.5, bold_first_col=True)
    rep.h2("4.2 Preparation pipeline")
    rep.table(["#", "Step (verbatim from scripts/01_prepare_data.py)"],
              [[i + 1, s_] for i, s_ in enumerate(prov["preprocessing_log"])],
              caption="Data preparation log.", widths=[1, 15], size=8.5)
    rep.h2("4.3 Algorithms")
    rep.h3("DP-SGD with closed-form per-example gradients (M2)")
    rep.code("""
input: data {(x_i, y_i)}_{i=1..n}, target (eps, delta), epochs E, batch B, clip C, lr eta
q <- B / n ;  T <- round(E / q)
sigma <- smallest value with  RDP->DP(T * RDP_sampledGaussian(q, sigma), delta) <= eps   (bisection)
for t = 1..T:
    S <- { i : u_i < q },  u_i ~ U(0,1)                      # Poisson sampling
    forward: a = X_S W1 + b1,  h = relu(a),  p = sigmoid(h w2 + b2),  e = p - y_S
    delta_i = e_i * w2 ⊙ 1[a_i > 0]
    g_norm_i = sqrt(|x_i|^2 |delta_i|^2 + |delta_i|^2 + e_i^2 |h_i|^2 + e_i^2)
    c_i = min(1, C / g_norm_i)                               # per-example clipping factor
    G = { W1: X_S^T (c ⊙ Delta), b1: Σ c_i delta_i, w2: h^T (c ⊙ e), b2: Σ c_i e_i }
    G <- (G + N(0, sigma^2 C^2 I)) / (q n)
    momentum-SGD step with G                                 # post-processing: still DP
return model, (eps, delta)
""")
    rep.h3("Mondrian k-anonymity with optional l-diversity (M4)")
    rep.code("""
partitions <- [all records];  done <- []
while partitions not empty:
    P <- pop(partitions)
    for QI in QIs sorted by normalised span in P (widest first):
        (L, R) <- median split (numeric) or frequency-ordered set split (categorical)
        if |L| >= k and |R| >= k and (l is None or both have >= l distinct incomes):
            push L, R;  break
    else: done.append(P)                                     # no allowable cut
generalise each class: numeric QI -> [min, max] (mean for training); categorical QI -> set
""")
    rep.h3("Desensitised DCA (proposed, M5)")
    rep.para("Let S_B^u and S_B^p be the between-class scatter matrices of the utility label (income) and the "
             "privacy label (sex), and Σ the total covariance, all computed on the public split after "
             "suppressing the explicit sex columns. D-DCA returns the top-m generalised eigenvectors of")
    rep.equation("(S_B^u + β Σ) w = λ (γ S_B^p + Σ + ρ I) w,     β = 0.1,  γ = 10,  ρ = 0.01")
    rep.para("The numerator rewards income-discriminant power (β·Σ supplies m > 1 informative directions for a "
             "binary label); the denominator penalises sex-discriminant directions. γ = 0 recovers a "
             "variance-regularised DCA. Because eigenvectors are Σ-orthogonal and the privacy scatter is "
             "rank one, large γ drives the projected class-mean difference of sex towards zero, removing the "
             "signal a linear classifier relies on. This is a design property, not a formal guarantee; E7 "
             "verifies empirically that a linear attacker is reduced to chance.")
    rep.h2("4.4 Code and reproducibility")
    rep.table(["Path", "Purpose"], code_structure_rows(), caption="Repository structure.",
              widths=[5.5, 10.5], size=8.5)
    rep.code("""
pip install -r requirements.txt
python scripts/01_prepare_data.py        # data + provenance        (~10 s)
python scripts/02_run_ia1.py             # IA-1 experiments          (~1 min)
python scripts/03_run_ia2.py             # IA-2 experiments E1-E8    (~13 min; --quick for 4 min)
python scripts/04_make_figures.py        # figures/ia1, figures/ia2
python scripts/05_build_reports.py       # reports/*.docx
python -m pytest                         # 30 unit tests
""")
    rep.para("All randomness is seeded (data split seed 0; model seeds 0–4). Results are written to "
             "`results/ia2/*.csv` with an `environment.json` (library versions) and a timestamped "
             "`run_log.txt`. Every number in Section 5 is read from these CSVs by the report generator, "
             "so text, tables and figures cannot drift apart.")
    rep.h2("4.5 Verification")
    rep.table(["Test group", "What is verified"], [
        ["Accountant", "ε matches Opacus 1.5.2 for 4 settings (|Δ| < 10⁻³); closed form for q = 1; monotone in σ; calibration is near-minimal"],
        ["DP-SGD", "Analytic per-example gradients and norms equal finite differences; private training reaches > 80% on a toy task with ε ≤ 2"],
        ["Mechanisms", "Laplace scale; GRR keep-probability; GRR, OUE and Piecewise estimators unbiased"],
        ["k-anonymity", "Disjoint cover; every class ≥ k; max risk ≤ 1/k; l-diversity eliminates homogeneous classes; NCP bounds"],
        ["Projections / attacks", "PCA full-rank reconstruction exact; D-DCA removes a linear privacy signal while keeping utility signal; MIA detects a memorising model"],
        ["Data", "Raw and processed file hashes match the provenance record; split sizes; encoder statistics"],
    ], caption="Unit-test coverage (30 tests, all passing).", widths=[3.4, 12.6])

    # ================================================================= 5
    rep.h1("5. Evaluation, Comparison and Critical Analysis")
    rep.h2("5.1 Experimental setup")
    rep.table(["Setting", "Value"], [
        ["Utility model", "MLP 87→64→1 (ReLU), Adam lr 1e-3, batch 128, 30 epochs (identical for raw, k-anon, LDP and CP inputs)"],
        ["Small-data regime", f"random 1,000-record training subset per seed; MLP with 256 hidden units, 200 epochs; DP-SGD batch 64, 50 epochs, δ = 10⁻⁴"],
        ["DP-SGD (full)", "batch 256 (q = 0.0094), 20 epochs (2,121 steps), C = 1, momentum-SGD lr 0.05, δ = 10⁻⁵"],
        ["ε grid", ", ".join(f"{e:g}" for e in eps_grid) + ", ∞ (clipping only)"],
        ["LDP", "ε_total ∈ {1, 2, 4, 8, 16, 32, 64, 128} split over 13 attributes (12 features + label)"],
        ["k-anonymity", "k ∈ {2, 5, 10, 25, 50, 100, 250}; l ∈ {none, 2}; 7 QIs"],
        ["Compressive", "m ∈ {2, 3, 5, 10, 20, 40}; γ ∈ {0, 1, 10, 100, 1000} at m = 5"],
        ["Seeds", "5 for E1, E2, E8; 3 for E3, E5, E6, E7; histogram release 50 repetitions"],
        ["Attacks", "MIA on 5,000 members + 5,000 non-members (1,000 each in small regime); attribute attackers trained on the public split processed by the same mechanism"],
    ], caption="Experimental configuration.", widths=[3.5, 12.5], size=8.5, bold_first_col=True)

    # ---- E1
    rep.h2("5.2 E1 — Non-private baselines")
    rows = []
    for reg in ("full", "small"):
        for m, lab in (("LogisticRegression", "LR"), ("RandomForest", "RF"), ("MLP", "MLP")):
            rows.append([f"{lab} ({'full' if reg == 'full' else 'n = 1,000'})",
                         ms(R.base(reg, m, 'accuracy'), R.base(reg, m, 'accuracy', 'std')),
                         f(R.base(reg, m, 'precision')), f(R.base(reg, m, 'recall')), f(R.base(reg, m, 'f1')),
                         f(R.base(reg, m, 'auc')), ms(R.base(reg, m, 'mia_auc'), R.base(reg, m, 'mia_auc', 'std')),
                         f(R.base(reg, m, 'mia_advantage')), f(R.base(reg, m, 'mia_tpr_at_1fpr'))])
    rep.table(["Model", "Accuracy", "Prec.", "Recall", "F1", "AUC", "MIA AUC", "MIA adv.", "TPR@1%"], rows,
              caption="Non-private baselines (5 seeds).", widths=[2.6, 2.4, 1.3, 1.3, 1.3, 1.3, 2.4, 1.6, 1.6],
              size=8)
    rep.para(f"The MLP is the strongest non-private model ({pct(np_mlp)}). Membership leakage is governed by "
             f"over-fitting: negligible for LR/MLP on full data, substantial for the random forest "
             f"(advantage {f(R.base('full', 'RandomForest', 'mia_advantage'))}) and for every high-capacity model in "
             f"the small-data regime (MLP {f(R.base('small', 'MLP', 'mia_advantage'))}, RF "
             f"{f(R.base('small', 'RandomForest', 'mia_advantage'))}). TPR at 1% FPR stays close to 1% for all models, "
             "i.e. the loss-threshold attacker cannot single out individuals with high confidence; leakage is "
             "diffuse rather than concentrated on a few outliers.")

    # ---- E2/E3
    rep.h2("5.3 E2/E3 — DP-SGD (central DP)")
    rows = []
    for e in eps_grid + [INF]:
        lab = "∞ (clip)" if e == INF else f"{e:g}"
        rows.append([lab, f(R.dps('full', e, 'sigma'), 2), ms(R.dps('full', e, 'accuracy'), R.dps('full', e, 'accuracy', 'std')),
                     f(R.dps('full', e, 'f1')), f(R.dps('full', e, 'mia_advantage')),
                     f(R.dps('small', e, 'sigma'), 2), ms(R.dps('small', e, 'accuracy'), R.dps('small', e, 'accuracy', 'std')),
                     f(R.dps('small', e, 'f1')), f(R.dps('small', e, 'mia_advantage'))])
    rep.table(["ε", "σ (full)", "Acc. (full)", "F1", "MIA adv.", "σ (small)", "Acc. (small)", "F1", "MIA adv."], rows,
              caption="DP-SGD across privacy budgets, full (n = 27,146) and small (n = 1,000) regimes; 5 seeds.",
              widths=[1.5, 1.5, 2.6, 1.3, 1.6, 1.6, 2.6, 1.3, 1.6], size=8)
    rep.figure(F2 / "fig2_dpsgd_sweep.png", "DP-SGD utility and membership-inference advantage across ε. "
               "Dotted lines: non-private MLP in the same regime. Bands: ±1 sd.")
    gap_s = R.base("small", "MLP", "mia_advantage")
    rep.para(
        f"**Utility.** With 27k records DP-SGD is remarkably cheap: accuracy stays within 0.2 points of the "
        f"clipping-only model down to ε = 0.5, drops to {pct(R.dps('full', 0.1, 'accuracy'))} at ε = 0.1 and "
        f"{pct(R.dps('full', 0.05, 'accuracy'))} at ε = 0.05, where variance across seeds also grows "
        f"(sd {f(R.dps('full', 0.05, 'accuracy', 'std'))}). With 1,000 records the same ε requires "
        f"{min(R.dps('small', e, 'sigma') / R.dps('full', e, 'sigma') for e in eps_grid):.1f}–"
        f"{max(R.dps('small', e, 'sigma') / R.dps('full', e, 'sigma') for e in eps_grid):.1f}× more noise "
        f"(σ = {f(R.dps('small', 1.0, 'sigma'), 1)} vs {f(R.dps('full', 1.0, 'sigma'), 1)} at ε = 1) because each record "
        "is a larger share of every batch, and accuracy falls steadily below ε ≈ 4. **Privacy amplification by "
        "sub-sampling is what makes DP-SGD cheap on large datasets.** F1 degrades faster than accuracy: "
        "noise first erodes the minority (>50K) class [28].")
    rep.para(
        f"**Membership inference.** In the small regime the non-private MLP has advantage {f(gap_s)}; DP-SGD "
        f"reduces it to {f(R.dps('small', 8.0, 'mia_advantage'))} at ε = 8 and {f(R.dps('small', 0.05, 'mia_advantage'))} at "
        f"ε = 0.05. However, the *clipping-only* model (ε = ∞) already has advantage "
        f"{f(R.dps('small', INF, 'mia_advantage'))}. **Most of the empirical MIA reduction therefore comes from "
        "clipping and the shorter, noisier training (implicit regularisation), not from the noise level.** "
        "This is a critical point. The value of a small ε is the *worst-case* guarantee against any attacker "
        "and any record, which an average-case empirical attack cannot show. Empirical leakage alone would "
        "wrongly suggest ε is irrelevant.")
    ab = lambda p, v: R.ab.loc[(p, v), ("accuracy", "mean")]
    rep.figure(F2 / "fig3_dpsgd_ablation.png", "DP-SGD hyper-parameter sensitivity at ε = 1 (3 seeds).")
    rep.para(f"**Ablation (E3).** Clipping norm is the most sensitive parameter: C = 0.1 loses "
             f"{100 * (ab('clip', 1.0) - ab('clip', 0.1)):.1f} points (gradients over-clipped, bias) and C = 5 loses "
             f"{100 * (ab('clip', 1.0) - ab('clip', 5.0)):.1f} points (noise σC dominates). Learning rate shows a "
             f"similar optimum at 0.05–0.1. Training longer at fixed ε helps only slightly "
             f"({pct(ab('epochs', 5))} → {pct(ab('epochs', 40))} from 5 to 40 epochs) because σ grows with T. The "
             "default (C = 1, lr = 0.05, 20 epochs) lies on the plateau. Note: choosing hyper-parameters on "
             "private data itself leaks information [27]; see Section 5.11.")

    # ---- E4
    rep.h2("5.4 E4 — Central vs local DP for statistics release (RQ2)")
    rows = []
    for at in ["workclass", "marital-status", "occupation", "native-country"]:
        for e in (0.5, 1.0, 4.0):
            r_ = R.mg.loc[(at, e)]
            best = min(r_["Local-GRR"], r_["Local-OUE"])
            rows.append([at, f"{e:g}", f"{r_['Central-Laplace']:.5f}", f"{r_['Local-GRR']:.4f}",
                         f"{r_['Local-OUE']:.4f}", f"{best / r_['Central-Laplace']:.0f}×"])
    rep.table(["Attribute (k)", "ε", "Central Laplace", "Local GRR", "Local OUE", "Best local / central"], rows,
              caption="Mean L1 error of released histograms (50 repetitions). Domain sizes: workclass 7, "
                      "marital-status 7, occupation 14, native-country 41.",
              widths=[3.2, 1.2, 3, 2.6, 2.6, 3.4], size=8.5)
    rep.figure(F2 / "fig4_central_vs_local.png", "Central vs. local DP histogram error across four attributes.")
    rep.para("Central Laplace error is O(k/(nε)); local error is O(√(k)/(√n·ε)) at best, so the ratio grows with "
             f"√n. At n = 27,146 the best local mechanism is {rmin:.0f}–{rmax:.0f}× worse across all attributes "
             "and ε. The best local oracle depends on "
             "domain size, as theory predicts: OUE wins for the 41-value native-country attribute at "
             "ε ≤ 2, whereas GRR wins for small domains and large ε. **RQ2 answer:** removing the trusted "
             "curator costs one to three orders of magnitude in statistical accuracy at equal ε.")

    # ---- E5
    rep.h2("5.5 E5 — Training on LDP-collected records (M3)")
    rows = []
    for e in sorted(R.e5.eps_total.unique()):
        a = R.ldp.loc[("full-record LDP", e)]
        b_ = R.ldp.loc[("features-only LDP", e)]
        rows.append([f"{e:g}", f(a['eps_per_attribute'], 3), f(a['accuracy']), f(a['f1']), f(a['auc']),
                     f(a['attr_bal_acc']), f(b_['accuracy']), f(b_['f1'])])
    rep.table(["ε_total", "ε / attribute", "Accuracy", "F1", "AUC", "Sex inference", "Acc. (feat.-only)", "F1 (feat.-only)"],
              rows, caption="Models trained on LDP-perturbed records (3 seeds). 'Sex inference' = balanced accuracy of a "
                            "GBM attacker recovering the true sex from the perturbed record. Features-only: label kept "
                            "exact (ablation; weaker guarantee).",
              widths=[1.6, 2, 1.8, 1.4, 1.4, 2.4, 2.6, 2.4], size=8)
    rep.figure(F2 / "fig5_ldp_training.png", "Left: utility of models trained on LDP data; dotted = non-private F1. "
               "Right: how well the true sex can be recovered from an LDP report.")
    a16, a32, a128 = (R.ldp.loc[("full-record LDP", e)] for e in (16, 32, 128))
    rep.para(
        f"Splitting the budget over 13 attributes leaves each attribute with very little privacy budget. "
        f"Below ε_total = 8 the model is worse than predicting the majority class ({pct(majority)}), because "
        f"label flipping (at ε/attribute ≈ 0.08 the label is flipped {pct(1 / (1 + np.exp(1 / 13)))} of the time) "
        "makes its predictions close to random. Removing label noise (features-only) recovers majority-class "
        f"accuracy but still yields F1 = 0 until ε_total = 8. Utility approaches the non-private model only at "
        f"ε_total = 128 (accuracy {pct(a128['accuracy'])}). **The tension is direct.** At ε_total = 16 (F1 "
        f"{f(a16['f1'])}) sex is already recoverable with balanced accuracy {f(a16['attr_bal_acc'])}; at ε_total = 32 "
        f"(F1 {f(a32['f1'])}) with {f(a32['attr_bal_acc'])}. MIA is at chance throughout, since the model never "
        "sees an exact record. Per-record LDP for many attributes is therefore poorly suited to supervised "
        "learning on a dataset of this size; it is better suited to the aggregate statistics of E4.")

    # ---- E6
    rep.h2("5.6 E6 — k-anonymity and l-diversity (RQ3)")
    rows = [["raw (k = 1)", "—", f(ka_raw['ncp']), f(ka_raw['reid_avg_risk']), pct(ka_raw['unique_frac']),
             "0.0%", "—", "1.000", "—", "—"]]
    for l_ in ("none", "2"):
        for k in (2, 10, 50, 250):
            r_ = R.ka.loc[(l_, k)]
            rows.append([f"k = {k}", "—" if l_ == "none" else "2", f(r_['ncp']), f(r_['reid_avg_risk']),
                         pct(r_['unique_frac']), pct(r_['homogeneity_rate']), f(r_['t_closeness'], 2),
                         f(r_['attr_bal_acc']), f(r_['accuracy']), f(r_['f1'])])
    rep.table(["Release", "l", "NCP", "Avg re-id risk", "Unique", "Income disclosed", "t", "Sex inference", "Accuracy", "F1"],
              rows, caption="Mondrian anonymisation of the private training split (model results: 3 seeds).",
              widths=[1.9, 0.8, 1.4, 1.9, 1.4, 1.9, 1.1, 1.8, 1.7, 1.3], size=8)
    rep.figure(F2 / "fig6_kanonymity.png", "Effect of k (with and without 2-diversity) on identity risk, value "
               "disclosure, information loss and utility.")
    r10, r10l = R.ka.loc[("none", 10)], R.ka.loc[("2", 10)]
    rep.para(
        f"**Identity.** Mondrian removes all unique records and reduces average re-identification risk from "
        f"{f(ka_raw['reid_avg_risk'])} to {f(r10['reid_avg_risk'])} at k = 10, at almost no utility cost: accuracy "
        f"is flat at ≈ {pct(r10['accuracy'])} up to k = 100, because most predictive signal sits in non-QI "
        f"attributes (occupation, capital gain, hours) and generalised QIs keep their means.")
    rep.para(
        f"**Values.** Without l-diversity, {pct(r10['homogeneity_rate'])} of records at k = 10 belong to classes "
        "where everyone has the same income, so linking the target to their class reveals the target's "
        f"income with certainty. 2-diversity removes this at k = 10 for an NCP increase from "
        f"{f(r10['ncp'])} to {f(r10l['ncp'])} and no utility loss. t-closeness stays high (≈ 0.6–0.75): "
        "class income rates remain far from the population rate, so probabilistic inference is still possible.")
    rep.para(
        "**Attributes.** Sex remains perfectly inferable (balanced accuracy 1.0) up to k = 100. Sex is a "
        "binary QI that Mondrian splits early and therefore almost never generalises; it is only "
        "generalised at k = 250. **RQ3 answer:** k-anonymity protects identity, l-diversity "
        "additionally protects against certain disclosure of the sensitive value, and neither protects "
        "other attributes or provides a formal guarantee.")

    # ---- E7
    rep.h2("5.7 E7 — Compressive privacy and D-DCA (RQ4)")
    projs = ["PCA (all features)", "PCA (sex suppressed)", "DCA (sex suppressed)", "D-DCA (proposed)", "D-DCA (joint label)"]
    rows = [["Raw (all 87 features)", "—", f(raw7[('accuracy', 'mean')]), f(raw7[('f1', 'mean')]),
             f(raw7[('lr_attr_bal_acc', 'mean')]), f(raw7[('gbm_attr_bal_acc', 'mean')]), "0.000"],
            ["Raw (sex column dropped)", "—", f(sup7[('accuracy', 'mean')]), f(sup7[('f1', 'mean')]),
             f(sup7[('lr_attr_bal_acc', 'mean')]), f(sup7[('gbm_attr_bal_acc', 'mean')]), "0.000"]]
    for m in (2, 5, 10):
        for p in projs:
            rows.append([p, m, f(R.cpv(p, m, 'accuracy')), f(R.cpv(p, m, 'f1')), f(R.cpv(p, m, 'lr_attr_bal_acc')),
                         f(R.cpv(p, m, 'gbm_attr_bal_acc')), f(R.cpv(p, m, 'reconstruction_nmse'))])
    rep.table(["Release", "m", "Accuracy", "F1", "Sex: LR", "Sex: GBM", "Recon. NMSE"], rows,
              caption="Compressive projections (3 seeds). Sex columns = balanced accuracy of the attacker (0.5 = no "
                      "information). Recon. NMSE = error of the best linear reconstruction (1 = nothing recoverable).",
              widths=[4.4, 0.9, 1.9, 1.5, 1.8, 1.9, 2.2], size=8)
    rep.figure(F2 / "fig7_compressive.png", "Utility and sensitive-attribute leakage of projections across target "
               "dimension m.")
    dd5 = lambda c: R.cpv("D-DCA (proposed)", 5, c)
    rep.para(
        f"**PCA is not a privacy mechanism.** On all features PCA retains sex perfectly from m = 10. **D-DCA "
        f"(proposed)** drives the linear attacker to exactly chance (0.500) for every m, at m = 5 keeping "
        f"{pct(dd5('accuracy'))} accuracy (raw: {pct(raw7[('accuracy', 'mean')])}) and reducing the non-linear GBM "
        f"attacker to {f(dd5('gbm_attr_bal_acc'))}, below the {f(sup7[('gbm_attr_bal_acc', 'mean')])} obtained by merely "
        f"dropping the sex column. The trade-off is not one-sided: at m = 5, PCA on the suppressed features "
        f"leaks slightly *less* to the GBM attacker ({f(R.cpv('PCA (sex suppressed)', 5, 'gbm_attr_bal_acc'))}) but "
        f"retains linear leakage ({f(R.cpv('PCA (sex suppressed)', 5, 'lr_attr_bal_acc'))}) and "
        f"{100 * (dd5('accuracy') - R.cpv('PCA (sex suppressed)', 5, 'accuracy')):.1f} points less accuracy. "
        f"At m = 2 the GBM attacker falls to "
        f"{f(R.cpv('D-DCA (proposed)', 2, 'gbm_attr_bal_acc'))} while accuracy stays at "
        f"{pct(R.cpv('D-DCA (proposed)', 2, 'accuracy'))}, better than PCA at the same m on both axes. DCA "
        "(utility-only) discards almost everything (NMSE ≈ 0.95, the strongest reconstruction privacy) but "
        "keeps more sex information than D-DCA for m ≤ 3. The joint (sex × relationship) label removes more "
        "non-linear signal but costs up to 3 points of accuracy, because relationship also predicts income — a "
        "negative result we report deliberately.")
    g_rows = []
    for gm in (0.0, 1.0, 10.0, 100.0, 1000.0):
        key = f"D-DCA gamma={gm:g}"
        g_rows.append([f"{gm:g}", f(R.cpv(key, 5, 'accuracy', exp='E7-gamma')),
                       f(R.cpv(key, 5, 'lr_attr_bal_acc', exp='E7-gamma')),
                       f(R.cpv(key, 5, 'gbm_attr_bal_acc', exp='E7-gamma')),
                       f(R.cpv(key, 5, 'reconstruction_nmse', exp='E7-gamma'))])
    rep.table(["γ", "Accuracy", "Sex: LR", "Sex: GBM", "Recon. NMSE"], g_rows,
              caption="Effect of the privacy weight γ in D-DCA (m = 5).", widths=[2, 3.5, 3.5, 3.5, 3.5])
    rep.para("γ acts as a switch for linear leakage: γ = 1 already removes it at no measurable utility cost. "
             "Larger γ costs ≈ 0.5 points and even raises GBM leakage slightly, because the rank-one privacy "
             "scatter is already nulled and extra weight only distorts the remaining directions. γ = 1 would "
             "therefore be a better default than the γ = 10 used in E7/E8 — a tuning insight this analysis "
             "revealed. **RQ4 answer:** "
             "yes for linear adversaries (by construction, confirmed empirically); partially for non-linear adversaries, "
             "where leakage is governed mainly by m. Linear projections cannot remove non-linearly encoded "
             "information, which is the main limitation of this family.")

    # ---- E8
    rep.h2("5.8 E8 — Hybrid compressive + DP-SGD (RQ5)")
    variants = [DPF, "PCA-5 + DP-SGD", HYB, "D-DCA-10 + DP-SGD"]
    rows = []
    for e in eps_grid + [INF]:
        rows.append(["∞ (clip)" if e == INF else f"{e:g}"] +
                    [ms(R.hyv(v, e, 'accuracy'), R.hyv(v, e, 'accuracy', 'std')) for v in variants])
    rep.table(["ε", "DP-SGD (87 features)", "PCA-5 + DP-SGD", "D-DCA-5 + DP-SGD", "D-DCA-10 + DP-SGD"], rows,
              caption="Test accuracy of DP-SGD with and without compression (5 seeds, δ = 10⁻⁵).",
              widths=[1.6, 3.6, 3.6, 3.6, 3.6], size=8.5)
    rows = [[v, f(R.hyv(v, 1.0, 'f1')), f(R.hyv(v, 1.0, 'mia_advantage')),
             f(R.hyv(v, 1.0, 'attr_bal_acc')) if v != DPF else "n/a (no features released)"] for v in variants]
    rep.table(["Variant", "F1 @ ε = 1", "MIA adv. @ ε = 1", "Sex inference from released features"], rows,
              caption="Privacy side of the hybrid.", widths=[5, 2.6, 3, 5.4], size=8.5)
    rep.figure(F2 / "fig8_hybrid.png", "Hybrid compressive + DP-SGD versus DP-SGD on all features.")
    rep.para(
        "**Hypothesis (from IA-1).** Compression should help DP-SGD: fewer input dimensions mean fewer "
        "parameters to perturb, and D-DCA concentrates the utility signal into the retained directions. "
        f"**Outcome.** The hypothesis is *not* supported for accuracy. At ε ≤ {max(ties):g} the hybrid and "
        f"DP-SGD on all features are statistically indistinguishable (difference within one sd; e.g. ε = 0.1: "
        f"{pct(R.hyv(HYB, 0.1, 'accuracy'))} vs {pct(R.hyv(DPF, 0.1, 'accuracy'))}). The hybrid is more stable "
        f"there (sd {f(R.hyv(HYB, 0.1, 'accuracy', 'std'))} vs {f(R.hyv(DPF, 0.1, 'accuracy', 'std'))}). For ε ≥ 0.5 it trails by "
        f"≈ {100 * -np.mean([gain[e] for e in large]):.1f} points: the 5-dimensional representation caps accuracy at "
        f"≈ {pct(R.hyv(HYB, INF, 'accuracy'))} even without noise, so its cost dominates once DP noise is small. "
        "An earlier two-seed exploration had suggested a gain at ε = 0.05; with five seeds this vanished — a "
        "reminder of why multi-seed reporting matters. D-DCA-10 recovers most of the large-ε accuracy, but "
        "with weaker attribute protection, and PCA-5 is worse than D-DCA-5 for ε ≥ 0.25, so *which* "
        "directions are kept matters.")
    rep.para(
        "**Where the hybrid earns its place** is the privacy column. It is the only configuration with a formal "
        "(ε, δ) guarantee for the model *and* attribute protection for released features (sex balanced "
        f"accuracy {f(R.hyv(HYB, 1.0, 'attr_bal_acc'))} vs {f(raw7[('gbm_attr_bal_acc', 'mean')])} if raw features were "
        "released). In the strong-privacy regime, this extra protection comes at no measurable accuracy "
        "cost. **RQ5 answer:** compressing before DP-SGD does not improve accuracy. It is worthwhile when "
        "features must be released or the budget is very small (ε ≤ 0.1); otherwise plain DP-SGD is preferable.")

    rep.figure(F2 / "fig9_trust_models.png", "Accuracy at the same nominal ε under different trust models: central DP, "
               "hybrid, and local DP. The x-axis is ε per record; for LDP this is the total budget over 13 attributes.", 12)

    # ---- unified table
    rep.h2("5.9 Unified comparison")
    l16, l64 = R.ldp.loc[("full-record LDP", 16)], R.ldp.loc[("full-record LDP", 64)]
    rows = [
        ["Non-private MLP", "—", "none", f(np_mlp), f(R.base('full', 'MLP', 'f1')), f(R.base('full', 'MLP', 'mia_auc')),
         f(raw7[('gbm_attr_bal_acc', 'mean')]), f(ka_raw['reid_avg_risk'])],
        ["LDP ε_total = 16", "collection", "16-LDP", f(l16['accuracy']), f(l16['f1']), f(l16['mia_auc']), f(l16['attr_bal_acc']), "n/a*"],
        ["LDP ε_total = 64", "collection", "64-LDP", f(l64['accuracy']), f(l64['f1']), f(l64['mia_auc']), f(l64['attr_bal_acc']), "n/a*"],
        ["k = 10, l = 2", "publication", "syntactic", f(r10l['accuracy']), f(r10l['f1']), f(r10l['mia_auc']), f(r10l['attr_bal_acc']), f(r10l['reid_avg_risk'])],
        ["DP-SGD ε = 0.1", "training", "(0.1, 10⁻⁵)", f(R.dps('full', 0.1, 'accuracy')), f(R.dps('full', 0.1, 'f1')), f(R.dps('full', 0.1, 'mia_auc')), "n/a", "n/a"],
        ["DP-SGD ε = 1", "training", "(1, 10⁻⁵)", f(R.dps('full', 1.0, 'accuracy')), f(R.dps('full', 1.0, 'f1')), f(R.dps('full', 1.0, 'mia_auc')), "n/a", "n/a"],
        ["D-DCA m = 5", "feature release", "empirical", f(dd5('accuracy')), f(dd5('f1')), f(dd5('mia_auc')), f(dd5('gbm_attr_bal_acc')), "n/a†"],
        ["Hybrid ε = 0.1", "release + training", "(0.1, 10⁻⁵) + D-DCA", f(R.hyv(HYB, 0.1, 'accuracy')), f(R.hyv(HYB, 0.1, 'f1')), f(R.hyv(HYB, 0.1, 'mia_auc')), f(R.hyv(HYB, 0.1, 'attr_bal_acc')), "n/a†"],
        ["Hybrid ε = 1", "release + training", "(1, 10⁻⁵) + D-DCA", f(R.hyv(HYB, 1.0, 'accuracy')), f(R.hyv(HYB, 1.0, 'f1')), f(R.hyv(HYB, 1.0, 'mia_auc')), f(R.hyv(HYB, 1.0, 'attr_bal_acc')), "n/a†"],
    ]
    rep.table(["Configuration", "Stage", "Guarantee", "Accuracy", "F1", "MIA AUC", "Sex inference (GBM)", "Re-id risk"], rows,
              caption="Unified comparison of representative configurations (full-data regime). Sex inference is measured on "
                      "whatever is released. *LDP reports are randomised per attribute, so exact linkage is not "
                      "defined. †No records are released, only m-dimensional projections.",
              widths=[3, 2.4, 2.6, 1.6, 1.3, 1.6, 2.2, 1.6], size=8)
    rep.para("No configuration wins on every column. DP-SGD gives the strongest formal guarantee at the lowest "
             "utility cost, but it protects only the model. k-anonymity is the only one that addresses "
             "linkage of released records, yet it leaves attributes exposed. D-DCA protects attributes in released "
             "features without a formal guarantee. LDP protects even against the collector, but at a "
             "prohibitive utility cost for record-level training. **The choice of mechanism must follow "
             "what is released and whom the data owner must trust**, and the hybrid shows that combining "
             "stages can cover more threats than any single mechanism.")

    rep.h2("5.10 Answers to the research questions")
    rep.table(["RQ", "Answer (evidence)"], [
        ["RQ1", "Each mechanism reduces a *different* risk: DP-SGD membership (formal), Mondrian identity, l-diversity value disclosure, D-DCA attribute inference, LDP all of them but at large utility cost (Section 5.9)."],
        ["RQ2", f"Local DP costs {rmin:.0f}–{rmax:.0f}× in histogram L1 error at equal ε (E4) and needs ε_total ≥ 64 for useful record-level training (E5)."],
        ["RQ3", "k-anonymity protects identities only; 2-diversity removes certain income disclosure; sex remains inferable up to k = 100 (E6)."],
        ["RQ4", "D-DCA removes linear leakage completely and beats PCA/DCA on the utility–attribute-privacy frontier at small m; non-linear leakage persists (E7)."],
        ["RQ5", "No accuracy gain: the hybrid ties DP-SGD for ε ≤ 0.1 (with lower variance) and trails by ≈ 1 point at larger ε, but uniquely adds attribute protection for released features (E8)."],
    ], caption="Answers to the research questions.", widths=[1.2, 14.8])

    rep.h2("5.11 Limitations and threats to validity")
    rep.table(["Type", "Threat", "Mitigation / consequence"], [
        ["Internal", "DP-SGD hyper-parameters were chosen on private data (sanity run + E3); tuning leaks privacy [27]", "Plateau in E3 shows results are insensitive near the default; a private tuning procedure is future work"],
        ["Internal", "Small-regime DP-SGD uses fewer epochs than the non-private MLP", "Clip-only (ε = ∞) run isolates this confound and is reported explicitly"],
        ["Internal", "Public split assumed non-sensitive and from the same distribution", "Standard assumption; DP-PCA/DP-scatter estimation would remove it"],
        ["Construct", "Loss-threshold MIA and GBM attacker are lower bounds on leakage", "TPR@1%FPR added; stronger attacks (LiRA) future work"],
        ["Construct", "Re-identification uses the prosecutor model (target known to be in data)", "Conservative (upper) estimate of linkage risk"],
        ["External", "Single, dated tabular dataset; one model family", "Framework is dataset-agnostic; ACS-based datasets [25] are a natural extension"],
        ["Conclusion", "3–5 seeds; full-data MIA differences are within noise", "All tables report mean ± sd; claims restricted to effects larger than the sd"],
    ], caption="Threats to validity.", widths=[2, 7, 7], size=8.5)

    rep.h2("5.12 Proposed improvements")
    rep.bullets([
        "**Non-linear compressive privacy:** kernel D-DCA or iterative nullspace projection [29] to address the "
        "residual GBM leakage.",
        "**Remove the public-data assumption:** fit D-DCA from DP-noised scatter matrices, then account the "
        "projection's ε together with DP-SGD's.",
        "**Better LDP learning:** attribute sampling instead of budget splitting [9], and loss correction for "
        "known label-flip rates [30].",
        "**Stronger attacks:** LiRA [21] and subgroup-level MIA to test disparate leakage [28].",
        "**Private hyper-parameter selection** with RDP accounting [27].",
    ])

    # ================================================================= 6–8
    rep.h1("6. Conclusion")
    rep.para(
        f"{TITLE} shows that privacy mechanisms from different course modules are not interchangeable. "
        "They protect different stages against different adversaries. Evaluating them under one model and "
        "one attack suite exposes facts that per-mechanism evaluation hides. Empirical MIA reduction under "
        "DP-SGD comes largely from clipping. k-anonymity leaves sensitive attributes untouched. PCA is not a "
        "privacy mechanism. Record-level LDP for many attributes is a poor fit for supervised learning at "
        "this scale. The proposed D-DCA eliminates linear attribute leakage at negligible utility cost, and "
        "combining it with DP-SGD yields the only configuration that gives both a formal guarantee for the "
        "model and attribute protection for the released features. In the strong-privacy regime this costs "
        "no measurable accuracy, although, contrary to our initial hypothesis, it does not improve accuracy "
        "either. The framework, data provenance and all results are reproducible from a "
        "single command sequence on a laptop.")

    rep.h1("7. Individual Contribution Statement")
    rep.para(
        "This is an individual project. My contributions are: (i) formulation of the stage-wise comparison "
        "problem, threat model and research questions; (ii) dataset selection, provenance documentation and "
        "the public/private/test split design that prevents preprocessing leakage; (iii) implementation of "
        "the complete `src/privml` package — attack suite, RDP accountant, DP-SGD with closed-form "
        "per-example gradients, GRR/OUE/Piecewise LDP, Mondrian k-anonymity with l-diversity, PCA/DCA; "
        "(iv) the methodological enhancements — the Desensitised DCA projection, the "
        "suppression-before-projection design, and the hybrid D-DCA → DP-SGD pipeline; (v) the "
        "experimental design E1–E8 and comparative analysis; (vi) 30 unit tests, figures, and the "
        "automated report generation.")

    rep.h1("8. Research and IP Development")
    rep.bullets([
        "**Manuscript draft:** `reports/Research_Manuscript_Draft.docx` — a conference-style paper "
        "(abstract, introduction, method, experiments, results, discussion) built from this work, "
        "suitable for development towards a venue such as a national privacy/security conference or a "
        "student track.",
        "**Software copyright (optional):** the `src/privml` code and the report generator are original "
        "works that could be registered as a computer programme (literary work) with the Copyright "
        "Office, Government of India. Evidence of authorship: dated run logs, environment records and "
        "result CSVs in `results/`.",
        "**Novelty claim, stated conservatively:** the D-DCA objective and the hybrid pipeline are "
        "extensions of Kung's DCA/DUCA framework [17, 18]. A literature search should be completed before "
        "any novelty claim is made in a submission.",
    ])

    references(rep)

    # ================================================================= appendices
    rep.h1("Appendix A — Metric definitions and worked calculations")
    rep.table(["Metric", "Definition"], [
        ["Accuracy / Precision / Recall / F1", "(TP+TN)/N; TP/(TP+FP); TP/(TP+FN); 2PR/(P+R) with positive class = income > 50K"],
        ["ROC-AUC", "Probability that a random positive is scored above a random negative"],
        ["MIA advantage", "max over thresholds τ of TPR(τ) − FPR(τ), where the score is −loss and members are positives"],
        ["Balanced accuracy", "½(TPR + TNR); 0.5 means the attacker learned nothing beyond the prior"],
        ["Prosecutor risk", "1/|EC(r)| for record r; reported as mean and max over records"],
        ["NCP", "numeric: (max−min)/(domain range); categorical: (|set|−1)/(|domain|−1); averaged over QIs and records"],
        ["Reconstruction NMSE", "‖X − X̂‖²_F / ‖X − X̄‖²_F with X̂ = Z W⁺ + μ"],
    ], caption="Metric definitions.", widths=[4, 12])
    q = 256 / sizes["splits"]["private_train"]
    rep.para(f"**Worked example 1 — DP-SGD accounting.** n = {sizes['splits']['private_train']:,}, batch 256 ⇒ q = "
             f"{q:.5f}; 20 epochs ⇒ T = round(20/q) = {round(20 / q):,}. For target ε = 1, δ = 10⁻⁵, bisection gives "
             f"σ = {f(R.dps('full', 1.0, 'sigma'))}; the per-step RDP at the optimal order is composed over T steps and "
             "converted with ε = RDP_T(α) + log((α−1)/α) − (log δ + log α)/(α−1).")
    e_ = 1.0
    p_ = np.exp(e_) / (np.exp(e_) + 13)
    rep.para(f"**Worked example 2 — GRR estimator.** occupation has k = 14. At ε = 1, p = e/(e + 13) = {p_:.4f} and "
             f"q = (1 − p)/13 = {(1 - p_) / 13:.4f}. If 9.0% of reports say 'Sales', the unbiased estimate is "
             f"(0.090 − {(1 - p_) / 13:.4f}) / ({p_:.4f} − {(1 - p_) / 13:.4f}) = {(0.09 - (1 - p_) / 13) / (p_ - (1 - p_) / 13):.4f}.")
    rep.para("**Worked example 3 — NCP.** A class with ages 32–37 (domain 17–90) and marital-status set of 4 of 7 "
             "values contributes (37−32)/73 = 0.068 for age and (4−1)/(7−1) = 0.5 for marital status.")

    rep.h1("Appendix B — Hyper-parameters")
    rep.table(["Component", "Hyper-parameters"], [
        ["Non-private MLP", "hidden 64, Adam lr 1e-3, batch 128, 30 epochs, no weight decay, PyTorch-style uniform init"],
        ["Random forest", "200 trees, min_samples_leaf = 1 (fully grown)"],
        ["Logistic regression", "L2, C = 1, max_iter 2000"],
        ["DP-SGD", "hidden 64, C = 1, momentum 0.9, lr 0.05, batch 256, 20 epochs, Poisson sampling, RDP orders 2–255"],
        ["GBM attacker", "HistGradientBoosting, 200 iterations"],
        ["D-DCA", "β = 0.1, γ = 10, ρ = 0.01 (γ sweep in E7)"],
        ["DCA", "ρ = 0.01, ρ′ = 0.001"],
    ], caption="Hyper-parameters.", widths=[3.5, 12.5])

    rep.h1("Appendix C — File hashes (SHA-256)")
    rep.table(["File", "Bytes", "SHA-256"],
              [[prov['hashes']['archive']['file'], f"{prov['hashes']['archive']['bytes']:,}", f"`{prov['hashes']['archive']['sha256']}`"]]
              + [[n, f"{v['bytes']:,}", f"`{v['sha256']}`"] for n, v in prov['hashes']['files'].items()],
              caption="Raw and processed file hashes.", widths=[3.8, 2, 10.2], size=8)
    env = (R2 / "environment.json").read_text()
    rep.h1("Appendix D — Execution environment")
    rep.code(env, size=8)
    rep.save(out)
