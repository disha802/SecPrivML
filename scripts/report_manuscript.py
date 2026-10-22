"""Research manuscript draft (conference-paper structure) built from the IA-2 results."""

from __future__ import annotations

import numpy as np

from report_common import LONG_TITLE, REFERENCES, provenance
from report_ia2 import DPF, F2, HYB, INF, Results, f, pct
from report_lib import Report

# References used by the manuscript, in citation order (indices into REFERENCES, 1-based)
REFS = [11, 22, 19, 20, 21, 1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 17, 18, 29, 28, 23, 25, 27]
N = {orig: i + 1 for i, orig in enumerate(REFS)}


def c(*orig):
    return "[" + ", ".join(str(N[o]) for o in orig) + "]"


def build(out):
    R = Results()
    prov = provenance()
    n_total = prov["sizes"]["final"]["total_rows"]
    np_acc = R.base("full", "MLP", "accuracy")
    raw7 = R.cp.loc[("E7", "Raw (all features)", 87)]
    sup7 = R.cp.loc[("E7", "Raw (sex suppressed)", 85)]
    dd = lambda m, col: R.cpv("D-DCA (proposed)", m, col)
    r10, r10l, raw_k = R.ka.loc[("none", 10)], R.ka.loc[("2", 10)], R.ka.loc[("none", 1)]
    l32 = R.ldp.loc[("full-record LDP", 32)]
    ratios = [min(R.mg.loc[k_, "Local-GRR"], R.mg.loc[k_, "Local-OUE"]) / R.mg.loc[k_, "Central-Laplace"]
              for k_ in R.mg.index]

    rep = Report("Manuscript draft — not for distribution")
    d = rep.doc
    p = rep.para(LONG_TITLE, align="center", size=17)
    for r in p.runs:
        r.bold = True
    rep.para("Disha Kataria¹, Ashwini Dalvi¹", align="center", size=11)
    rep.para("¹ Department of Information Technology, K. J. Somaiya School of Engineering, Mumbai, India — disha.kataria@somaiya.edu", align="center", size=9.5, italic=True)

    rep.h2("Abstract")
    rep.para(
        "Privacy mechanisms for machine learning act at different pipeline stages — local differential "
        "privacy (LDP) at collection, k-anonymity at publication, DP-SGD at training and compressive "
        "projections at feature release — and are usually evaluated in isolation with incomparable "
        "metrics. We present PrivBench-Adult, a reproducible framework that subjects one mechanism per "
        "stage to the same model and the same attack suite (membership inference, sensitive-attribute "
        f"inference and re-identification) on the UCI Adult census data ({n_total:,} records). We also "
        "propose Desensitised Discriminant Component Analysis (D-DCA), a privacy-penalised generalised-"
        "eigenvector projection, and a hybrid D-DCA → DP-SGD pipeline. Across 3–5 seeds we find: DP-SGD "
        f"stays within 0.2 points of non-private accuracy down to ε = 0.5, but its empirical "
        "membership-inference reduction stems largely from gradient clipping; Mondrian k-anonymity "
        f"removes unique records at no utility cost but leaves income disclosed for {pct(r10['homogeneity_rate'])} "
        "of records at k = 10 without l-diversity, and leaves sex fully inferable; record-level LDP for 13 "
        "attributes needs ε_total ≥ 64 for useful utility, where sensitive attributes become inferable again; "
        f"D-DCA reduces a linear attacker to chance at {pct(dd(5, 'accuracy'))} accuracy. The hybrid does not "
        "improve accuracy over DP-SGD, but it uniquely combines a formal guarantee for the model with "
        "attribute protection for released features, at no measurable cost for ε ≤ 0.1.")
    rep.para("**Index Terms** — differential privacy, local differential privacy, k-anonymity, compressive "
             "privacy, membership inference, attribute inference, privacy–utility trade-off.", align="left")

    rep.h2("I. Introduction")
    rep.para(
        f"Removing identifiers does not anonymise data {c(11, 22)}, and trained models leak membership of "
        f"their training records {c(19, 20, 21)}. Four families of defences address these risks at different "
        f"points of the ML pipeline: central differential privacy (DP) {c(1, 2)}, local DP {c(7, 8)}, syntactic "
        f"anonymisation {c(11, 12, 13)} and compressive privacy {c(17, 18)}. Each comes with its own guarantee "
        "and metric (ε, k, reconstruction error), so practitioners cannot directly compare them, nor see which "
        "threats each one leaves open.")
    rep.para("**Contributions.** (1) A unified, open, laptop-scale evaluation framework in which mechanisms from "
             "four families are applied at their natural stage and evaluated with one model and one "
             "Kerckhoffs-style attack suite. (2) From-scratch, unit-tested implementations, including a Rényi-DP "
             "accountant that matches Opacus and closed-form per-example gradients for DP-SGD. (3) D-DCA, a "
             "privacy-penalised extension of DCA, and a suppression-before-projection design that defeats proxy "
             "leakage. (4) An empirical study of eight experiments that exposes non-obvious gaps between formal "
             "guarantees and empirical risk, including a negative result for the hybrid's accuracy.")

    rep.h2("II. Threat Model and Background")
    rep.para(
        "The adversary knows the mechanism and its parameters and holds an auxiliary sample from the same "
        "distribution (modelled by a 10% public split). It pursues three goals. **Membership inference** uses "
        f"the loss-threshold attack {c(20)}, reported as AUC, advantage max(TPR − FPR) and TPR at 1% FPR "
        f"{c(21)}. **Attribute inference** of *sex* uses logistic-regression and gradient-boosting attackers "
        "trained on the public split processed by the same mechanism. **Re-identification** is the "
        f"prosecutor risk 1/|EC| over seven quasi-identifiers. DP-SGD {c(2)} clips per-example gradients to "
        f"norm C and adds N(0, σ²C²) noise; privacy is accounted with RDP of the sub-sampled Gaussian "
        f"{c(3, 4)} and converted to (ε, δ) {c(5)}. GRR and OUE {c(7, 8)} and the Piecewise mechanism {c(9)} "
        f"provide LDP. Mondrian {c(12)} yields multidimensional k-anonymity, extended here with distinct "
        f"l-diversity {c(13)} and measured t-closeness {c(14)}.")

    rep.h2("III. Method")
    rep.figure(F2 / "fig1_architecture.png", "PrivBench-Adult architecture.", 16)
    rep.para(
        "**Pipeline.** The official Adult training file is split (seed 0) into a public split (3,016) and a "
        "private training split (27,146); the official test set (15,060) is held out. All preprocessing "
        "statistics and projections are fitted on the public split only, so no private statistic leaks "
        "outside the mechanisms' accounting. Every protected output trains the same MLP (87→64→1).")
    rep.para(
        "**D-DCA.** Let S_B^u and S_B^p be the between-class scatters of the utility label (income) and the "
        "privacy label (sex), and Σ the covariance. D-DCA solves the generalised eigenproblem")
    rep.equation("(S_B^u + β Σ) w = λ (γ S_B^p + Σ + ρ I) w")
    rep.para(
        f"and keeps the top-m eigenvectors. The β-term yields m > 1 informative directions for a binary label; "
        f"the γ-term penalises sex-discriminant directions. γ = 0 recovers a variance-regularised DCA {c(17)}. "
        "We apply D-DCA *after suppressing* the explicit sex columns. A linear projection of a binary one-hot "
        "column leaves a bimodal component that a non-linear attacker can decode, so suppression removes the "
        "attribute and D-DCA removes its linear proxies. **Hybrid.** DP-SGD is trained on the standardised "
        "D-DCA features; since the projection is fitted on public data, the model's (ε, δ) is unchanged.")

    rep.h2("IV. Experimental Setup")
    rep.para(
        "DP-SGD uses batch 256 (q = 0.0094), 20 epochs, C = 1, momentum-SGD with learning rate 0.05 and "
        "δ = 10⁻⁵ over ε ∈ {0.05, …, 8}; a small-data regime (n = 1,000) probes over-fitting. LDP perturbs all "
        "13 attributes with ε_total split equally. Mondrian uses k ∈ {2, …, 250}, with and without "
        "2-diversity. Projections use m ∈ {2, …, 40}. Results are means over 5 seeds (3 for the LDP, "
        "k-anonymity and projection sweeps). Code, data hashes and logs are released with the paper.")

    rep.h2("V. Results")
    rows = [
        ["Non-private MLP", "—", f(np_acc), f(R.base('full', 'MLP', 'mia_auc')), f(raw7[('gbm_attr_bal_acc', 'mean')]), f(raw_k['reid_avg_risk'])],
        ["LDP, ε_total = 32", "32-LDP", f(l32['accuracy']), f(l32['mia_auc']), f(l32['attr_bal_acc']), "—"],
        ["Mondrian k = 10", "syntactic", f(r10['accuracy']), f(r10['mia_auc']), f(r10['attr_bal_acc']), f(r10['reid_avg_risk'])],
        ["Mondrian k = 10, l = 2", "syntactic", f(r10l['accuracy']), f(r10l['mia_auc']), f(r10l['attr_bal_acc']), f(r10l['reid_avg_risk'])],
        ["DP-SGD ε = 1", "(1, 10⁻⁵)", f(R.dps('full', 1.0, 'accuracy')), f(R.dps('full', 1.0, 'mia_auc')), "n/a", "n/a"],
        ["DP-SGD ε = 0.1", "(0.1, 10⁻⁵)", f(R.dps('full', 0.1, 'accuracy')), f(R.dps('full', 0.1, 'mia_auc')), "n/a", "n/a"],
        ["D-DCA, m = 5", "empirical", f(dd(5, 'accuracy')), f(dd(5, 'mia_auc')), f(dd(5, 'gbm_attr_bal_acc')), "—"],
        ["Hybrid, ε = 0.1", "(0.1, 10⁻⁵)", f(R.hyv(HYB, 0.1, 'accuracy')), f(R.hyv(HYB, 0.1, 'mia_auc')), f(R.hyv(HYB, 0.1, 'attr_bal_acc')), "—"],
    ]
    rep.table(["Configuration", "Guarantee", "Accuracy", "MIA AUC", "Sex inference", "Re-id risk"], rows,
              caption="Representative configurations. Sex inference = balanced accuracy of a GBM attacker on the "
                      "released data (0.5 = none).", widths=[4, 2.6, 2.2, 2.2, 2.6, 2.4], size=8.5)
    rep.para(
        f"**Central DP.** Accuracy falls from {pct(R.dps('full', INF, 'accuracy'))} (clipping only) to "
        f"{pct(R.dps('full', 0.1, 'accuracy'))} at ε = 0.1 and {pct(R.dps('full', 0.05, 'accuracy'))} at ε = 0.05. In "
        f"the small-data regime, the non-private MLP's MIA advantage of {f(R.base('small', 'MLP', 'mia_advantage'))} "
        f"drops to {f(R.dps('small', INF, 'mia_advantage'))} with clipping alone and to "
        f"{f(R.dps('small', 0.05, 'mia_advantage'))} at ε = 0.05. Empirical attacks thus credit DP-SGD mainly for its "
        "regularisation; the formal ε matters for worst-case records that average-case attacks miss.")
    rep.figure(F2 / "fig2_dpsgd_sweep.png", "DP-SGD utility and MIA advantage across ε in two data regimes.", 16)
    rep.para(
        f"**Local vs central.** At equal ε, LDP histograms are {min(ratios):.0f}–{max(ratios):.0f}× less accurate than Laplace histograms. "
        f"Training on LDP records splits the budget over 13 attributes; at ε_total = 32, F1 reaches "
        f"{f(l32['f1'])} but sex is recoverable from the reports with balanced accuracy {f(l32['attr_bal_acc'])}.")
    rep.para(
        f"**Syntactic anonymisation.** Mondrian lowers average re-identification risk from "
        f"{f(raw_k['reid_avg_risk'])} to {f(r10['reid_avg_risk'])} (k = 10) without utility loss. Without "
        f"l-diversity, {pct(r10['homogeneity_rate'])} of records have their income disclosed; with 2-diversity, "
        f"none. Sex remains perfectly inferable up to k = 100: k-anonymity protects identity, not attributes "
        f"{c(13, 14)}.")
    rep.figure(F2 / "fig7_compressive.png", "Utility and sex leakage of projections versus dimension m.", 16)
    rep.para(
        f"**Compressive privacy.** PCA retains sex completely for m ≥ 10. D-DCA keeps the linear attacker at "
        f"0.500 for every m, and at m = 5 reduces the GBM attacker from {f(raw7[('gbm_attr_bal_acc', 'mean')])} "
        f"(raw) and {f(sup7[('gbm_attr_bal_acc', 'mean')])} (sex column dropped) to {f(dd(5, 'gbm_attr_bal_acc'))}, with "
        f"accuracy {pct(dd(5, 'accuracy'))}. At m = 2 it dominates PCA on both axes. A γ sweep shows that γ = 1 "
        f"already suffices. Residual non-linear leakage is the limit of linear projections {c(29)}.")
    gain = {e: R.hyv(HYB, e, 'accuracy') - R.hyv(DPF, e, 'accuracy') for e in (0.05, 0.1, 1.0)}
    rep.para(
        f"**Hybrid.** Contrary to our hypothesis, compression does not raise DP-SGD accuracy: differences at "
        f"ε = 0.05 and 0.1 ({100 * gain[0.05]:+.1f} and {100 * gain[0.1]:+.1f} points) are within one standard "
        f"deviation, and at ε = 1 the hybrid trails by {100 * -gain[1.0]:.1f} points. Its value is that the "
        "released features become attribute-protected while the model keeps its formal guarantee.")
    rep.figure(F2 / "fig8_hybrid.png", "Hybrid D-DCA → DP-SGD versus DP-SGD on all features.", 16)

    rep.h2("VI. Discussion and Limitations")
    rep.bullets([
        "**Guarantee ≠ empirical risk.** Formal ε and empirical MIA can diverge in both directions. Reports "
        "should state both.",
        "**Match the mechanism to what is released.** DP-SGD protects only models, k-anonymity only "
        "identities, D-DCA only attributes of released features, and LDP protects against the collector at "
        "a large utility cost.",
        f"**Limitations.** A single tabular dataset {c(23)}; lower-bound attacks; non-private selection of "
        f"hyper-parameters {c(27)}; reliance on a public split; possible disparate impact {c(28)} not measured.",
    ])

    rep.h2("VII. Conclusion")
    rep.para("A common model and attack suite turns heterogeneous privacy guarantees into comparable evidence. "
             "On Adult, no single mechanism covers membership, attribute and identity risk. D-DCA closes the "
             "attribute gap for released features at negligible cost, and its combination with DP-SGD offers "
             "the broadest coverage. Future work: non-linear desensitisation, DP-estimated projections, LiRA "
             f"attacks and additional datasets {c(25)}.")

    rep.h2("Acknowledgement")
    rep.para("I acknowledge the use of Claude (Anthropic), an AI assistant, for support with code development and report preparation.", size=9.5)

    rep.h2("References")
    for i, orig in enumerate(REFS, 1):
        rep.para(f"[{i}] {REFERENCES[orig - 1]}", align="left", size=9)
    rep.save(out)
