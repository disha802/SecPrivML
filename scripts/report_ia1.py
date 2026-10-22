"""IA-1 report: Problem Formulation and Preliminary Implementation."""

from __future__ import annotations

import pandas as pd

from _common import ROOT
from report_common import (COURSE, SUBMISSION_DATE, LONG_TITLE, MODULE_MAP_HEADER, MODULE_MAP_IA1,
                           STUDENT_FIELDS, TITLE, code_structure_rows, provenance, references)
from report_lib import Report

R1 = ROOT / "results" / "ia1"
F1 = ROOT / "figures" / "ia1"


def f(x, d=3):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def build(out):
    prov = provenance()
    p1 = pd.read_csv(R1 / "p1_baselines.csv")
    p2 = pd.read_csv(R1 / "p2_exposure.csv").set_index("analysis")
    p3 = pd.read_csv(R1 / "p3_dpsgd.csv")
    p4 = pd.read_csv(R1 / "p4_marginals.csv")
    p5 = pd.read_csv(R1 / "p5_kanon.csv")
    p6 = pd.read_csv(R1 / "p6_pca.csv")
    sample = pd.read_csv(R1 / "p5_kanon_sample_k10.csv")
    b = p1.drop(columns=["experiment"]).groupby(["regime", "model"]).agg(["mean", "std"])
    g = lambda reg, m, c: b.loc[(reg, m), (c, "mean")]
    s = lambda reg, m, c: b.loc[(reg, m), (c, "std")]
    reid = p2.loc["reidentification (raw QIs)"]
    att = lambda k: p2.loc[k]
    d3 = p3.groupby("target_eps").mean(numeric_only=True)
    l1 = p4.groupby(["eps", "mechanism"]).l1_error.mean().unstack()
    sizes = prov["sizes"]

    rep = Report(f"IA-1 | {COURSE} | Problem Formulation and Preliminary Implementation")
    rep.title_page(COURSE, "Internal Assessment 1 — Problem Formulation and Preliminary Implementation",
                   TITLE, LONG_TITLE, STUDENT_FIELDS + [("Submission date", SUBMISSION_DATE)])
    rep.toc()

    # ---------------------------------------------------------------- summary
    rep.h1("Executive Summary")
    rep.para(
        "Organisations increasingly train machine-learning (ML) models on personal records, and a "
        "growing set of privacy techniques claims to make this safe: differential privacy (DP) at "
        "training time, local DP at collection time, k-anonymity at publication time and compressive "
        "(lossy-projection) privacy at feature-release time. These techniques act at **different "
        "stages of the data pipeline, offer different kinds of guarantee and are usually evaluated in "
        "isolation with incomparable metrics**. A practitioner therefore cannot answer a simple "
        "question: *for a given utility target, which mechanism, at which stage, actually reduces "
        "which privacy risk?*")
    rep.para(
        f"This project builds **{TITLE}**, a unified evaluation framework in which one representative "
        "mechanism from each of Modules 2–5 is plugged into the same data pipeline, trained with the "
        "same model and attacked with the same Module-1 attack suite (membership inference, attribute "
        "inference and re-identification). The study uses the public UCI Adult census dataset "
        f"({sizes['final']['total_rows']:,} records after cleaning).")
    rep.para("IA-1 status. The problem, objectives, scope and five-module mapping are fixed; the dataset "
             "is acquired, hashed, cleaned and split; all five modules have working code with 30 passing "
             "unit tests; and preliminary experiments have produced the first baseline evidence:")
    rep.bullets([
        f"Non-private baselines reach **{pct(g('full', 'MLP', 'accuracy'))}** test accuracy (MLP). A fully "
        f"grown random forest leaks membership (MIA advantage **{f(g('full', 'RandomForest', 'mia_advantage'))}**), "
        f"and leakage grows in a small-data regime (MLP: {f(g('small', 'MLP', 'mia_advantage'))}).",
        f"**{pct(reid['unique_frac'])}** of records are unique on seven quasi-identifiers, and the sensitive "
        f"attribute *sex* can be inferred with **{pct(att('attribute inference, sex column dropped (gbm)')['attr_acc'])}** "
        "accuracy even after the column is deleted, owing to proxies such as *relationship*.",
        f"A from-scratch DP-SGD with a Rényi-DP accountant (validated against Opacus to 4 decimal places) "
        f"keeps **{pct(d3.loc[1.0, 'accuracy'])}** accuracy at ε = 1.",
        f"At equal ε = 1, local-DP histograms are **{l1.loc[1.0, 'Local-GRR'] / l1.loc[1.0, 'Central-Laplace']:.0f}×** "
        "less accurate than central-DP histograms: the price of removing the trusted curator.",
    ])

    rep.h2("Rubric compliance map")
    rep.table(["IA-1 stage", "Where addressed", "Evidence"], [
        ["1. Problem definition & module mapping", "Section 1", "Problem statement, objectives, scope, outcomes, Table 2 (module mapping)"],
        ["2. Conceptual understanding", "Section 2", "Concept notes, threat model, Figure 1 (workflow), treatment classification, justification"],
        ["3. Dataset identification & preparation", "Section 3", "Source/licence table, SHA-256 hashes, preparation log, schema, limitations"],
        ["4. Initial implementation", "Section 4", "Code structure, tools, design choices, execution instructions, run log, unit tests"],
        ["5. Preliminary evaluation & documentation", "Section 5", "Baseline results, observations, limitations, IA-2 evaluation plan, contribution statement"],
    ], caption="Mapping of IA-1 assessment stages to sections of this report.", widths=[4.5, 2.5, 9])

    # ---------------------------------------------------------------- 1
    rep.h1("1. Problem Definition and Module Mapping")
    rep.h2("1.1 Background and motivation")
    rep.para(
        "Removing names from a dataset is not anonymisation. Sweeney showed that 87% of Americans are "
        "unique on {ZIP, birth date, sex} [11]; Narayanan and Shmatikov de-anonymised Netflix users from "
        "sparse ratings [22]; and trained models themselves leak whether a person was in the training set "
        "(membership inference [19, 20]). Data-protection law now makes such leakage a compliance "
        "issue, e.g. India's Digital Personal Data Protection Act 2023 [31] and the EU GDPR [32]. "
        "The research community answered with a family of techniques, but each is typically "
        "presented with its own threat model and metric: ε for DP, k for anonymisation, "
        "reconstruction error for compressive privacy. They are rarely compared against the *same* "
        "attacker on the *same* data.")
    rep.h2("1.2 Problem statement")
    rep.callout(
        "**Problem statement.** Given a tabular dataset of personal records used to train a binary "
        "classifier (income > 50K), determine — under one common data pipeline, one utility model and "
        "one attack suite — how much privacy protection (against membership inference, sensitive-attribute "
        "inference and re-identification) and how much utility loss each of four privacy-preserving "
        "mechanisms produces when applied at its natural pipeline stage: **local DP at collection**, "
        "**k-anonymity at publication**, **DP-SGD at training** and **compressive projection at feature "
        "release**; and identify whether a combination of mechanisms gives a better privacy–utility "
        "trade-off than any single one.")
    rep.h2("1.3 Significance")
    rep.bullets([
        "**Practical decision support.** Engineers must choose *where* in a pipeline to spend a privacy "
        "budget. A side-by-side, attack-based comparison is directly actionable.",
        "**Exposes mismatched guarantees.** Syntactic methods (k-anonymity) protect identity but not "
        "attributes; DP protects membership but not population-level inference. A unified attack suite "
        "makes such gaps measurable rather than rhetorical.",
        "**Reproducible teaching artefact.** All five course modules become runnable, tested code on "
        "a real dataset, executable on a laptop CPU in minutes.",
    ])
    rep.h2("1.4 Objectives")
    rep.table(["ID", "Objective", "Measurable success criterion"], [
        ["O1", "Quantify the privacy exposure of the raw data and of non-private models", "MIA AUC/advantage, QI uniqueness, attribute-inference accuracy reported with ≥3 seeds"],
        ["O2", "Implement one mechanism per module (M2–M5) from first principles", "Unit-tested implementations; DP accountant matches a reference library"],
        ["O3", "Evaluate every mechanism with the same model and attack suite", "One results table covering utility + all applicable attacks"],
        ["O4", "Characterise each privacy–utility trade-off across its parameter (ε, k, m)", "Trade-off curves with mean ± sd across seeds"],
        ["O5", "Propose and test a combined mechanism", "Hybrid configuration compared to the single mechanisms at matched ε"],
    ], caption="Project objectives and success criteria.", widths=[1, 7, 8])
    rep.h2("1.5 Scope")
    rep.table(["In scope", "Out of scope (with reason)"], [
        ["Tabular binary classification on UCI Adult", "Images/text/LLMs — different threat surface; compute-heavy"],
        ["Membership, attribute and re-identification attacks", "Model inversion / reconstruction of training images (not meaningful for tabular one-hot data)"],
        ["DP-SGD (central), GRR/OUE/PM (local), Mondrian k-anonymity + l-diversity, PCA/DCA projections", "Cryptographic PPDM (secure multi-party computation, homomorphic encryption) — discussed analytically only"],
        ["Laptop-CPU reproducibility; NumPy/scikit-learn only", "Federated learning deployment, GPU-scale models"],
    ], caption="Project scope.", widths=[8, 8])
    rep.h2("1.6 Expected outcomes")
    rep.bullets([
        "A tested, open, reproducible Python framework (`src/privml`) covering all five modules.",
        "A provenance-documented, hashed dataset and preprocessing pipeline.",
        "Privacy–utility curves and a unified comparison table across mechanisms.",
        "Evidence-based guidance on which mechanism to apply at which pipeline stage.",
        "A research-manuscript draft (IA-2) describing the framework and the proposed hybrid.",
    ])
    rep.h2("1.7 Module-wise concept mapping")
    rep.table(MODULE_MAP_HEADER, MODULE_MAP_IA1,
              caption="Mapping of the problem to all five course modules (IA-1 version; refined in IA-2).",
              widths=[3.2, 5.2, 5.2, 2.6], bold_first_col=True)

    # ---------------------------------------------------------------- 2
    rep.h1("2. Conceptual Understanding")
    rep.h2("2.1 Threat model")
    rep.para(
        "The data owner holds records x = (age, workclass, education-num, marital-status, occupation, "
        "relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country) with label "
        "y = 1[income > 50K]. The adversary follows **Kerckhoffs's principle**: they know the mechanism and "
        "its parameters, and they hold a small auxiliary sample from the same distribution (modelled by "
        "the 10% *public split*). Three adversary goals are considered:")
    rep.table(["Attack", "Adversary observes", "Goal", "Success metric"], [
        ["Membership inference (MIA)", "Trained model's output on a target record", "Decide whether the target was in the training set", "ROC-AUC, advantage = max(TPR−FPR), TPR at 1% FPR"],
        ["Attribute inference", "Released record / representation (sex column possibly removed)", "Infer the target's *sex*", "Attacker accuracy and balanced accuracy vs. prior"],
        ["Re-identification (linkage)", "Released table + target's quasi-identifiers from an external source", "Single out the target's record", "Unique fraction, average prosecutor risk 1/|EC|"],
    ], caption="Threat model: adversaries evaluated in this project.", widths=[3.3, 4.5, 4, 4.2])

    rep.h2("2.2 Concept notes")
    rep.h3("Module 1 — Privacy considerations in ML")
    rep.para(
        "*Principle.* Privacy risk in ML arises at every stage: collection, storage/publication, training "
        "and deployment. Over-fitted models assign lower loss to training points, which the "
        "loss-threshold MIA of Yeom et al. [20] exploits: predict *member* if ℓ(f(x), y) < τ. "
        "*Assumptions:* the attacker can query the model and knows the true label. *Advantages:* "
        "empirical attacks measure leakage in the units that matter to an individual. *Limitations:* "
        "an attack only gives a lower bound on leakage; a weak attack does not prove privacy.")
    rep.h3("Module 2 — Differential privacy")
    rep.para("A randomised mechanism M is (ε, δ)-DP if for all neighbouring datasets D, D′ (differing in one "
             "record) and all output sets S:")
    rep.equation("Pr[M(D) ∈ S]  ≤  e^ε · Pr[M(D′) ∈ S] + δ")
    rep.para(
        "The Laplace mechanism adds Lap(Δ₁/ε) noise to a query of L1-sensitivity Δ₁. **DP-SGD** [2] "
        "trains a model privately: each step Poisson-samples records with rate q, clips every "
        "per-example gradient to norm C, sums, adds N(0, σ²C²I) and updates. Privacy over T steps is "
        "tracked with Rényi DP [3, 4]; for integer order α the per-step RDP of the sub-sampled Gaussian is")
    rep.equation("ε_RDP(α) = 1/(α−1) · log Σₖ₌₀^α C(α,k) (1−q)^(α−k) q^k exp((k²−k)/(2σ²))")
    rep.para(
        "which composes additively over T steps and is converted to (ε, δ)-DP [5]. "
        "*Assumptions:* a trusted curator holds the raw data. *Advantages:* a formal guarantee that holds "
        "against any attacker, including future ones; composition. *Limitations:* utility cost at small ε, "
        "and disparate impact on minority groups [28]; the guarantee says nothing about population-level "
        "inferences.")
    rep.h3("Module 3 — Local differential privacy")
    rep.para(
        "Each user perturbs their own value before sending it, so even the collector is not trusted. "
        "Generalised randomised response (GRR) [6, 7] reports the true value with probability "
        "p = e^ε/(e^ε + k − 1) and any other value otherwise; an unbiased frequency estimate is "
        "f̂ᵥ = (ĉᵥ/n − q)/(p − q). Optimised unary encoding (OUE) [8] is better for large domains "
        "(k > 3e^ε + 2). Numeric values use the Piecewise mechanism [9]. *Advantages:* no trusted party; "
        "deployed at scale (e.g. RAPPOR [10]). *Limitations:* estimation error grows as O(√n) rather than "
        "O(1), and multi-attribute records must split the budget across attributes.")
    rep.h3("Module 4 — Privacy-preserving data mining")
    rep.para(
        "A table is **k-anonymous** if every record is indistinguishable from at least k−1 others on its "
        "quasi-identifiers (QIs) [11]. Mondrian [12] achieves this by recursive median partitioning of "
        "the QI space. k-anonymity does not stop *homogeneity attacks* (all records in a class share "
        "the sensitive value), which motivates **l-diversity** [13] and **t-closeness** [14]. Information "
        "loss is measured with the Normalised Certainty Penalty (NCP) [15]. *Advantages:* the output is "
        "truthful microdata, interpretable by analysts. *Limitations:* no formal guarantee against "
        "background knowledge; it protects identity, not attributes.")
    rep.h3("Module 5 — Compressive privacy")
    rep.para(
        "Compressive privacy [17, 18] releases a lossy projection Z = W⊤(x − μ) with m ≪ d dimensions "
        "instead of the record itself. **PCA** keeps directions of maximal variance (utility-agnostic); "
        "**Discriminant Component Analysis (DCA)** keeps directions maximising the utility-class "
        "discriminant ratio, i.e. the top generalised eigenvectors of (S_B + ρ′I)w = λ(S_W + ρI)w. "
        "*Advantages:* data minimisation at source; cheap; composable with DP. *Limitations:* the "
        "guarantee is empirical (no ε), and a linear projection cannot remove non-linear information.")

    rep.h2("2.3 Methodology and workflow")
    rep.figure(F1 / "fig1_workflow.png",
               "Proposed workflow. Each module protects a different pipeline stage; all feed the same "
               "utility model and the same Module-1 attack suite, which makes results directly comparable.")
    rep.bullets([
        "**Step 1 — Data.** Download, hash, clean and split UCI Adult into public (10%), private-train (90%) "
        "and official test sets. Fit the encoder on the public split only.",
        "**Step 2 — Baseline exposure.** Measure QI uniqueness and attribute inference on raw data; train "
        "non-private LR/RF/MLP and attack them.",
        "**Step 3 — Mechanisms.** Apply each module's mechanism at its stage, sweeping its privacy parameter.",
        "**Step 4 — Common evaluation.** Train the same MLP on each protected output; report utility "
        "(accuracy, precision, recall, F1, AUC) and all applicable attacks.",
        "**Step 5 — Synthesis.** Compare trade-off curves and propose a combined mechanism.",
    ], numbered=False)

    rep.h2("2.4 Justification for concept selection")
    rep.table(["Choice", "Alternatives considered", "Why this choice"], [
        ["DP-SGD + RDP accountant", "Output perturbation, objective perturbation, PATE", "Model-agnostic, standard in practice, tight accounting; PATE needs many teachers and unlabelled public data"],
        ["GRR, OUE, Piecewise (LDP)", "Basic RR, RAPPOR, Duchi et al.", "GRR/OUE are the optimal frequency oracles for small/large domains [8]; PM is unbiased with bounded output [9]"],
        ["Mondrian k-anonymity + l-diversity", "Datafly, Incognito, t-closeness enforcement", "Multidimensional partitioning gives far lower information loss than single-dimension generalisation; l-diversity directly targets homogeneity"],
        ["PCA / DCA projections", "Autoencoders, adversarial representation learning", "Closed-form, deterministic, interpretable, and central to Kung's compressive-privacy framework [17]"],
        ["Loss-threshold MIA", "Shadow-model MIA [19], LiRA [21]", "Strong for tabular models, assumption-light and cheap enough to run on every configuration"],
    ], caption="Justification for the concepts and techniques selected.", widths=[4, 5, 7])
    rep.h2("2.5 Treatment of concepts")
    rep.table(["Concept", "Implemented", "Demonstrated", "Experimentally evaluated", "Analytical"], [
        ["MIA, attribute inference, re-identification (M1)", "✔", "", "✔", ""],
        ["Laplace mechanism / DP histogram (M2)", "✔", "✔", "✔", ""],
        ["DP-SGD + RDP accountant (M2)", "✔", "", "✔", ""],
        ["Gaussian mechanism (M2)", "✔", "✔", "", ""],
        ["GRR, OUE frequency oracles (M3)", "✔", "", "✔", ""],
        ["Piecewise mechanism, LDP record collection (M3)", "✔", "", "IA-2", ""],
        ["Mondrian k-anonymity, l-diversity (M4)", "✔", "✔", "✔", ""],
        ["t-closeness (M4)", "measured", "", "", "✔"],
        ["Secure multi-party computation / HE (M4)", "", "", "", "✔"],
        ["PCA (M5)", "✔", "", "✔", ""],
        ["DCA and privacy-aware projection (M5)", "✔", "", "IA-2", ""],
    ], caption="How each concept is treated (✔ = in IA-1; IA-2 = planned for IA-2).", widths=[6, 2.2, 2.4, 3, 2.2])

    # ---------------------------------------------------------------- 3
    rep.h1("3. Dataset Identification and Preliminary Preparation")
    rep.h2("3.1 Dataset and suitability")
    rep.para(
        "**UCI Adult (Census Income)** [23, 24] contains demographic and employment records extracted from "
        "the 1994 US Current Population Survey. It is suitable because (i) it contains genuine personal "
        "attributes, including classic quasi-identifiers (age, sex, race, marital status, native "
        "country) and a sensitive attribute (sex), which enables all three attack types; (ii) it is the "
        "most common benchmark in the DP, LDP and k-anonymity literature, so results can be compared "
        "with published work; (iii) it is openly licensed; and (iv) at ~45k records it is large enough "
        "for DP-SGD to be meaningful and small enough for laptop experiments.")
    rep.h2("3.2 Source details")
    rep.table(["Field", "Value"], [
        ["Name", prov["dataset"]],
        ["Landing page", prov["source_url"]],
        ["Download URL", prov["download_url"]],
        ["DOI", prov["doi"]],
        ["Creators", prov["creators"]],
        ["Publisher", prov["publisher"]],
        ["Licence", prov["license"]],
        ["Original collection", prov["original_collection"]],
        ["Version", prov["version"]],
        ["Retrieved on / method", f"{prov['retrieved_on']} — {prov['retrieval_method']}"],
        ["SHA-256 (adult.zip)", f"`{prov['hashes']['archive']['sha256']}`"],
    ], caption="Dataset source and provenance record (machine-readable copy: data/provenance.json).",
        widths=[3.5, 12.5], size=8.5)
    rep.h2("3.3 Schema and attribute roles")
    rep.table(["Attribute", "Type", "Role in this study"], [
        ["age", "numeric (17–90)", "feature; quasi-identifier"],
        ["workclass", "categorical (7)", "feature; quasi-identifier"],
        ["education-num", "numeric (1–16)", "feature; quasi-identifier"],
        ["marital-status", "categorical (7)", "feature; quasi-identifier"],
        ["occupation", "categorical (14)", "feature"],
        ["relationship", "categorical (6)", "feature; strong proxy of sex"],
        ["race", "categorical (5)", "feature; quasi-identifier"],
        ["sex", "binary", "feature; quasi-identifier; **sensitive attribute** for attribute inference"],
        ["capital-gain / capital-loss", "numeric", "feature"],
        ["hours-per-week", "numeric (1–99)", "feature"],
        ["native-country", "categorical (41)", "feature; quasi-identifier"],
        ["income", "binary", "**utility label** (>50K = 1); sensitive value for l-diversity"],
        ["fnlwgt, education", "—", "dropped (census sampling weight; duplicate of education-num)"],
    ], caption="Attribute schema and roles.", widths=[4, 3.5, 8.5])
    rep.h2("3.4 Preliminary preparation log")
    rep.para("All steps are executed by `scripts/01_prepare_data.py`; the log below is copied verbatim from "
             "its output and stored in `data/provenance.json`.")
    rep.table(["#", "Step"], [[i + 1, s_] for i, s_ in enumerate(prov["preprocessing_log"])],
              caption="Data preparation log.", widths=[1, 15], size=8.5)
    rep.table(["Quantity", "Original", "After preparation"], [
        ["Training rows", f"{sizes['original']['train_rows']:,}", f"{sizes['final']['train_rows']:,} (public {sizes['splits']['public']:,} + private {sizes['splits']['private_train']:,})"],
        ["Test rows", f"{sizes['original']['test_rows']:,}", f"{sizes['final']['test_rows']:,}"],
        ["Attributes", "14 + label", f"{sizes['final']['n_features_raw']} features + label ({sizes['final']['n_features_encoded']} after one-hot encoding)"],
        ["Positive rate (>50K)", "—", f"train {sizes['class_balance']['train_pos_rate']}, test {sizes['class_balance']['test_pos_rate']}"],
        ["Male share", "—", f"{sizes['sensitive_attribute']['train_male_rate']}"],
    ], caption="Dataset size before and after preparation.", widths=[4, 3, 9])
    rep.para(
        "**Why a public split?** 10% of the official training file is set aside as *public* data. The "
        "encoder's means/standard deviations and the compressive projections are fitted on it, so no "
        "statistic of the private training data leaks through preprocessing (a common, often-ignored "
        "leak in DP pipelines). The attacker's auxiliary data is drawn from the same split.")
    rep.h2("3.5 Initial exposure analysis")
    rep.figure(F1 / "fig2_data_exposure.png",
               "Left: income rate differs strongly by sex. Centre: relationship is a near-deterministic proxy of "
               "sex (Husband → male, Wife → female). Right: a quarter of records are unique on seven QIs.")
    rep.h2("3.6 Known dataset limitations")
    rep.bullets([
        "Collected in 1994 in the US; distributions are dated and not representative of other populations [25].",
        "Records with missing values (7.4% of rows) were dropped; this may bias the sample towards "
        "respondents with complete employment information.",
        "The dataset is already public, so the experiments *simulate* a private dataset; absolute risk "
        "numbers describe the methodology, not harm to real respondents.",
        "Sex is recorded as binary; this reflects the source and limits the attribute-inference analysis.",
    ])

    # ---------------------------------------------------------------- 4
    rep.h1("4. Initial Implementation")
    rep.h2("4.1 Tools and libraries")
    rep.table(["Tool", "Version", "Used for"], [
        ["Python", "3.12.4", "Language"],
        ["NumPy", "2.x", "MLP, DP-SGD, LDP mechanisms, projections"],
        ["SciPy", "1.18", "log-sum-exp & log-gamma for the accountant; generalised eigensolver"],
        ["pandas", "2.2", "Data handling, Mondrian tables, result CSVs"],
        ["scikit-learn", "1.8", "LR / RF baselines, gradient-boosting attacker, metrics"],
        ["matplotlib", "3.9", "Figures"],
        ["pytest", "9.0", "Unit tests"],
        ["Opacus", "1.5.2", "*Only* as an external reference to validate the accountant (not a runtime dependency)"],
    ], caption="Tools and libraries.", widths=[3, 2, 11])
    rep.h2("4.2 Code structure")
    rep.table(["Path", "Purpose"], code_structure_rows(), caption="Repository structure.",
              widths=[5.5, 10.5], size=8.5)
    rep.h2("4.3 Key implementation choices")
    rep.bullets([
        "**DP-SGD without a deep-learning framework.** For a one-hidden-layer MLP the per-example gradient "
        "norm has a closed form: ‖∇ℓᵢ‖² = ‖xᵢ‖²‖δᵢ‖² + ‖δᵢ‖² + eᵢ²‖hᵢ‖² + eᵢ², where eᵢ = pᵢ − yᵢ, hᵢ is "
        "the hidden activation and δᵢ = eᵢw₂ ⊙ 1[aᵢ > 0]. The clipped sum is Xᵀ(c ⊙ Δ). This avoids "
        "materialising per-example weight matrices and runs a 20-epoch DP-SGD in ~1.5 s on a CPU. It also "
        "removed an incompatibility between the installed PyTorch build and NumPy 2.",
        "**Poisson sub-sampling** (each record included independently with probability q) is used so that "
        "the implementation matches the assumption of the RDP accountant exactly.",
        "**Accountant validated externally.** ε values for four (q, σ, T) settings match Opacus 1.5.2 to "
        "four decimal places (`tests/test_dp.py`).",
        "**Noise calibration by bisection.** `calibrate_sigma` finds the smallest σ meeting a target ε.",
        "**Set-valued generalisation** for categorical QIs in Mondrian (Adult has no published "
        "hierarchies), encoded as fractional multi-hot vectors so models can train on anonymised data.",
        "**Seeds everywhere.** Every random step takes an explicit seed; results are reported over seeds.",
    ])
    rep.h2("4.4 Inputs, processing and outputs")
    rep.table(["Component", "Input", "Processing", "Output"], [
        ["Data preparation", "UCI archive (URL)", "hash → parse → drop missing → drop 2 columns → split", "clean CSVs, provenance.json"],
        ["Encoder", "public split", "standardise 5 numeric, one-hot 7 categorical", "87-dim feature matrix"],
        ["Baselines", "private-train features/labels", "fit LR, RF, MLP", "models"],
        ["DP-SGD", "features, labels, target ε, δ", "calibrate σ, Poisson sample, clip, noise, SGD", "private MLP + (ε, δ, σ, steps)"],
        ["Histogram release", "categorical column, ε", "Laplace (central) or GRR/OUE perturb + debias (local)", "estimated frequencies"],
        ["Mondrian", "raw table, k, (l)", "recursive median / set splits on 7 QIs", "equivalence classes, generalised table"],
        ["PCA", "public features, m", "eigen-decomposition of covariance", "projection W (87×m)"],
        ["Attack suite", "model or released data", "loss-threshold MIA; GBM/LR attribute attacker; class-size risk", "privacy metrics"],
    ], caption="Inputs, processing steps and outputs of each component.", widths=[2.8, 3.6, 5.8, 3.8], size=8.5)
    rep.h2("4.5 Execution instructions")
    rep.code("""
# 1. install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. download, hash and clean the dataset  (~10 s)
python scripts/01_prepare_data.py

# 3. run the IA-1 preliminary experiments  (~1 min)
python scripts/02_run_ia1.py

# 4. make figures and run the unit tests
python scripts/04_make_figures.py
python -m pytest
""")
    rep.h2("4.6 Output log (excerpt)")
    log = (R1 / "run_log.txt").read_text(encoding="utf-8").splitlines()
    rep.code("\n".join(l_[:110] for l_ in log[1:]), size=7.5)
    rep.h2("4.7 Verification")
    rep.para("`python -m pytest` runs **30 unit tests, all passing**. They check: the accountant against "
             "Opacus; ε monotonicity and σ calibration; analytic per-example gradients against finite "
             "differences; Laplace noise scale; unbiasedness of the GRR, OUE and Piecewise estimators; "
             "that Mondrian output is a disjoint k-anonymous partition with max risk ≤ 1/k; that l-diversity "
             "removes homogeneous classes; that D-DCA removes a linear privacy signal on synthetic data; "
             "that the MIA detects a memorising model; and that dataset hashes match the provenance record.")

    # ---------------------------------------------------------------- 5
    rep.h1("5. Preliminary Evaluation and Documentation")
    rep.h2("5.1 Evaluation measures")
    rep.table(["Dimension", "Metric", "Interpretation"], [
        ["Utility", "Accuracy, precision, recall, F1 (positive = >50K), ROC-AUC", "F1 matters because only ~25% of records are positive"],
        ["Membership privacy", "MIA ROC-AUC; advantage = max(TPR−FPR); TPR at 1% FPR", "0.5 / 0 / 0.01 = no leakage"],
        ["Attribute privacy", "Attacker accuracy and balanced accuracy on sex", "0.5 balanced accuracy = no leakage"],
        ["Identity privacy", "Fraction unique on QIs; average & max prosecutor risk", "max risk ≤ 1/k under k-anonymity"],
        ["Disclosure (M4)", "Homogeneity rate; t-closeness", "share of records whose income is revealed"],
        ["Information loss", "NCP (M4); normalised reconstruction MSE (M5)", "0 = no loss"],
        ["Formal guarantee", "(ε, δ) for DP/LDP mechanisms", "smaller ε = stronger guarantee"],
    ], caption="Evaluation measures.", widths=[3, 7, 6])

    rep.h2("5.2 Baseline results (non-private models)")
    rows = []
    for reg in ("full", "small"):
        for m, lab in (("LogisticRegression", "LR"), ("RandomForest", "RF"), ("MLP", "MLP")):
            rows.append([f"{lab} ({'n=27,146' if reg == 'full' else 'n=1,000'})",
                         f"{f(g(reg, m, 'accuracy'))} ± {f(s(reg, m, 'accuracy'))}",
                         f(g(reg, m, 'f1')), f(g(reg, m, 'auc')),
                         f"{f(g(reg, m, 'mia_auc'))} ± {f(s(reg, m, 'mia_auc'))}",
                         f(g(reg, m, 'mia_advantage')), f(g(reg, m, 'train_loss')), f(g(reg, m, 'test_loss'))])
    rep.table(["Model", "Accuracy", "F1", "AUC", "MIA AUC", "MIA adv.", "Train loss", "Test loss"], rows,
              caption="Non-private baselines (mean ± sd over 3 seeds). Small-data regime = random 1,000-record "
                      "training subset with a high-capacity MLP (256 hidden units, 200 epochs).",
              widths=[3.2, 2.6, 1.5, 1.5, 2.6, 1.6, 1.6, 1.6], size=8.5)
    rep.figure(F1 / "fig3_baselines.png", "Baseline utility and membership-inference advantage.", 15)
    rep.para(
        f"With the full training set, LR and the MLP generalise well (train/test loss gap ≈ "
        f"{f(g('full', 'MLP', 'test_loss') - g('full', 'MLP', 'train_loss'), 2)}), and average-case MIA is near "
        f"chance (AUC ≈ {f(g('full', 'MLP', 'mia_auc'))}). The fully grown random forest memorises its training "
        f"data (train loss {f(g('full', 'RandomForest', 'train_loss'))} vs test {f(g('full', 'RandomForest', 'test_loss'))}) "
        f"and leaks membership (advantage {f(g('full', 'RandomForest', 'mia_advantage'))}). In the small-data regime "
        f"the MLP's advantage rises to {f(g('small', 'MLP', 'mia_advantage'))} and the RF's to "
        f"{f(g('small', 'RandomForest', 'mia_advantage'))}: **leakage is driven by the generalisation gap**, "
        "consistent with Yeom et al. [20].")

    rep.h2("5.3 Privacy exposure of the raw data")
    rep.table(["Analysis", "Result"], [
        ["Records unique on 7 QIs", pct(reid["unique_frac"])],
        ["Average prosecutor re-identification risk", f(reid["reid_avg_risk"])],
        ["Number of QI equivalence classes", f"{int(reid['n_classes']):,}"],
        ["Infer sex — all features, linear attacker", pct(att("attribute inference, all features (lr)")["attr_acc"])],
        ["Infer sex — sex column dropped, linear attacker", pct(att("attribute inference, sex column dropped (lr)")["attr_acc"])],
        ["Infer sex — sex column dropped, GBM attacker", pct(att("attribute inference, sex column dropped (gbm)")["attr_acc"])],
        ["Majority-class prior for sex", pct(att("attribute inference, all features (lr)")["attr_prior"])],
    ], caption="Exposure of the unprotected data.", widths=[10, 6])
    rep.para("Two Module-1 lessons follow immediately: (i) removing direct identifiers does not anonymise "
             "— one in four records can be singled out by linkage; (ii) deleting a sensitive column "
             "(\"fairness/privacy through unawareness\") is weak protection, because proxies retain "
             f"most of the signal ({pct(att('attribute inference, sex column dropped (lr)')['attr_bal_acc'])} balanced "
             "accuracy versus 50% for no information).")

    rep.h2("5.4 Preliminary DP-SGD results")
    rows = [[f"{e:g}", f(d3.loc[e, 'sigma']), f(d3.loc[e, 'epsilon']), f(d3.loc[e, 'accuracy']),
             f(d3.loc[e, 'f1']), f(d3.loc[e, 'mia_auc']), f(d3.loc[e, 'mia_advantage'])]
            for e in d3.index if e != float("inf")]
    rows.append(["∞ (clip only)", "0", "∞", f(d3.loc[float('inf'), 'accuracy']), f(d3.loc[float('inf'), 'f1']),
                 f(d3.loc[float('inf'), 'mia_auc']), f(d3.loc[float('inf'), 'mia_advantage'])])
    rep.table(["Target ε", "σ", "Accounted ε", "Accuracy", "F1", "MIA AUC", "MIA adv."], rows,
              caption="DP-SGD on the full training set (δ = 1e-5, 20 epochs, batch 256, C = 1; mean of 3 seeds).",
              widths=[2.4, 2, 2.4, 2.2, 2, 2.2, 2.2], size=8.5)
    rep.figure(F1 / "fig4_dpsgd_prelim.png", "Preliminary DP-SGD utility versus privacy budget.", 11)
    rep.para("Worked example of the accounting: with n = 27,146, batch 256 ⇒ q = 0.00943, T = round(20/q) = "
             f"2,121 steps. For ε = 1 the bisection returns σ = {f(d3.loc[1.0, 'sigma'])}; the accountant gives ε = "
             f"{f(d3.loc[1.0, 'epsilon'], 4)} at δ = 10⁻⁵.")

    rep.h2("5.5 Central versus local DP for histogram release")
    rows = [[f"{e:g}", f"{l1.loc[e, 'Central-Laplace']:.5f}", f"{l1.loc[e, 'Local-GRR']:.4f}",
             f"{l1.loc[e, 'Local-OUE']:.4f}", f"{l1.loc[e, 'Local-GRR'] / l1.loc[e, 'Central-Laplace']:.0f}×"]
            for e in l1.index]
    rep.table(["ε", "Central Laplace", "Local GRR", "Local OUE", "GRR / central"], rows,
              caption="Mean L1 error of the 14-category occupation histogram (20 repetitions).",
              widths=[2, 3.5, 3.5, 3.5, 3.5])
    rep.figure(F1 / "fig5_central_vs_local_prelim.png", "Central vs. local DP error (log–log).", 11)
    rep.para("Central DP is two to three orders of magnitude more accurate at the same ε. Between the local "
             "oracles, OUE wins for ε ≤ 1 and GRR for ε ≥ 2, matching the theoretical cross-over "
             "k = 3e^ε + 2 (for k = 14 the cross-over is ε ≈ 1.3) [8].")

    rep.h2("5.6 Preliminary k-anonymity")
    rep.table(["k", "Classes", "NCP", "Homogeneity rate", "Unique fraction", "Avg risk", "Max risk"],
              [[int(r.k), int(r.n_classes), f(r.ncp), pct(r.homogeneity_rate), pct(r.unique_frac),
                f(r.reid_avg_risk), f(r.reid_max_risk)] for r in p5.itertuples()],
              caption="Mondrian k-anonymity on the 7 QIs (private training split).",
              widths=[1.2, 2, 2, 3, 2.6, 2.4, 2.4])
    cols = ["age", "workclass", "education-num", "marital-status", "race", "sex", "native-country", "income"]
    rep.table(cols, sample[cols].astype(str).values.tolist(),
              caption="First records of the 10-anonymous release (generalised QIs shown as ranges/sets).",
              widths=[1.5, 2.2, 1.6, 3.6, 1.3, 1.4, 3.2, 1.2], size=7.5)
    rep.para(f"Mondrian removes all unique records with little information loss (NCP {f(p5.ncp.iloc[1])} at k = 10). "
             f"However, **{pct(p5.homogeneity_rate.iloc[1])} of records sit in classes where every member has the "
             "same income**, so their income is disclosed despite k-anonymity — the homogeneity attack that "
             "motivates l-diversity in IA-2.")

    rep.h2("5.7 Preliminary compressive privacy (PCA)")
    rep.table(["m", "Reconstruction NMSE", "Infer sex (LR) bal. acc.", "Infer sex (GBM) bal. acc."],
              [[int(r.m), f(r.reconstruction_nmse), f(r.lr_attr_bal_acc), f(r.gbm_attr_bal_acc)]
               for r in p6.itertuples()],
              caption="PCA on all 87 features: sensitive-attribute leakage from the projection.",
              widths=[2, 4.5, 4.5, 4.5])
    rep.para("Utility-agnostic PCA is *not* privacy-preserving: at m ≥ 10 the sex column survives the "
             "projection almost intact (balanced accuracy ≈ 1.0). Compression alone only helps when m is "
             "very small, which also destroys utility. A projection that is explicitly *penalised* for "
             "retaining the sensitive attribute is required.")

    rep.h2("5.8 Preliminary observations")
    rep.table(["#", "Observation", "Consequence for IA-2"], [
        ["O1", "Average-case MIA on well-generalising models is near chance; leakage concentrates in high-capacity / small-data settings", "Add a small-data regime to every DP experiment; report TPR at 1% FPR"],
        ["O2", "DP-SGD costs < 0.3 points of accuracy for ε ≥ 0.5 on Adult", "Extend the ε grid down to 0.05 to expose the real trade-off"],
        ["O3", "k-anonymity removes uniqueness but leaves high homogeneity", "Evaluate l-diversity; measure attribute inference on anonymised data"],
        ["O4", "PCA retains the sensitive attribute", "Design a privacy-aware projection (Desensitised DCA) and a hybrid with DP"],
        ["O5", "Local DP loses ~100–1000× accuracy vs central DP in histograms", "Measure the effect on model training with LDP-collected records"],
    ], caption="Preliminary observations and their consequences.", widths=[1, 8, 7])

    rep.h2("5.9 Known limitations at IA-1")
    rep.bullets([
        "Only three seeds; confidence intervals are wide for small effects.",
        "Hyper-parameters of DP-SGD (lr = 0.05, C = 1, 20 epochs) were chosen by a brief sanity run; "
        "tuning on private data has its own privacy cost [27] and is not yet accounted for.",
        "The loss-threshold MIA is a lower bound on leakage; stronger attacks (LiRA [21]) may find more.",
        "LDP record collection, DCA variants and the hybrid have code but no experiments yet.",
    ])

    rep.h2("5.10 Proposed evaluation plan for IA-2")
    rep.table(["Exp.", "Question", "Design", "Metrics"], [
        ["E1", "How much do non-private models leak?", "LR / RF / MLP, full and small data, 5 seeds", "utility, MIA"],
        ["E2", "Privacy–utility curve of DP-SGD", "ε ∈ {0.05 … 8, ∞}, both regimes, 5 seeds", "utility, MIA, σ"],
        ["E3", "Sensitivity to DP-SGD hyper-parameters", "clip C, learning rate, epochs at ε = 1", "accuracy"],
        ["E4", "Cost of removing the trusted curator", "Laplace vs GRR vs OUE, 4 attributes, 50 reps", "L1, MSE"],
        ["E5", "Can we train on LDP-collected records?", "ε_total ∈ {1 … 128}, with/without label perturbation", "utility, MIA, attribute inference"],
        ["E6", "k-anonymity and l-diversity trade-off", "k ∈ {2 … 250}, l ∈ {none, 2}", "NCP, risk, homogeneity, utility, attribute inference"],
        ["E7", "Can a projection hide sex but keep income?", "PCA / DCA / D-DCA, m ∈ {2 … 40}, γ sweep", "utility, LR & GBM attribute inference, reconstruction"],
        ["E8", "Does compression help DP-SGD?", "D-DCA → DP-SGD vs DP-SGD across ε", "utility, MIA, attribute inference"],
    ], caption="Evaluation plan for IA-2.", widths=[1.2, 5, 5.5, 4.3], size=8.5)

    rep.h2("5.11 Individual contribution statement")
    rep.para(
        "This is an individual project. I defined the problem and threat model, selected the dataset and "
        "documented its provenance, and designed the unified evaluation methodology. I implemented the "
        "`src/privml` package — the data pipeline with a public split, the attack suite, the RDP "
        "accountant and DP-SGD (including the closed-form per-example gradients of the NumPy MLP), the "
        "GRR/OUE/Piecewise mechanisms, Mondrian k-anonymity with l-diversity, and the PCA/DCA "
        "projections — together with the unit tests, experiment scripts, figures and this report. "
        "Third-party libraries are used only for standard components (NumPy/SciPy linear algebra, "
        "scikit-learn baselines and attackers, matplotlib); Opacus was used solely as an external "
        "check of the accountant.")

    references(rep)
    rep.save(out)
