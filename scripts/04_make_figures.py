"""
Step 4 — Publication figures from results/ia1 and results/ia2 CSVs.

Usage:  python scripts/04_make_figures.py
Outputs: figures/ia1/*.png, figures/ia2/*.png (300 dpi)
"""

from _common import ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Validated categorical palette (fixed order) + text/surface tokens
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

R1, R2 = ROOT / "results" / "ia1", ROOT / "results" / "ia2"
F1, F2 = ROOT / "figures" / "ia1", ROOT / "figures" / "ia2"


def agg(df, by, cols):
    g = df.groupby(by)[cols]
    return g.mean().join(g.std(ddof=1).fillna(0), rsuffix="_std").reset_index()


def line(ax, x, y, e=None, color=C[0], label=None, marker="o", ls="-"):
    ax.plot(x, y, color=color, marker=marker, ls=ls, label=label)
    if e is not None:
        ax.fill_between(x, np.asarray(y) - e, np.asarray(y) + e, color=color, alpha=0.15, lw=0)


def eps_axis(ax, vals):
    ax.set_xscale("log")
    ax.set_xticks(vals, [f"{v:g}" for v in vals])
    ax.minorticks_off()


# entity -> colour, fixed across figures (colour follows the entity, never its rank)
ENT = {"DP-SGD (all features)": C[0], "PCA (all features)": C[0], "PCA (sex suppressed)": C[1],
       "PCA-5 + DP-SGD": C[1], "DCA (sex suppressed)": C[2], "D-DCA (proposed)": C[3],
       "D-DCA-5 + DP-SGD (proposed)": C[3], "D-DCA (joint label)": C[4], "D-DCA-10 + DP-SGD": C[6]}
EPS_T = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8]


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


# ------------------------------------------------------------------ diagrams
def box(ax, x, y, w, h, text, fc, ec, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))


def architecture(path, final: bool):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.grid(False)
    light = ["#e7f0fb", "#fdebe3", "#e3f5ee", "#fdf3dc", "#fbe9f0", "#efedfb"]
    box(ax, 0.1, 4.5, 2.1, 0.8, "UCI Adult (CC BY 4.0)\nSHA-256 provenance", "#f4f4f2", INK2, bold=True)
    box(ax, 0.1, 3.3, 2.1, 0.8, "Clean + encode\npublic 10% / private 90%\n/ official test", "#f4f4f2",
        INK2, fs=7)
    arrow(ax, 1.15, 4.5, 1.15, 4.1)
    stages = [
        ("M3  Local DP\n(collection)", "GRR / OUE / Piecewise\nper-user ε split", light[2], C[2]),
        ("M4  PPDM\n(publishing)", "Mondrian k-anonymity\n+ l-diversity", light[3], C[3]),
        ("M2  Central DP\n(training)", "DP-SGD + RDP\naccountant; Laplace\nhistograms", light[0], C[0]),
        ("M5  Compressive\n(feature release)", "PCA / DCA / D-DCA" + ("\n+ hybrid\nD-DCA → DP-SGD" if final else ""),
         light[4], C[4]),
    ]
    for i, (t, sub, fc, ec) in enumerate(stages):
        x = 2.75 + i * 1.82
        box(ax, x, 3.95, 1.68, 0.85, t, fc, ec, bold=True)
        box(ax, x, 3.0, 1.68, 0.8, sub, SURF, ec, fs=7)
    arrow(ax, 2.2, 3.7, 2.75, 4.3)
    box(ax, 2.75, 1.75, 7.1, 0.85,
        "Shared utility model: MLP (87→64→1)   •   Metrics: accuracy, precision, recall, F1, ROC-AUC",
        "#f4f4f2", INK2)
    box(ax, 2.75, 0.45, 7.1, 1.0,
        "M1  Unified attack suite (Kerckhoffs: attacker knows the mechanism)\n"
        "Membership inference (AUC, advantage, TPR@1%FPR)  •  Attribute inference of sex (LR, GBM)\n"
        "Re-identification / homogeneity on QIs  •  Formal guarantee (ε, δ) where defined",
        light[1], C[1], bold=False)
    for i in range(4):
        arrow(ax, 2.75 + i * 1.82 + 0.84, 3.0, 2.75 + i * 1.82 + 0.84, 2.6)
    arrow(ax, 6.3, 1.75, 6.3, 1.45)
    box(ax, 0.1, 0.45, 2.1, 2.15, "Comparison\n& Pareto analysis\n\nprivacy–utility\ntrade-off per\npipeline stage",
        "#f4f4f2", INK2, bold=True)
    arrow(ax, 2.75, 0.95, 2.2, 0.95)
    ax.set_title("PrivBench-Adult: " + ("final architecture (IA-2)" if final else "proposed workflow (IA-1)"),
                 loc="left", color=INK)
    save(fig, path)


# ----------------------------------------------------------------------- IA-1
def ia1():
    architecture(F1 / "fig1_workflow.png", final=False)
    train = pd.read_csv(ROOT / "data" / "processed" / "adult_train_clean.csv")

    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
    rate = train.groupby("sex")["income"].mean()
    axs[0].bar(rate.index, rate.values, color=[C[0], C[1]], width=0.55)
    axs[0].set_title("P(income>50K) by sex")
    axs[0].set_ylim(0, rate.max() * 1.2)
    for i, v in enumerate(rate.values):
        axs[0].text(i, v + 0.01, f"{v:.2f}", ha="center", color=INK)
    rel = train.groupby("relationship")["sex"].apply(lambda s: (s == "Male").mean()).sort_values()
    axs[1].barh(rel.index, rel.values, color=C[0], height=0.6)
    axs[1].set_title("Share male by relationship\n(proxy for sex)")
    axs[1].set_xlim(0, 1)
    key = train[["age", "sex", "race", "marital-status", "native-country", "education-num",
                 "workclass"]].astype(str).agg("|".join, axis=1)
    sizes = key.map(key.value_counts())
    bins = [1, 2, 5, 10, 50, 10_000]
    labels = ["1 (unique)", "2–4", "5–9", "10–49", "≥50"]
    share = pd.cut(sizes, bins=bins, right=False, labels=labels).value_counts(normalize=True)[labels]
    axs[2].bar(labels, share.values, color=C[7], width=0.6)
    axs[2].set_title("Records by QI equivalence-class size")
    axs[2].tick_params(axis="x", rotation=20)
    axs[2].set_ylim(0, share.max() * 1.2)
    for i, v in enumerate(share.values):
        axs[2].text(i, v + 0.01, f"{v:.0%}", ha="center", color=INK, fontsize=8)
    fig.tight_layout()
    save(fig, F1 / "fig2_data_exposure.png")

    p1 = pd.read_csv(R1 / "p1_baselines.csv")
    a = agg(p1, ["model", "regime"], ["accuracy", "mia_auc", "mia_advantage"])
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
    models = ["LogisticRegression", "RandomForest", "MLP"]
    for i, reg in enumerate(["full", "small"]):
        sub = a[a.regime == reg].set_index("model").loc[models]
        lab = "full data (n=27,146)" if reg == "full" else "small data (n=1,000)"
        x = np.arange(3) + (i - 0.5) * 0.25
        axs[0].errorbar(x, sub["accuracy"], yerr=sub["accuracy_std"], fmt="o", ms=8, color=C[i],
                        ecolor=INK2, elinewidth=0.8, label=lab)
        axs[1].bar(x, sub["mia_advantage"], 0.23, yerr=sub["mia_advantage_std"], color=C[i],
                   label=lab, error_kw=dict(ecolor=INK2, lw=0.8))
    for ax in axs:
        ax.set_xticks(range(3), ["LR", "RF", "MLP"])
    axs[0].set_title("Test accuracy (mean ± sd, 3 seeds)")
    axs[1].set_title("MIA advantage (TPR−FPR)")
    axs[1].legend(loc="upper left")
    fig.tight_layout()
    save(fig, F1 / "fig3_baselines.png")

    p3 = agg(pd.read_csv(R1 / "p3_dpsgd.csv"), ["target_eps"], ["accuracy", "f1"])
    fin = p3[np.isfinite(p3.target_eps)]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    line(ax, fin.target_eps, fin.accuracy, fin.accuracy_std, C[0], "DP-SGD")
    base = p1[(p1.model == "MLP") & (p1.regime == "full")].accuracy.mean()
    ax.axhline(base, color=INK2, ls="--", lw=1)
    ax.text(fin.target_eps.min(), base - 0.002, "non-private MLP", ha="left", va="top", color=INK2)
    ax.set_ylim(0.82, 0.86)
    ax.set_xscale("log")
    ax.set_xticks(fin.target_eps, [f"{v:g}" for v in fin.target_eps])
    ax.minorticks_off()
    ax.set_xlabel("privacy budget ε (δ = 1e-5)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Preliminary DP-SGD utility")
    save(fig, F1 / "fig4_dpsgd_prelim.png")

    p4 = agg(pd.read_csv(R1 / "p4_marginals.csv"), ["mechanism", "eps"], ["l1_error"])
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for i, m in enumerate(["Central-Laplace", "Local-GRR", "Local-OUE"]):
        s = p4[p4.mechanism == m]
        line(ax, s.eps, s.l1_error, None, C[i], m)
        ax.text(s.eps.iloc[-1] * 1.08, s.l1_error.iloc[-1], m, color=INK, va="center", fontsize=8)
    eps_axis(ax, [0.1, 0.25, 0.5, 1, 2, 4, 8])
    ax.set_yscale("log")
    ax.set_xlim(0.08, 20)
    ax.set_xlabel("ε")
    ax.set_ylabel("L1 error of occupation histogram")
    ax.set_title("Central vs local DP (preliminary)")
    save(fig, F1 / "fig5_central_vs_local_prelim.png")


# ----------------------------------------------------------------------- IA-2
def ia2():
    architecture(F2 / "fig1_architecture.png", final=True)
    e1 = pd.read_csv(R2 / "e1_baselines.csv")
    e2 = pd.read_csv(R2 / "e2_dpsgd.csv")
    cols = ["accuracy", "f1", "auc", "mia_auc", "mia_advantage", "mia_tpr_at_1fpr", "sigma"]

    # DP-SGD sweep: utility and MIA, both regimes (two separate panels, one axis each)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    for i, reg in enumerate(["full", "small"]):
        a = agg(e2[e2.regime == reg], ["target_eps"], cols)
        fin = a[np.isfinite(a.target_eps)]
        lab = "full data (n=27,146)" if reg == "full" else "small data (n=1,000)"
        line(axs[0], fin.target_eps, fin.accuracy, fin.accuracy_std, C[i], lab)
        line(axs[1], fin.target_eps, fin.mia_advantage, fin.mia_advantage_std, C[i], lab)
        np_mlp = e1[(e1.model == "MLP") & (e1.regime == reg)]
        axs[0].axhline(np_mlp.accuracy.mean(), color=C[i], ls=":", lw=1.2)
        axs[1].axhline(np_mlp.mia_advantage.mean(), color=C[i], ls=":", lw=1.2)
    for ax in axs:
        eps_axis(ax, EPS_T)
        ax.set_xlabel("privacy budget ε")
    axs[0].set_ylabel("test accuracy")
    axs[0].set_title("DP-SGD utility (dotted = non-private MLP)")
    axs[1].set_ylabel("MIA advantage")
    axs[1].set_title("Membership-inference advantage")
    axs[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, F2 / "fig2_dpsgd_sweep.png")

    e3 = pd.read_csv(R2 / "e3_ablation.csv")
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
    for ax, (p, lab) in zip(axs, [("clip", "clipping norm C"), ("lr", "learning rate"),
                                  ("epochs", "epochs")]):
        a = agg(e3[e3.param == p], ["value"], ["accuracy"])
        line(ax, a.value, a.accuracy, a.accuracy_std, C[0])
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.set_xticks(a.value, [f"{v:g}" for v in a.value])
        ax.minorticks_off()
    axs[0].set_ylabel("test accuracy (ε = 1)")
    fig.suptitle("DP-SGD hyper-parameter sensitivity at ε = 1", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, F2 / "fig3_dpsgd_ablation.png")

    e4 = agg(pd.read_csv(R2 / "e4_marginals.csv"), ["attribute", "k", "mechanism", "eps"], ["l1_error"])
    attrs = ["workclass", "marital-status", "occupation", "native-country"]
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, at in zip(axs, attrs):
        for i, m in enumerate(["Central-Laplace", "Local-GRR", "Local-OUE"]):
            s = e4[(e4.attribute == at) & (e4.mechanism == m)]
            line(ax, s.eps, s.l1_error, None, C[i], m)
        eps_axis(ax, [0.1, 0.5, 2, 8])
        ax.set_yscale("log")
        ax.set_title(f"{at} (k={int(s.k.iloc[0])})")
        ax.set_xlabel("ε")
    axs[0].set_ylabel("L1 error")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, F2 / "fig4_central_vs_local.png")

    e5 = pd.read_csv(R2 / "e5_ldp_training.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.9))
    for i, v in enumerate(["full-record LDP", "features-only LDP"]):
        a = agg(e5[e5.variant == v], ["eps_total"], ["f1", "accuracy"])
        line(axs[0], a.eps_total, a.f1, a.f1_std, C[i], v)
    a = agg(e5[e5.variant == "full-record LDP"], ["eps_total"], ["attr_bal_acc"])
    line(axs[1], a.eps_total, a.attr_bal_acc, a.attr_bal_acc_std, C[7], "GBM attacker")
    axs[1].axhline(0.5, color=INK2, ls="--", lw=1)
    axs[1].text(128, 0.52, "random guess", color=INK2, fontsize=8, ha="right")
    axs[0].axhline(e1[(e1.model == "MLP") & (e1.regime == "full")].f1.mean(), color=INK2, ls=":", lw=1.2)
    for ax in axs:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"])
        ax.minorticks_off()
        ax.set_xlabel("per-user budget ε_total (split over 13 attributes)")
    axs[0].set_ylabel("test F1 (positive = >50K)")
    axs[0].set_title("Utility of models trained on LDP data")
    axs[0].legend(loc="upper left")
    axs[1].set_ylabel("balanced accuracy")
    axs[1].set_title("Attribute inference of true sex from LDP report")
    fig.tight_layout()
    save(fig, F2 / "fig5_ldp_training.png")

    e6 = pd.read_csv(R2 / "e6_kanon.csv")
    e6k = e6[e6.k > 1]
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
    panels = [("reid_avg_risk", "avg re-identification risk"), ("homogeneity_rate", "income disclosed\n(homogeneity rate)"),
              ("ncp", "information loss (NCP)"), ("accuracy", "test accuracy")]
    for ax, (m, t) in zip(axs, panels):
        for i, l in enumerate(["none", "2"]):
            a = agg(e6k[e6k.l.astype(str) == l], ["k"], [m])
            line(ax, a.k, a[m], a[f"{m}_std"], C[i], "k-anonymity" if l == "none" else "k-anon + 2-diversity")
        eps_axis(ax, [2, 10, 50, 250])
        ax.set_xlabel("k")
        ax.set_title(t)
    raw = e6[e6.k == 1].iloc[0]
    axs[0].axhline(raw.reid_avg_risk, color=C[7], ls="--", lw=1)
    axs[0].text(2, raw.reid_avg_risk * 0.9, "raw data", color=INK2, fontsize=8, va="top")
    axs[3].axhline(e1[(e1.model == "MLP") & (e1.regime == "full")].accuracy.mean(), color=INK2, ls=":", lw=1.2)
    axs[3].set_ylim(0.8, 0.87)
    axs[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, F2 / "fig6_kanonymity.png")

    e7 = pd.read_csv(R2 / "e7_compressive.csv")
    main = e7[e7.experiment == "E7"]
    projs = ["PCA (all features)", "PCA (sex suppressed)", "DCA (sex suppressed)", "D-DCA (proposed)",
             "D-DCA (joint label)"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    for p in projs:
        a = agg(main[main.projection == p], ["m"], ["accuracy", "gbm_attr_bal_acc", "lr_attr_bal_acc"])
        ls = "--" if p == "D-DCA (joint label)" else "-"
        line(axs[0], a.m, a.accuracy, a.accuracy_std, ENT[p], p, ls=ls)
        line(axs[1], a.m, a.lr_attr_bal_acc, None, ENT[p], p, ls=ls)
        line(axs[2], a.m, a.gbm_attr_bal_acc, None, ENT[p], p, ls=ls)
    axs[1].text(8, 0.6, "both D-DCA variants\nstay at 0.50 (chance)", color=INK, fontsize=7)
    for ax, t in zip(axs, ["utility: test accuracy", "linear attacker (LR) on sex",
                           "non-linear attacker (GBM) on sex"]):
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 5, 10, 20, 40], ["2", "3", "5", "10", "20", "40"])
        ax.minorticks_off()
        ax.set_xlabel("projected dimension m")
        ax.set_title(t)
    for ax in axs[1:]:
        ax.axhline(0.5, color=INK2, ls="--", lw=1)
        ax.set_ylabel("balanced accuracy")
    axs[0].set_ylabel("accuracy")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, F2 / "fig7_compressive.png")

    e8 = pd.read_csv(R2 / "e8_hybrid.csv")
    variants = ["DP-SGD (all features)", "PCA-5 + DP-SGD", "D-DCA-5 + DP-SGD (proposed)", "D-DCA-10 + DP-SGD"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    for v in variants:
        a = agg(e8[e8.variant == v], ["target_eps"], ["accuracy", "f1"])
        fin = a[np.isfinite(a.target_eps)]
        line(axs[0], fin.target_eps, fin.accuracy, fin.accuracy_std, ENT[v], v)
        line(axs[1], fin.target_eps, fin.f1, fin.f1_std, ENT[v], v)
    for ax, t in zip(axs, ["test accuracy", "test F1"]):
        eps_axis(ax, EPS_T)
        ax.set_xlabel("privacy budget ε (δ = 1e-5)")
        ax.set_ylabel(t)
    axs[0].set_title("Hybrid compressive + DP-SGD")
    axs[1].set_title("F1 (minority class >50K)")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, F2 / "fig8_hybrid.png")

    # Central vs local vs hybrid on one ε axis (all with formal guarantees)
    fig, ax = plt.subplots(figsize=(6, 3))
    a = agg(e2[e2.regime == "full"], ["target_eps"], ["accuracy"])
    a = a[np.isfinite(a.target_eps)]
    line(ax, a.target_eps, a.accuracy, None, C[0], "central DP (DP-SGD)")
    a = agg(e8[e8.variant == "D-DCA-5 + DP-SGD (proposed)"], ["target_eps"], ["accuracy"])
    a = a[np.isfinite(a.target_eps)]
    line(ax, a.target_eps, a.accuracy, None, C[3], "hybrid D-DCA-5 + DP-SGD")
    a = agg(e5[e5.variant == "full-record LDP"], ["eps_total"], ["accuracy"])
    line(ax, a.eps_total, a.accuracy, None, C[2], "local DP (full record)")
    majority = 1 - pd.read_csv(ROOT / "data" / "processed" / "adult_test_clean.csv")["income"].mean()
    ax.axhline(majority, color=INK2, ls="--", lw=1)
    ax.text(0.05, majority + 0.003, "majority-class rate", color=INK2, fontsize=8, va="bottom")
    eps_axis(ax, [0.05, 0.25, 1, 4, 16, 64])
    ax.set_xlabel("ε (per record)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Same ε, different trust model")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, F2 / "fig9_trust_models.png")


if __name__ == "__main__":
    ia1()
    if (R2 / "e8_hybrid.csv").exists():
        ia2()
