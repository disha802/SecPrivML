"""
Step 2 — IA-1 preliminary implementation and baseline evaluation.

Usage:  python scripts/02_run_ia1.py          (≈ 2 minutes on a laptop CPU)

Runs the preliminary subset of the study described in the IA-1 report:
  P1  non-private baselines (LR, RF, MLP) with utility + membership inference
  P2  privacy exposure of the raw data: QI uniqueness / re-identification risk,
      attribute inference of `sex` with and without the explicit column
  P3  first DP-SGD runs (ε ∈ {0.5, 1, 2, 4, 8})
  P4  central Laplace vs local GRR/OUE histogram release (occupation)
  P5  first Mondrian k-anonymity runs (k ∈ {5, 10})
  P6  PCA compression and attribute leakage (m ∈ {5, 10, 20})

Outputs: results/ia1/*.csv, results/ia1/run_log.txt
"""

from _common import ROOT, Run

import numpy as np

from src.privml import attacks as A
from src.privml import compressive as CP
from src.privml import data as D
from src.privml import experiments as E
from src.privml import ppdm

SEEDS = [0, 1, 2]


def main() -> None:
    run = Run(ROOT / "results" / "ia1")
    P = D.prepare()
    run.log(f"Data: public={len(P.X_pub)} private={len(P.X_tr)} test={len(P.X_te)} "
            f"d={P.X_tr.shape[1]}")

    rows = run.stage("P1 baselines", E.baselines, P, SEEDS)
    run.save("p1_baselines", rows)

    def exposure():
        out = [{"analysis": "reidentification (raw QIs)", **A.reidentification_risk(P.splits.train)}]
        sup = CP.non_sensitive_columns(P.encoder.feature_names)
        for kind in ("lr", "gbm"):
            out.append({"analysis": f"attribute inference, all features ({kind})",
                        **A.attribute_inference(P.X_pub, P.s_pub, P.X_te, P.s_te, kind)})
            out.append({"analysis": f"attribute inference, sex column dropped ({kind})",
                        **A.attribute_inference(P.X_pub[:, sup], P.s_pub, P.X_te[:, sup],
                                                P.s_te, kind)})
        return out
    run.save("p2_exposure", run.stage("P2 raw-data exposure", exposure))

    rows = run.stage("P3 DP-SGD (preliminary)", E.dpsgd_sweep, P, SEEDS,
                     eps_grid=[0.5, 1.0, 2.0, 4.0, 8.0], regimes=("full",))
    run.save("p3_dpsgd", rows)

    rows = run.stage("P4 central vs local histogram", E.marginal_release, P, reps=20,
                     attrs=("occupation",))
    run.save("p4_marginals", rows)

    def kanon():
        out = []
        for k in (5, 10):
            r = ppdm.anonymise(P.splits.train, k)
            out.append({"k": k, "n_classes": r["n_classes"], "ncp": r["ncp"],
                        "homogeneity_rate": r["homogeneity_rate"],
                        **A.reidentification_risk(r["train_view"])})
            if k == 10:
                r["display_view"].head(8).to_csv(run.out / "p5_kanon_sample_k10.csv", index=False)
        return out
    run.save("p5_kanon", run.stage("P5 k-anonymity (preliminary)", kanon))

    def pca():
        out = []
        for m in (5, 10, 20):
            proj = CP.PCA(m).fit(P.X_pub)
            Zp, Zte = proj.transform(P.X_pub), proj.transform(P.X_te)
            out.append({"m": m, "reconstruction_nmse": CP.reconstruction_error(proj, P.X_te),
                        **{f"lr_{k}": v for k, v in A.attribute_inference(Zp, P.s_pub, Zte, P.s_te, "lr").items()},
                        **{f"gbm_{k}": v for k, v in A.attribute_inference(Zp, P.s_pub, Zte, P.s_te, "gbm").items()}})
        return out
    run.save("p6_pca", run.stage("P6 PCA compression (preliminary)", pca))
    run.log("IA-1 run complete")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
