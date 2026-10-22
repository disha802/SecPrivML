"""
Step 3 — IA-2 full experimental study (E1–E8).

Usage:  python scripts/03_run_ia2.py [--quick]

--quick uses one seed and is meant only for smoke-testing (≈ 3 min).
The full run uses 5 seeds for the model experiments (3 for the slower
LDP / k-anonymity / compressive sweeps) and takes ≈ 20–30 min on a laptop CPU.

Outputs: results/ia2/e*.csv, results/ia2/run_log.txt
"""

import argparse

from _common import ROOT, Run

import numpy as np

from src.privml import data as D
from src.privml import experiments as E


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", help="run a subset, e.g. --only e2 e8")
    args = ap.parse_args()

    seeds5 = [0] if args.quick else [0, 1, 2, 3, 4]
    seeds3 = [0] if args.quick else [0, 1, 2]
    run = Run(ROOT / "results" / ("ia2_quick" if args.quick else "ia2"))
    P = D.prepare()

    plan = {
        "e1": ("E1 non-private baselines", E.baselines, (P, seeds5)),
        "e2": ("E2 DP-SGD privacy budget sweep", E.dpsgd_sweep, (P, seeds5)),
        "e3": ("E3 DP-SGD hyper-parameter ablation", E.dpsgd_ablation, (P, seeds3)),
        "e4": ("E4 central vs local DP marginals", E.marginal_release, (P,)),
        "e5": ("E5 training on LDP-collected data", E.ldp_training, (P, seeds3)),
        "e6": ("E6 Mondrian k-anonymity sweep", E.kanon_sweep, (P, seeds3)),
        "e7": ("E7 compressive privacy sweep", E.compressive_sweep, (P, seeds3)),
        "e8": ("E8 hybrid compressive + DP-SGD", E.hybrid_cp_dp, (P, seeds5)),
    }
    names = {"e1": "e1_baselines", "e2": "e2_dpsgd", "e3": "e3_ablation", "e4": "e4_marginals",
             "e5": "e5_ldp_training", "e6": "e6_kanon", "e7": "e7_compressive", "e8": "e8_hybrid"}
    for key, (title, fn, fargs) in plan.items():
        if args.only and key not in args.only:
            continue
        run.save(names[key], run.stage(title, fn, *fargs))
    run.log("IA-2 run complete")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
