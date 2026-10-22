"""
Experiment definitions shared by the IA-1 and IA-2 runner scripts.

Every function returns a list of flat dict rows (one per configuration and
seed) that the runner writes to CSV. The same utility model (the MLP from
``models.py``) and the same attack suite (``attacks.py``) are used throughout,
which is what makes results comparable across modules.

Experiment map
--------------
E1  baselines           non-private LR / RF / MLP, full and small-data regimes  (M1)
E2  dpsgd_sweep         DP-SGD across ε, full and small-data regimes             (M2)
E3  dpsgd_ablation      clip norm, learning rate and epochs at fixed ε           (M2)
E4  marginal_release    central Laplace vs local GRR / OUE frequency estimation  (M2, M3)
E5  ldp_training        train on LDP-collected records across ε_total           (M3)
E6  kanon_sweep         Mondrian k-anonymity (± l-diversity) across k            (M4)
E7  compressive_sweep   PCA / DCA / D-DCA across target dimension m, and γ       (M5)
E8  hybrid_cp_dp        D-DCA compression followed by DP-SGD across ε            (M5 + M2)
"""

from __future__ import annotations

import time

import numpy as np

from . import attacks as A
from . import compressive as CP
from . import data as D
from . import dp, ldp, ppdm
from . import models as M
from .metrics import utility

EPS_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
MLP_CFG = dict(epochs=30, hidden=64, lr=1e-3, batch=128)          # shared utility model
DPSGD_CFG = dict(epochs=20, batch=256, clip=1.0, lr=0.05, hidden=64)
SMALL_N = 1000
SMALL_MLP_CFG = dict(epochs=200, hidden=256, lr=1e-3, batch=64)  # high-capacity regime
SMALL_DPSGD_CFG = dict(epochs=50, batch=64, clip=1.0, lr=0.05, hidden=256)


def _evaluate(model, X_in, y_in, X_te, y_te, seed, n_mia=5000) -> dict:
    p = model.predict_proba(X_te)[:, 1]
    return {**utility(y_te, p), **A.membership_inference(model, X_in, y_in, X_te, y_te,
                                                         n=n_mia, seed=seed)}


def _small(P: D.Prepared, seed: int):
    idx = np.random.default_rng(1000 + seed).choice(len(P.X_tr), SMALL_N, replace=False)
    return P.X_tr[idx], P.y_tr[idx]


def _standardise(Z_fit, *Zs):
    mu, sd = Z_fit.mean(0), Z_fit.std(0) + 1e-12
    return [(Z - mu) / sd for Z in Zs]


# --------------------------------------------------------------------------- E1
def baselines(P: D.Prepared, seeds) -> list[dict]:
    rows = []
    for seed in seeds:
        for regime in ("full", "small"):
            Xtr, ytr = (P.X_tr, P.y_tr) if regime == "full" else _small(P, seed)
            cfg = MLP_CFG if regime == "full" else SMALL_MLP_CFG
            n_mia = 5000 if regime == "full" else SMALL_N
            for name, fit in (("LogisticRegression", lambda: M.train_lr(Xtr, ytr, seed)),
                              ("RandomForest", lambda: M.train_rf(Xtr, ytr, seed)),
                              ("MLP", lambda: M.train_mlp(Xtr, ytr, seed=seed, **cfg))):
                t = time.time()
                model = fit()
                rows.append({"experiment": "E1", "model": name, "regime": regime, "seed": seed,
                             "n_train": len(Xtr), "fit_seconds": time.time() - t,
                             **_evaluate(model, Xtr, ytr, P.X_te, P.y_te, seed, n_mia)})
    return rows


# --------------------------------------------------------------------------- E2
def dpsgd_sweep(P: D.Prepared, seeds, eps_grid=EPS_GRID, regimes=("full", "small"),
                delta: float = 1e-5) -> list[dict]:
    rows = []
    for regime in regimes:
        cfg = DPSGD_CFG if regime == "full" else SMALL_DPSGD_CFG
        for seed in seeds:
            Xtr, ytr = (P.X_tr, P.y_tr) if regime == "full" else _small(P, seed)
            d = delta if regime == "full" else 1e-4       # δ < 1/n in both regimes
            n_mia = 5000 if regime == "full" else SMALL_N
            for eps in list(eps_grid) + [None]:
                t = time.time()
                model, info = dp.train_dpsgd(Xtr, ytr, eps, delta=d, seed=seed, **cfg)
                rows.append({"experiment": "E2", "regime": regime, "seed": seed,
                             "target_eps": eps if eps is not None else np.inf,
                             "fit_seconds": time.time() - t, **info,
                             **_evaluate(model, Xtr, ytr, P.X_te, P.y_te, seed, n_mia)})
    return rows


# --------------------------------------------------------------------------- E3
def dpsgd_ablation(P: D.Prepared, seeds, eps: float = 1.0) -> list[dict]:
    grid = ([("clip", c) for c in (0.1, 0.5, 1.0, 2.0, 5.0)]
            + [("lr", v) for v in (0.01, 0.05, 0.1, 0.2, 0.5)]
            + [("epochs", v) for v in (5, 10, 20, 40)])
    rows = []
    for seed in seeds:
        for param, val in grid:
            cfg = dict(DPSGD_CFG, **{param: val})
            model, info = dp.train_dpsgd(P.X_tr, P.y_tr, eps, seed=seed, **cfg)
            p = model.predict_proba(P.X_te)[:, 1]
            rows.append({"experiment": "E3", "param": param, "value": val, "seed": seed,
                         **info, **utility(P.y_te, p)})
    return rows


# --------------------------------------------------------------------------- E4
def marginal_release(P: D.Prepared, reps: int = 50,
                     eps_grid=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
                     attrs=("workclass", "marital-status", "occupation", "native-country")) -> list[dict]:
    rows = []
    train = P.splits.train
    for attr in attrs:
        dom = P.encoder.domains[attr]
        vals = train[attr].to_numpy()
        truth = np.array([(vals == v).mean() for v in dom])
        for eps in eps_grid:
            for r in range(reps):
                rng = np.random.default_rng(r)
                est = {
                    "Central-Laplace": dp.dp_histogram(vals, dom, eps, rng),
                    "Local-GRR": ldp.grr_estimate(ldp.grr_perturb(vals, dom, eps, rng), dom, eps),
                    "Local-OUE": ldp.oue_estimate(ldp.oue_perturb(vals, dom, eps, rng), eps),
                }
                for mech, f in est.items():
                    rows.append({"experiment": "E4", "attribute": attr, "k": len(dom),
                                 "eps": eps, "rep": r, "mechanism": mech,
                                 "l1_error": float(np.abs(f - truth).sum()),
                                 "mse": float(((f - truth) ** 2).mean())})
    return rows


# --------------------------------------------------------------------------- E5
def ldp_training(P: D.Prepared, seeds, eps_grid=(1, 2, 4, 8, 16, 32, 64, 128)) -> list[dict]:
    rows = []
    sp, dom = P.splits, P.encoder.domains
    for seed in seeds:
        for eps in eps_grid:
            for perturb_label in (True, False):
                rng = np.random.default_rng(seed)
                rel = ldp.perturb_dataset(sp.train, eps, dom, rng, perturb_label)
                pub = ldp.perturb_dataset(sp.public, eps, dom, rng, perturb_label)
                X_rel = P.encoder.transform(rel)
                model = M.train_mlp(X_rel, rel[D.TARGET].to_numpy(), seed=seed, **MLP_CFG)
                row = {"experiment": "E5", "eps_total": eps, "seed": seed,
                       "variant": "full-record LDP" if perturb_label else "features-only LDP",
                       "eps_per_attribute": rel.attrs["eps_per_attribute"],
                       **_evaluate(model, P.X_tr, P.y_tr, P.X_te, P.y_te, seed)}
                if perturb_label:   # attacker simulates the mechanism on public data
                    row.update(A.attribute_inference(P.encoder.transform(pub), P.s_pub,
                                                     X_rel, P.s_tr, "gbm", seed))
                rows.append(row)
    return rows


# --------------------------------------------------------------------------- E6
def kanon_sweep(P: D.Prepared, seeds, ks=(2, 5, 10, 25, 50, 100, 250),
                ls=(None, 2)) -> list[dict]:
    rows = []
    sp = P.splits
    raw = A.reidentification_risk(sp.train)
    rows.append({"experiment": "E6", "k": 1, "l": "none", "seed": -1, **raw,
                 "ncp": 0.0, **ppdm.sensitive_disclosure(sp.train, [np.arange(len(sp.train))])})
    for l_div in ls:
        for k in ks:
            t = time.time()
            res = ppdm.anonymise(sp.train, k, l_div)
            res_pub = ppdm.anonymise(sp.public, k, l_div)
            X_rel = P.encoder.transform(res["train_view"])
            X_pub = P.encoder.transform(res_pub["train_view"])
            base = {"experiment": "E6", "k": k, "l": l_div or "none",
                    "anon_seconds": time.time() - t, "n_classes_anon": res["n_classes"],
                    "avg_class": res["avg_class"], "ncp": res["ncp"],
                    "homogeneity_rate": res["homogeneity_rate"],
                    "t_closeness": res["t_closeness"],
                    **A.reidentification_risk(res["train_view"]),
                    **A.attribute_inference(X_pub, P.s_pub, X_rel, P.s_tr, "gbm")}
            for seed in seeds:
                model = M.train_mlp(X_rel, P.y_tr, seed=seed, **MLP_CFG)
                rows.append({**base, "seed": seed,
                             **_evaluate(model, P.X_tr, P.y_tr, P.X_te, P.y_te, seed)})
    return rows


# --------------------------------------------------------------------------- E7
def _projections(m: int, gamma: float = 10.0):
    return {
        "PCA (all features)": (CP.PCA(m), False, None),
        "PCA (sex suppressed)": (CP.PCA(m), True, None),
        "DCA (sex suppressed)": (CP.DCA(m), True, None),
        "D-DCA (proposed)": (CP.DDCA(m, gamma=gamma), True, "sex"),
        "D-DCA (joint label)": (CP.DDCA(m, gamma=gamma), True, "joint"),
    }


def _fit_projection(P: D.Prepared, proj, suppress: bool, label):
    cols = (CP.non_sensitive_columns(P.encoder.feature_names) if suppress
            else np.arange(P.X_tr.shape[1]))
    s = {None: None, "sex": P.splits.public["sex"].to_numpy(),
         "joint": CP.privacy_classes(P.splits.public)}[label]
    proj.fit(P.X_pub[:, cols], P.y_pub, s)
    Zp, Zt, Zte = (proj.transform(X[:, cols]) for X in (P.X_pub, P.X_tr, P.X_te))
    return cols, Zp, Zt, Zte


def _cp_row(P, name, m, proj, cols, Zp, Zt, Zte, seeds):
    priv = {f"lr_{k}": v for k, v in A.attribute_inference(Zp, P.s_pub, Zte, P.s_te, "lr").items()}
    priv.update({f"gbm_{k}": v for k, v in
                 A.attribute_inference(Zp, P.s_pub, Zte, P.s_te, "gbm").items()})
    rec = CP.reconstruction_error(proj, P.X_te[:, cols]) if proj is not None else 0.0
    Zp_s, Zt_s, Zte_s = _standardise(Zp, Zp, Zt, Zte)
    rows = []
    for seed in seeds:
        model = M.train_mlp(Zt_s, P.y_tr, seed=seed, **MLP_CFG)
        rows.append({"experiment": "E7", "projection": name, "m": m, "seed": seed,
                     "reconstruction_nmse": rec, **priv,
                     **_evaluate(model, Zt_s, P.y_tr, Zte_s, P.y_te, seed)})
    return rows


def compressive_sweep(P: D.Prepared, seeds, ms=(2, 3, 5, 10, 20, 40),
                      gammas=(0.0, 1.0, 10.0, 100.0, 1000.0)) -> list[dict]:
    rows = []
    sup = CP.non_sensitive_columns(P.encoder.feature_names)
    for name, cols in (("Raw (all features)", np.arange(P.X_tr.shape[1])),
                       ("Raw (sex suppressed)", sup)):
        rows += _cp_row(P, name, len(cols), None, cols, P.X_pub[:, cols], P.X_tr[:, cols],
                        P.X_te[:, cols], seeds)
    for m in ms:
        for name, (proj, suppress, label) in _projections(m).items():
            cols, Zp, Zt, Zte = _fit_projection(P, proj, suppress, label)
            rows += _cp_row(P, name, m, proj, cols, Zp, Zt, Zte, seeds)
    for g in gammas:                                   # γ sweep for D-DCA at m = 5
        proj = CP.DDCA(5, gamma=g)
        cols, Zp, Zt, Zte = _fit_projection(P, proj, True, "sex")
        for r in _cp_row(P, f"D-DCA gamma={g:g}", 5, proj, cols, Zp, Zt, Zte, seeds):
            rows.append({**r, "experiment": "E7-gamma", "gamma": g})
    return rows


# --------------------------------------------------------------------------- E8
def hybrid_cp_dp(P: D.Prepared, seeds, eps_grid=EPS_GRID, delta: float = 1e-5) -> list[dict]:
    variants = {"DP-SGD (all features)": None,
                "PCA-5 + DP-SGD": (CP.PCA(5), True, None),
                "D-DCA-5 + DP-SGD (proposed)": (CP.DDCA(5, gamma=10.0), True, "sex"),
                "D-DCA-10 + DP-SGD": (CP.DDCA(10, gamma=10.0), True, "sex")}
    rows = []
    for name, spec in variants.items():
        if spec is None:
            Ztr, Zte, priv = P.X_tr, P.X_te, {}
        else:
            proj, suppress, label = spec
            _, Zp, Zt, Zt_e = _fit_projection(P, proj, suppress, label)
            priv = A.attribute_inference(Zp, P.s_pub, Zt_e, P.s_te, "gbm")
            _, Ztr, Zte = _standardise(Zp, Zp, Zt, Zt_e)
        for seed in seeds:
            for eps in list(eps_grid) + [None]:
                model, info = dp.train_dpsgd(Ztr, P.y_tr, eps, delta=delta, seed=seed, **DPSGD_CFG)
                rows.append({"experiment": "E8", "variant": name, "seed": seed,
                             "target_eps": eps if eps is not None else np.inf, **info, **priv,
                             **_evaluate(model, Ztr, P.y_tr, Zte, P.y_te, seed)})
    return rows
