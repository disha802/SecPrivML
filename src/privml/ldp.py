"""
Module 3 — Local Differential Privacy.

Each individual perturbs their own record *before* it leaves their device, so
the data collector never sees true values (no trusted curator is needed).

Mechanisms
----------
GRR   Generalised (k-ary) Randomised Response  (Kairouz et al. 2016)
      report true v with p = e^ε / (e^ε + k - 1), otherwise a uniformly random
      other value. Frequency estimator: f̂_v = (ĉ_v/n - q) / (p - q), q = (1-p)/(k-1).
OUE   Optimised Unary Encoding (Wang et al., USENIX Security 2017)
      one-hot encode, keep the true bit at 1 w.p. 1/2, flip every 0-bit to 1
      w.p. 1/(e^ε + 1). Better than GRR when k > 3e^ε + 2.
PM    Piecewise Mechanism for numeric values in [-1, 1] (Wang et al., ICDE 2019)
      unbiased, bounded output in [-C, C], C = (e^{ε/2} + 1) / (e^{ε/2} - 1).

``perturb_dataset`` applies these attribute-wise to a full record, splitting
the per-user budget ε_total equally across all d attributes (sequential
composition), including the label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL, NUMERIC, TARGET

# Public attribute bounds from adult.names / census definitions (not data-dependent)
NUMERIC_BOUNDS = {
    "age": (17, 90), "education-num": (1, 16), "capital-gain": (0, 99999),
    "capital-loss": (0, 4356), "hours-per-week": (1, 99),
}


# -------------------------------------------------------------------------- GRR
def grr_perturb(values: np.ndarray, domain: list, eps: float, rng) -> np.ndarray:
    k = len(domain)
    codes = pd.Categorical(values, categories=domain).codes
    p = np.exp(eps) / (np.exp(eps) + k - 1)
    keep = rng.random(len(codes)) < p
    other = rng.integers(0, k - 1, len(codes))
    other = other + (other >= codes)                 # uniform over the k-1 other values
    out = np.where(keep, codes, other)
    return np.asarray(domain, dtype=object)[out]


def grr_estimate(reports: np.ndarray, domain: list, eps: float) -> np.ndarray:
    k, n = len(domain), len(reports)
    p = np.exp(eps) / (np.exp(eps) + k - 1)
    q = (1 - p) / (k - 1)
    counts = np.array([(reports == v).sum() for v in domain], dtype=float)
    return (counts / n - q) / (p - q)


# -------------------------------------------------------------------------- OUE
def oue_perturb(values: np.ndarray, domain: list, eps: float, rng) -> np.ndarray:
    codes = pd.Categorical(values, categories=domain).codes
    n, k = len(codes), len(domain)
    q = 1.0 / (np.exp(eps) + 1)
    bits = rng.random((n, k)) < q
    bits[np.arange(n), codes] = rng.random(n) < 0.5
    return bits


def oue_estimate(bits: np.ndarray, eps: float) -> np.ndarray:
    p, q = 0.5, 1.0 / (np.exp(eps) + 1)
    return (bits.mean(0) - q) / (p - q)


# --------------------------------------------------------------------------- PM
def pm_perturb(t: np.ndarray, eps: float, rng) -> np.ndarray:
    """Piecewise mechanism; input t must lie in [-1, 1]."""
    e2 = np.exp(eps / 2)
    C = (e2 + 1) / (e2 - 1)
    left = (C + 1) / 2 * t - (C - 1) / 2
    right = left + C - 1
    centre = rng.random(len(t)) < e2 / (e2 + 1)
    in_band = left + rng.random(len(t)) * (C - 1)
    # outside the band: uniform on [-C, left) U (right, C], total length C + 1
    u = rng.random(len(t)) * (C + 1)
    len_left = left + C
    outside = np.where(u < len_left, -C + u, right + (u - len_left))
    return np.where(centre, in_band, outside)


def to_unit(x, lo, hi):
    return 2 * (np.clip(x, lo, hi) - lo) / (hi - lo) - 1


def from_unit(t, lo, hi):
    return (t + 1) / 2 * (hi - lo) + lo


# ----------------------------------------------------------- record perturbation
def perturb_dataset(df: pd.DataFrame, eps_total: float, domains: dict, rng,
                    perturb_label: bool = True) -> pd.DataFrame:
    """ε_total-LDP release of every record (attribute-wise budget split)."""
    attrs = NUMERIC + CATEGORICAL + ([TARGET] if perturb_label else [])
    eps = eps_total / len(attrs)
    out = df.copy()
    for c in NUMERIC:
        lo, hi = NUMERIC_BOUNDS[c]
        out[c] = from_unit(pm_perturb(to_unit(df[c].to_numpy(float), lo, hi), eps, rng), lo, hi)
    for c in CATEGORICAL:
        out[c] = grr_perturb(df[c].to_numpy(), domains[c], eps, rng)
    if perturb_label:
        out[TARGET] = grr_perturb(df[TARGET].to_numpy(), [0, 1], eps, rng).astype(int)
    out.attrs["eps_per_attribute"] = eps
    return out
