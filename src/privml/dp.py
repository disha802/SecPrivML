"""
Module 2 — Central Differential Privacy.

Contents
--------
laplace_mechanism, gaussian_mechanism   output perturbation for numeric queries
dp_histogram                             ε-DP frequency release (Laplace)
rdp_sampled_gaussian, rdp_to_dp          Rényi-DP accountant for DP-SGD
compute_epsilon, calibrate_sigma         ε for a given noise level, and its inverse
train_dpsgd                              DP-SGD (Abadi et al., CCS 2016) with
                                         Poisson sub-sampling, per-sample clipping
                                         and Gaussian noise

The accountant is implemented from first principles:

* RDP of the Poisson-subsampled Gaussian mechanism at integer order α
  (Mironov, Talwar & Zhang 2019, Thm. 9 / Sec. 3.3):
      A_α = Σ_{k=0..α} C(α,k) (1-q)^{α-k} q^k exp((k²-k) / (2σ²))
      RDP(α) = log(A_α) / (α-1)
* Composition over T steps:  T · RDP(α)
* Conversion to (ε, δ)-DP (Balle et al. 2020, Thm. 21, as used by Opacus):
      ε = RDP_T(α) + log((α-1)/α) - (log δ + log α) / (α-1),  minimised over α
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

from .models import MLP, SGDMomentum

ORDERS = np.arange(2, 256)


# ------------------------------------------------------------- basic mechanisms
def laplace_mechanism(value, sensitivity: float, epsilon: float, rng=None):
    rng = rng or np.random.default_rng()
    return value + rng.laplace(0.0, sensitivity / epsilon, size=np.shape(value))


def gaussian_mechanism(value, l2_sensitivity: float, epsilon: float, delta: float, rng=None):
    """Classic Gaussian mechanism (Dwork & Roth 2014, Thm. A.1), valid for ε < 1."""
    rng = rng or np.random.default_rng()
    sigma = l2_sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    return value + rng.normal(0.0, sigma, size=np.shape(value))


def dp_histogram(values: np.ndarray, domain: list, epsilon: float, rng=None) -> np.ndarray:
    """ε-DP relative-frequency estimate. Under add/remove-one neighbouring, a
    histogram has L1 sensitivity 1, so Laplace(1/ε) noise per bin suffices.
    Post-processing (clip at 0, renormalise) does not affect the guarantee."""
    counts = np.array([(values == v).sum() for v in domain], dtype=float)
    noisy = np.clip(laplace_mechanism(counts, 1.0, epsilon, rng), 0, None)
    return noisy / noisy.sum() if noisy.sum() > 0 else np.full(len(domain), 1 / len(domain))


# ------------------------------------------------------------------- accountant
def rdp_sampled_gaussian(q: float, sigma: float, orders=ORDERS) -> np.ndarray:
    """RDP ε(α) of one step of the Poisson-subsampled Gaussian mechanism."""
    orders = np.asarray(orders)
    if sigma == 0:
        return np.full(len(orders), np.inf)
    if q == 1.0:
        return orders / (2 * sigma ** 2)
    out = np.empty(len(orders))
    for j, a in enumerate(orders):
        k = np.arange(a + 1)
        log_binom = gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1)
        terms = log_binom + (a - k) * np.log1p(-q) + k * np.log(q) + (k * k - k) / (2 * sigma ** 2)
        out[j] = logsumexp(terms) / (a - 1)
    return out


def rdp_to_dp(rdp: np.ndarray, delta: float, orders=ORDERS) -> tuple[float, int]:
    orders = np.asarray(orders, dtype=float)
    eps = rdp + np.log((orders - 1) / orders) - (np.log(delta) + np.log(orders)) / (orders - 1)
    j = int(np.nanargmin(eps))
    return float(max(eps[j], 0.0)), int(orders[j])


def compute_epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    return rdp_to_dp(steps * rdp_sampled_gaussian(q, sigma), delta)[0]


def calibrate_sigma(target_eps: float, delta: float, q: float, steps: int,
                    lo: float = 0.3, hi: float = 50.0, tol: float = 1e-3) -> float:
    """Smallest noise multiplier σ whose accounted ε does not exceed target_eps."""
    while compute_epsilon(q, hi, steps, delta) > target_eps:
        lo, hi = hi, hi * 2
        if hi > 1e5:
            raise ValueError(f"target ε={target_eps} unreachable")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if compute_epsilon(q, mid, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------- DP-SGD
def train_dpsgd(X: np.ndarray, y: np.ndarray, epsilon: float | None, delta: float = 1e-5,
                epochs: int = 20, batch: int = 256, clip: float = 1.0, lr: float = 0.05,
                momentum: float = 0.9, hidden: int = 64, seed: int = 0) -> tuple[MLP, dict]:
    """Train the shared MLP with DP-SGD.

    Each step: (1) Poisson-sample every record independently with rate q,
    (2) compute per-example gradients and clip each to L2 norm ≤ C,
    (3) sum, add N(0, σ²C²I), divide by the expected batch size qN,
    (4) take an SGD-with-momentum step (post-processing, so still private).

    ``epsilon=None`` gives clipped-but-noiseless SGD (an ablation isolating the
    effect of clipping from the effect of noise). Returns the model and an
    info dict with σ, the accounted ε and the number of steps.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(X)
    q = batch / n
    steps = int(round(epochs / q))
    sigma = 0.0 if epsilon is None else calibrate_sigma(epsilon, delta, q, steps)

    model = MLP(X.shape[1], hidden, seed)
    opt = SGDMomentum(model.params, lr=lr, momentum=momentum)

    for _ in range(steps):
        idx = np.flatnonzero(rng.random(n) < q)                     # Poisson sampling
        if len(idx):
            e, d, h = model.backward(X[idx], y[idx])
            norms = MLP.per_example_norms(X[idx], e, d, h)
            c = np.minimum(1.0, clip / (norms + 1e-12))                # per-example clipping
            g = MLP.weighted_grad_sum(X[idx], e, d, h, c)
        else:
            g = {k: np.zeros_like(v) for k, v in model.params.items()}
        for k in g:
            g[k] = (g[k] + rng.normal(0.0, sigma * clip, size=g[k].shape)) / (q * n)
        opt.step(g)

    eps_spent = math.inf if epsilon is None else compute_epsilon(q, sigma, steps, delta)
    return model, {"sigma": sigma, "epsilon": eps_spent, "delta": delta,
                   "steps": steps, "q": q, "clip": clip}
