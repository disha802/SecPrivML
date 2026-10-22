"""Correctness checks for Module 2 (central DP)."""

import math

import numpy as np
import pytest

from src.privml import dp
from src.privml.models import MLP

# Reference values produced by Opacus 1.5.2 (opacus.accountants.analysis.rdp)
# with the same integer orders 2..255 and δ = 1e-5.
OPACUS_REFERENCE = [
    (256 / 27146, 1.1, 2121, 2.3064),
    (0.01, 4.0, 10000, 1.0355),
    (256 / 60000, 1.1, 14062, 2.5970),
    (0.05, 0.8, 500, 14.6824),
]


@pytest.mark.parametrize("q,sigma,steps,expected", OPACUS_REFERENCE)
def test_accountant_matches_opacus(q, sigma, steps, expected):
    assert dp.compute_epsilon(q, sigma, steps, 1e-5) == pytest.approx(expected, abs=1e-3)


def test_full_batch_gaussian_rdp_closed_form():
    # Without sub-sampling the Gaussian mechanism has RDP α / (2σ²).
    assert np.allclose(dp.rdp_sampled_gaussian(1.0, 2.0, [2, 8]), [2 / 8, 8 / 8])


def test_epsilon_decreases_with_noise():
    eps = [dp.compute_epsilon(0.01, s, 1000, 1e-5) for s in (0.6, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(eps, eps[1:]))


def test_calibrate_sigma_hits_target():
    q, T = 0.01, 2000
    for target in (0.5, 1.0, 4.0):
        s = dp.calibrate_sigma(target, 1e-5, q, T)
        assert dp.compute_epsilon(q, s, T, 1e-5) <= target
        assert dp.compute_epsilon(q, s * 0.98, T, 1e-5) > target   # near-minimal


def test_laplace_noise_scale():
    rng = np.random.default_rng(0)
    noise = dp.laplace_mechanism(np.zeros(200_000), sensitivity=1.0, epsilon=0.5, rng=rng)
    assert np.mean(np.abs(noise)) == pytest.approx(2.0, rel=0.02)    # E|Lap(b)| = b


def test_dp_histogram_is_distribution():
    rng = np.random.default_rng(0)
    vals = np.array(list("aabbbc"))
    f = dp.dp_histogram(vals, ["a", "b", "c"], 1.0, rng)
    assert f.shape == (3,) and np.all(f >= 0) and f.sum() == pytest.approx(1.0)


def test_per_example_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 4))
    y = np.array([0, 1, 1, 0, 1], dtype=float)
    m = MLP(4, 3, seed=1)
    e, d, h = m.backward(X, y)

    def loss(i):
        p = m.predict_proba(X[i:i + 1])[0, 1]
        return -(y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p))

    for i in range(len(X)):
        onehot = np.zeros(len(X))
        onehot[i] = 1.0
        g = MLP.weighted_grad_sum(X, e, d, h, onehot)
        num = {}
        for k, v in m.params.items():
            num[k] = np.zeros_like(v)
            for idx in np.ndindex(v.shape):
                old = v[idx]
                v[idx] = old + 1e-6
                lp = loss(i)
                v[idx] = old - 1e-6
                lm = loss(i)
                v[idx] = old
                num[k][idx] = (lp - lm) / 2e-6
            assert np.allclose(g[k], num[k], atol=1e-5), k
        norm = math.sqrt(sum((num[k] ** 2).sum() for k in num))
        assert MLP.per_example_norms(X, e, d, h)[i] == pytest.approx(norm, rel=1e-4)


def test_dpsgd_runs_and_reports_budget():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6))
    y = (X[:, 0] > 0).astype(int)
    model, info = dp.train_dpsgd(X, y, epsilon=2.0, delta=1e-4, epochs=5, batch=50)
    assert info["epsilon"] <= 2.0 and info["sigma"] > 0
    acc = ((model.predict_proba(X)[:, 1] > 0.5) == y).mean()
    assert acc > 0.8
