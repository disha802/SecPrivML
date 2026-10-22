"""Correctness checks for Module 3 (local DP)."""

import numpy as np
import pytest

from src.privml import ldp

DOMAIN = ["a", "b", "c", "d", "e"]
TRUE_F = np.array([0.4, 0.3, 0.15, 0.1, 0.05])


def _sample(n, rng):
    return rng.choice(DOMAIN, size=n, p=TRUE_F)


def test_grr_keep_probability():
    rng = np.random.default_rng(0)
    vals = np.array(["a"] * 100_000)
    eps = 1.0
    out = ldp.grr_perturb(vals, DOMAIN, eps, rng)
    p = np.exp(eps) / (np.exp(eps) + len(DOMAIN) - 1)
    assert (out == "a").mean() == pytest.approx(p, abs=0.005)
    assert set(out) <= set(DOMAIN)


@pytest.mark.parametrize("eps", [0.5, 2.0])
def test_grr_estimator_unbiased(eps):
    rng = np.random.default_rng(1)
    vals = _sample(20_000, rng)
    emp = np.array([(vals == v).mean() for v in DOMAIN])
    est = np.mean([ldp.grr_estimate(ldp.grr_perturb(vals, DOMAIN, eps, rng), DOMAIN, eps)
                   for _ in range(60)], axis=0)
    assert np.allclose(est, emp, atol=0.01)


@pytest.mark.parametrize("eps", [0.5, 2.0])
def test_oue_estimator_unbiased(eps):
    rng = np.random.default_rng(2)
    vals = _sample(20_000, rng)
    emp = np.array([(vals == v).mean() for v in DOMAIN])
    est = np.mean([ldp.oue_estimate(ldp.oue_perturb(vals, DOMAIN, eps, rng), eps)
                   for _ in range(60)], axis=0)
    assert np.allclose(est, emp, atol=0.01)


def test_piecewise_mechanism_unbiased_and_bounded():
    rng = np.random.default_rng(3)
    eps = 1.0
    C = (np.exp(eps / 2) + 1) / (np.exp(eps / 2) - 1)
    for t in (-0.8, 0.0, 0.5):
        out = ldp.pm_perturb(np.full(400_000, t), eps, rng)
        assert out.min() >= -C - 1e-9 and out.max() <= C + 1e-9
        assert out.mean() == pytest.approx(t, abs=0.02)


def test_unit_mapping_roundtrip():
    x = np.array([17.0, 40.0, 90.0])
    assert np.allclose(ldp.from_unit(ldp.to_unit(x, 17, 90), 17, 90), x)
