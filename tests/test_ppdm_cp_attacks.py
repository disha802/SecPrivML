"""Correctness checks for Module 4 (k-anonymity), Module 5 (projections) and Module 1 (attacks)."""

import numpy as np
import pandas as pd
import pytest

from src.privml import attacks as A
from src.privml import compressive as CP
from src.privml import ppdm


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame({
        "age": rng.integers(17, 90, n).astype(float),
        "education-num": rng.integers(1, 17, n).astype(float),
        "sex": rng.choice(["Male", "Female"], n),
        "race": rng.choice(["White", "Black", "Asian-Pac-Islander", "Other"], n, p=[.7, .15, .1, .05]),
        "marital-status": rng.choice(["Married-civ-spouse", "Never-married", "Divorced"], n),
        "native-country": rng.choice(["United-States", "Mexico", "India"], n, p=[.85, .1, .05]),
        "workclass": rng.choice(["Private", "Self-emp-not-inc", "Local-gov"], n),
        "income": rng.integers(0, 2, n),
    })


@pytest.mark.parametrize("k", [2, 5, 20])
def test_mondrian_is_k_anonymous_partition(table, k):
    classes = ppdm.mondrian(table, k)
    allidx = np.concatenate(classes)
    assert sorted(allidx) == list(range(len(table)))          # disjoint cover
    assert min(len(c) for c in classes) >= k
    res = ppdm.anonymise(table, k)
    risk = A.reidentification_risk(res["train_view"])
    assert risk["min_class_size"] >= k and risk["reid_max_risk"] <= 1 / k + 1e-12


def test_l_diversity_enforced(table):
    res = ppdm.anonymise(table, 5, l_div=2)
    assert res["homogeneity_rate"] == 0.0
    for c in res["classes"]:
        assert table["income"].to_numpy()[c].min() != table["income"].to_numpy()[c].max()


def test_ncp_bounds(table):
    assert ppdm.ncp(table, [np.arange(len(table))]) == pytest.approx(1.0)
    assert ppdm.ncp(table, [np.array([i]) for i in range(len(table))]) == 0.0


def test_raw_table_uniqueness(table):
    assert A.reidentification_risk(table)["unique_frac"] > 0.5


def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = rng.integers(0, 2, n)
    X = np.column_stack([y + 0.3 * rng.normal(size=n),            # utility signal
                         s + 0.3 * rng.normal(size=n),            # privacy signal
                         rng.normal(size=(n, 4))])
    return X, y, s


def test_pca_full_rank_reconstructs():
    X, _, _ = _toy()
    assert CP.reconstruction_error(CP.PCA(X.shape[1]).fit(X), X) == pytest.approx(0.0, abs=1e-9)


def test_ddca_removes_linear_privacy_signal():
    X, y, s = _toy()
    Z = CP.DDCA(2, gamma=100.0).fit(X, y, s).transform(X)
    corr_s = max(abs(np.corrcoef(Z[:, j], s)[0, 1]) for j in range(2))
    corr_y = max(abs(np.corrcoef(Z[:, j], y)[0, 1]) for j in range(2))
    assert corr_s < 0.05 and corr_y > 0.8


def test_mia_detects_memorisation():
    class Memoriser:                         # perfect on members, uninformed on others
        def __init__(self, X, y):
            self.lookup = {tuple(r): v for r, v in zip(X, y)}

        def predict_proba(self, X):
            p = np.array([self.lookup.get(tuple(r), 0.5) for r in X], dtype=float)
            p = np.clip(p, 0.01, 0.99)
            return np.column_stack([1 - p, p])

    rng = np.random.default_rng(0)
    Xi, Xo = rng.normal(size=(300, 3)), rng.normal(size=(300, 3))
    yi, yo = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    res = A.membership_inference(Memoriser(Xi, yi), Xi, yi, Xo, yo, n=300)
    assert res["mia_auc"] > 0.95 and res["mia_advantage"] > 0.9
