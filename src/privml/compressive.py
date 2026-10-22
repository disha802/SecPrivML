"""
Module 5 — Compressive Privacy.

Compressive privacy (Kung, 2017) releases a *lossy, low-dimensional*
projection Z = XW of the data instead of X itself. The projection is chosen so
that the intended utility task stays solvable from Z, while other information
(here: the sensitive attribute ``sex`` and the exact record values) is
discarded at the source. All projections are fitted on the public split only.

Projections
-----------
PCA     unsupervised: maximise retained variance.                           (baseline)
DCA     Discriminant Component Analysis (Kung): utility-supervised,
        eigenvectors of (S_W + ρI)^{-1}(S_B^u + ρ'I).                          (utility-aware)
D-DCA   Desensitised DCA (proposed in this work, inspired by Kung's DUCA):
        generalised eigenproblem
            N w = λ D w,   N = S_B^u + β Σ,   D = γ S_B^p + Σ + ρI
        where S_B^u / S_B^p are the between-class scatters of the utility label
        (income) and the privacy label, and Σ is the total covariance.
        The numerator rewards utility-discriminant power (plus β·variance so
        that m > 1 informative directions exist for a binary label); the
        denominator penalises directions that separate the privacy classes.
        γ controls the privacy–utility trade-off (γ = 0 reduces to a
        variance-regularised DCA).

Design decision (IA-2): D-DCA is applied *after suppressing* the explicit
sensitive columns (sex=*). A linear projection of a binary one-hot column
leaves a bimodal component that a non-linear attacker can still decode, so
suppression removes the attribute itself and D-DCA removes its linear
proxies (relationship, occupation, hours-per-week, ...). A joint
(sex × relationship) privacy label was also evaluated (``privacy_classes``);
it removes more proxy signal but costs more utility, because relationship is
also strongly predictive of income.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _between_scatter(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    mu = X.mean(0)
    S = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(labels):
        Xc = X[labels == c]
        d = (Xc.mean(0) - mu)[:, None]
        S += len(Xc) / len(X) * (d @ d.T)
    return S


def _within_scatter(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    S = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(labels):
        Xc = X[labels == c]
        S += len(Xc) / len(X) * np.cov(Xc, rowvar=False, bias=True)
    return S


class Projection:
    """Common interface: fit on public data, transform anything, reconstruct."""

    def __init__(self, m: int):
        self.m = m
        self.mu_: np.ndarray | None = None
        self.W_: np.ndarray | None = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) @ self.W_

    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
        """Least-squares reconstruction X̂ = Z W⁺ + μ (best linear inversion)."""
        return Z @ np.linalg.pinv(self.W_) + self.mu_


class PCA(Projection):
    def fit(self, X, y=None, s=None):
        self.mu_ = X.mean(0)
        _, vecs = eigh(np.cov(X - self.mu_, rowvar=False))
        self.W_ = vecs[:, ::-1][:, :self.m]
        return self


class DCA(Projection):
    def __init__(self, m: int, rho: float = 1e-2, rho_prime: float = 1e-3):
        super().__init__(m)
        self.rho, self.rho_prime = rho, rho_prime

    def fit(self, X, y, s=None):
        self.mu_ = X.mean(0)
        d = X.shape[1]
        N = _between_scatter(X, y) + self.rho_prime * np.eye(d)
        D = _within_scatter(X, y) + self.rho * np.eye(d)
        _, vecs = eigh(N, D)
        self.W_ = vecs[:, ::-1][:, :self.m]
        return self


class DDCA(Projection):
    """Desensitised DCA (proposed). ``s`` is the privacy label per record."""

    def __init__(self, m: int, gamma: float = 50.0, beta: float = 0.1, rho: float = 1e-2):
        super().__init__(m)
        self.gamma, self.beta, self.rho = gamma, beta, rho

    def fit(self, X, y, s):
        self.mu_ = X.mean(0)
        d = X.shape[1]
        cov = np.cov(X - self.mu_, rowvar=False, bias=True)
        N = _between_scatter(X, y) + self.beta * cov
        D = self.gamma * _between_scatter(X, s) + cov + self.rho * np.eye(d)
        _, vecs = eigh(N, D)
        self.W_ = vecs[:, ::-1][:, :self.m]
        return self


def non_sensitive_columns(feature_names: list[str], attr: str = "sex") -> np.ndarray:
    """Indices of encoded features that are not one-hot columns of ``attr``."""
    return np.array([i for i, n in enumerate(feature_names) if not n.startswith(f"{attr}=")])


def privacy_classes(df) -> np.ndarray:
    """Joint (sex × relationship) privacy label used by D-DCA."""
    return (df["sex"].astype(str) + "|" + df["relationship"].astype(str)).to_numpy()


def reconstruction_error(proj: Projection, X: np.ndarray) -> float:
    """Mean squared error of the best linear reconstruction, normalised by the
    data variance (0 = perfect reconstruction, 1 = no better than the mean)."""
    Xh = proj.reconstruct(proj.transform(X))
    return float(((X - Xh) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
