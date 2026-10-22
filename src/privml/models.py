"""
Non-private baseline models.

* Logistic regression  — a well-regularised linear reference point.
* Random forest        — fully grown trees, a deliberately high-capacity model
                         that memorises training points (worst case for MIA).
* MLP                  — one hidden ReLU layer, implemented in NumPy. The same
                         class is trained by DP-SGD (src/privml/dp.py), so the
                         only difference between the non-private and private MLP
                         is the training algorithm.

The MLP is written in NumPy (no deep-learning framework) because a one-hidden-
layer network admits closed-form per-example gradients, which DP-SGD needs:
for example i with input x_i, hidden pre-activation a_i, hidden output h_i and
output error e_i = p_i - y_i,

    ∂L_i/∂w2 = e_i h_i          ∂L_i/∂b2 = e_i
    δ_i      = e_i w2 ⊙ 1[a_i > 0]
    ∂L_i/∂W1 = x_i δ_iᵀ         ∂L_i/∂b1 = δ_i

so ‖∇L_i‖² = ‖x_i‖²‖δ_i‖² + ‖δ_i‖² + e_i²‖h_i‖² + e_i², and the clipped sum
Σ c_i x_i δ_iᵀ = Xᵀ(c ⊙ Δ) never materialises per-example weight matrices.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))       # numerically stable


class MLP:
    def __init__(self, d_in: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        b1, b2 = 1 / np.sqrt(d_in), 1 / np.sqrt(hidden)   # PyTorch-style uniform init
        self.params = {
            "W1": rng.uniform(-b1, b1, (d_in, hidden)), "b1": rng.uniform(-b1, b1, hidden),
            "w2": rng.uniform(-b2, b2, hidden), "b2": rng.uniform(-b2, b2, 1),
        }

    def _forward(self, X):
        P = self.params
        a = X @ P["W1"] + P["b1"]
        h = np.maximum(a, 0.0)
        return a, h, _sigmoid(h @ P["w2"] + P["b2"][0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._forward(np.asarray(X, dtype=np.float64))[2]
        return np.column_stack([1 - p, p])

    def backward(self, X, y):
        """Per-example error terms needed for gradients: (e, δ, h)."""
        a, h, p = self._forward(X)
        e = p - y
        delta = (e[:, None] * self.params["w2"][None, :]) * (a > 0)
        return e, delta, h

    @staticmethod
    def per_example_norms(X, e, delta, h):
        return np.sqrt((X ** 2).sum(1) * (delta ** 2).sum(1) + (delta ** 2).sum(1)
                       + e ** 2 * (h ** 2).sum(1) + e ** 2)

    @staticmethod
    def weighted_grad_sum(X, e, delta, h, c):
        """Σ_i c_i ∇L_i for all parameters."""
        cd, ce = delta * c[:, None], e * c
        return {"W1": X.T @ cd, "b1": cd.sum(0), "w2": h.T @ ce, "b2": np.array([ce.sum()])}


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, g in grads.items():
            g = g + self.wd * self.p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class SGDMomentum:
    def __init__(self, params, lr=0.05, momentum=0.9):
        self.p, self.lr, self.mu = params, lr, momentum
        self.buf = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        for k, g in grads.items():
            self.buf[k] = self.mu * self.buf[k] + g
            self.p[k] -= self.lr * self.buf[k]


def train_mlp(X: np.ndarray, y: np.ndarray, epochs: int = 60, lr: float = 1e-3,
              batch: int = 128, hidden: int = 64, weight_decay: float = 0.0,
              seed: int = 0) -> MLP:
    """Standard (non-private) mini-batch Adam training. No early stopping, by
    design: the baseline is allowed to overfit, as deployed models often do."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    model = MLP(X.shape[1], hidden, seed)
    opt = Adam(model.params, lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            e, d, h = model.backward(X[idx], y[idx])
            g = MLP.weighted_grad_sum(X[idx], e, d, h, np.full(len(idx), 1 / len(idx)))
            opt.step(g)
    return model


def train_lr(X, y, seed: int = 0):
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(X, y)


def train_rf(X, y, seed: int = 0):
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=1, n_jobs=-1,
                                  random_state=seed).fit(X, y)
