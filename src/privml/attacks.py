"""
Module 1 — Privacy attacks used to measure *empirical* privacy leakage.

Three threat models are evaluated, one per data-release stage:

1. Membership inference (MIA) on a trained model
   Loss-threshold attack (Yeom et al., CSF 2018): a record is predicted to be a
   training member when the model's loss on it is low. Reported as
   ROC-AUC, membership advantage max(TPR - FPR), and TPR at 1 % FPR
   (Carlini et al., S&P 2022 recommend low-FPR reporting).

2. Attribute inference on released features / representations
   An adversary trains a classifier to predict the sensitive attribute (sex)
   from whatever is released (perturbed records, compressed features).
   Reported as attacker accuracy and balanced accuracy versus the
   majority-class prior.

3. Re-identification (linkage) on released records
   An adversary who knows a target's quasi-identifiers (QIs) links them to the
   released table. Prosecutor risk of a record = 1 / |equivalence class|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

QUASI_IDENTIFIERS = ["age", "sex", "race", "marital-status", "native-country",
                     "education-num", "workclass"]


# --------------------------------------------------------------------------- MIA
def _per_example_loss(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def membership_inference(model, X_in, y_in, X_out, y_out, n: int = 5000,
                         seed: int = 0) -> dict:
    """Loss-threshold MIA on a balanced sample of members and non-members."""
    rng = np.random.default_rng(seed)
    n = min(n, len(X_in), len(X_out))
    i = rng.choice(len(X_in), n, replace=False)
    o = rng.choice(len(X_out), n, replace=False)
    score = -np.concatenate([_per_example_loss(model, X_in[i], y_in[i]),
                             _per_example_loss(model, X_out[o], y_out[o])])
    member = np.concatenate([np.ones(n), np.zeros(n)])
    fpr, tpr, _ = roc_curve(member, score)
    return {
        "mia_auc": float(roc_auc_score(member, score)),
        "mia_advantage": float(np.max(tpr - fpr)),
        "mia_tpr_at_1fpr": float(np.interp(0.01, fpr, tpr)),
        "train_loss": float(_per_example_loss(model, X_in[i], y_in[i]).mean()),
        "test_loss": float(_per_example_loss(model, X_out[o], y_out[o]).mean()),
    }


# ------------------------------------------------------------ attribute inference
def attribute_inference(Z_attacker: np.ndarray, s_attacker: np.ndarray,
                        Z_target: np.ndarray, s_target: np.ndarray,
                        kind: str = "gbm", seed: int = 0) -> dict:
    """Train an attacker on (Z_attacker -> s) and evaluate on the target records.

    ``kind='lr'`` is a linear adversary, ``kind='gbm'`` a strong non-linear one.
    """
    if kind == "lr":
        clf = LogisticRegression(max_iter=2000)
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(Z_attacker, s_attacker)
    pred = clf.predict(Z_target)
    prior = max(s_target.mean(), 1 - s_target.mean())
    return {
        "attr_acc": float((pred == s_target).mean()),
        "attr_bal_acc": float(balanced_accuracy_score(s_target, pred)),
        "attr_prior": float(prior),
    }


# --------------------------------------------------------------- re-identification
def _hashable(v):
    return tuple(sorted(v)) if isinstance(v, (set, frozenset, list, tuple)) else v


def reidentification_risk(df: pd.DataFrame, qis: list[str] = QUASI_IDENTIFIERS) -> dict:
    """Prosecutor re-identification risk of a released table over its QIs."""
    key = df[qis].map(lambda v: str(_hashable(v))).agg("|".join, axis=1)
    sizes = key.map(key.value_counts()).to_numpy(float)
    risk = 1.0 / sizes
    return {
        "unique_frac": float((sizes == 1).mean()),
        "reid_avg_risk": float(risk.mean()),
        "reid_max_risk": float(risk.max()),
        "min_class_size": int(sizes.min()),
        "n_classes": int(key.nunique()),
    }
