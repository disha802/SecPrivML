"""Utility metrics shared by every experiment."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def utility(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1 (positive class = income>50K) and ROC-AUC."""
    pred = (prob >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:      # constant predictions
        auc = 0.5
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": auc,
    }


def mean_std(rows: list[dict]) -> dict:
    """Aggregate a list of metric dicts (one per seed) into mean and std."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float, np.floating))]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out
