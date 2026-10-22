"""
Module 4 — Privacy-Preserving Data Mining (syntactic anonymisation).

Mondrian multidimensional k-anonymity (LeFevre, DeWitt & Ramakrishnan, ICDE 2006)
-------------------------------------------------------------------------------
Recursively split the table on the quasi-identifier (QI) with the widest
normalised span until no allowable cut remains. A cut is allowable when both
halves contain at least k records (and, optionally, at least l distinct values
of the sensitive attribute — distinct l-diversity, Machanavajjhala et al. 2007).

* Numeric QIs are split at the median and generalised to the partition range;
  for model training the range is represented by the partition mean.
* Categorical QIs are split by a frequency-ordered binary set partition and
  generalised to the *set* of values present. (Adult ships without
  generalisation hierarchies, so set-valued generalisation is used; these sets
  are encoded as fractional multi-hot vectors for learning.)

Quality metrics
---------------
NCP           Normalised Certainty Penalty (Xu et al. 2006): range / domain range
              for numeric, (|set|-1)/(|domain|-1) for categorical; averaged over
              QIs and records (0 = no loss, 1 = fully suppressed).
homogeneity   fraction of records whose equivalence class has a single sensitive
              value — these records' income is disclosed with certainty.
t-closeness   max over classes of the distance between the class's and the
              table's sensitive distribution (for a binary attribute the EMD
              reduces to |p_class - p_table|).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .attacks import QUASI_IDENTIFIERS
from .data import NUMERIC, TARGET


def mondrian(df: pd.DataFrame, k: int, qis: list[str] = QUASI_IDENTIFIERS,
             l_div: int | None = None, sensitive: str = TARGET) -> list[np.ndarray]:
    """Return a list of index arrays, one per equivalence class."""
    num = {q: df[q].to_numpy(float) for q in qis if q in NUMERIC}
    cat = {q: pd.Categorical(df[q]).codes for q in qis if q not in NUMERIC}
    ranges = {q: (v.max() - v.min()) or 1.0 for q, v in num.items()}
    dom = {q: max(c.max() + 1, 2) for q, c in cat.items()}
    sens = df[sensitive].to_numpy()

    def allowable(idx):
        return len(idx) >= k and (l_div is None or len(np.unique(sens[idx])) >= l_div)

    def span(q, idx):
        if q in num:
            v = num[q][idx]
            return (v.max() - v.min()) / ranges[q]
        return (len(np.unique(cat[q][idx])) - 1) / (dom[q] - 1)

    def split(q, idx):
        if q in num:
            v = num[q][idx]
            med = np.median(v)
            lhs = v <= med
            if lhs.all():
                lhs = v < med
        else:
            v = cat[q][idx]
            vals, counts = np.unique(v, return_counts=True)
            if len(vals) < 2:
                return None
            order = vals[np.argsort(-counts)]
            cum = np.cumsum(np.sort(counts)[::-1])
            n_left = max(1, int(np.searchsorted(cum, len(idx) / 2)) + 1)
            n_left = min(n_left, len(vals) - 1)
            lhs = np.isin(v, order[:n_left])
        return idx[lhs], idx[~lhs]

    done, stack = [], [np.arange(len(df))]
    while stack:
        idx = stack.pop()
        spans = {q: span(q, idx) for q in qis}
        for q in sorted(qis, key=lambda q: -spans[q]):        # widest dimension first
            if spans[q] == 0:
                continue
            parts = split(q, idx)
            if parts and allowable(parts[0]) and allowable(parts[1]):
                stack.extend(parts)
                break
        else:                                                  # no allowable cut: leaf
            done.append(idx)
    return done


def generalise(df: pd.DataFrame, classes: list[np.ndarray],
               qis: list[str] = QUASI_IDENTIFIERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (training view, display view) of the anonymised table.

    Training view: numeric QIs -> class mean, categorical QIs -> frozenset.
    Display view:  numeric QIs -> "lo-hi" strings (as an analyst would see them).
    """
    train_view, disp = df.copy(), df.copy().astype({q: object for q in qis})
    for q in qis:
        train_view[q] = train_view[q].astype(object)
    for idx in classes:
        for q in qis:
            vals = df[q].to_numpy()[idx]
            if q in NUMERIC:
                lo, hi = vals.min(), vals.max()
                train_view.iloc[idx, train_view.columns.get_loc(q)] = float(vals.mean())
                disp.iloc[idx, disp.columns.get_loc(q)] = f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"
            else:
                s = frozenset(vals)
                cell = next(iter(s)) if len(s) == 1 else s
                col = train_view.columns.get_loc(q)
                for r in idx:
                    train_view.iat[r, col] = cell
                disp.iloc[idx, disp.columns.get_loc(q)] = "{" + ",".join(sorted(s)) + "}" \
                    if len(s) > 1 else next(iter(s))
    for q in qis:
        if q in NUMERIC:
            train_view[q] = train_view[q].astype(float)
    return train_view, disp


def ncp(df: pd.DataFrame, classes: list[np.ndarray], qis: list[str] = QUASI_IDENTIFIERS) -> float:
    total = 0.0
    for q in qis:
        v = df[q].to_numpy()
        if q in NUMERIC:
            rng = float(v.max() - v.min()) or 1.0
            total += sum(len(i) * (v[i].max() - v[i].min()) / rng for i in classes)
        else:
            d = max(len(np.unique(v)) - 1, 1)
            total += sum(len(i) * (len(np.unique(v[i])) - 1) / d for i in classes)
    return total / (len(df) * len(qis))


def sensitive_disclosure(df: pd.DataFrame, classes: list[np.ndarray],
                         sensitive: str = TARGET) -> dict:
    s = df[sensitive].to_numpy()
    p_all = s.mean()
    homog = sum(len(i) for i in classes if len(np.unique(s[i])) == 1) / len(df)
    t = max(abs(s[i].mean() - p_all) for i in classes)
    return {"homogeneity_rate": float(homog), "t_closeness": float(t)}


def anonymise(df: pd.DataFrame, k: int, l_div: int | None = None,
              qis: list[str] = QUASI_IDENTIFIERS) -> dict:
    classes = mondrian(df, k, qis, l_div)
    train_view, disp = generalise(df, classes, qis)
    sizes = np.array([len(i) for i in classes])
    return {"train_view": train_view, "display_view": disp, "classes": classes,
            "n_classes": len(classes), "min_class": int(sizes.min()),
            "avg_class": float(sizes.mean()), "ncp": ncp(df, classes, qis),
            **sensitive_disclosure(df, classes)}
