"""
Data acquisition, provenance and preprocessing for the UCI Adult dataset.

Pipeline
--------
1. ``download_raw``      fetch the official UCI archive, record SHA-256 hashes
2. ``load_clean``        parse train/test files, strip whitespace, fix the test
                         label suffix ("."), drop rows with missing values ("?"),
                         drop ``fnlwgt`` (census sampling weight) and
                         ``education`` (duplicate of ``education-num``)
3. ``make_splits``       public (10 % of official train) / private train / test
4. ``TabularEncoder``    standardise numeric columns, one-hot categorical columns;
                         fitted on the *public* split only so that no statistic
                         of the private training data leaks through preprocessing

The public split models the common assumption in the DP literature that a small
amount of non-sensitive data from the same distribution is available for
preprocessing and hyper-parameter choices.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

UCI_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "income"            # utility label: 1 if income > 50K
SENSITIVE = "sex"            # privacy label for attribute-inference / compressive privacy
DROPPED = ["fnlwgt", "education"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(force: bool = False) -> dict:
    """Download the UCI archive into data/raw and return a provenance dict."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zpath = RAW_DIR / "adult.zip"
    if force or not zpath.exists():
        with urllib.request.urlopen(UCI_URL, timeout=120) as r:
            zpath.write_bytes(r.read())
    with zipfile.ZipFile(zpath) as z:
        for name in ("adult.data", "adult.test", "adult.names"):
            (RAW_DIR / name).write_bytes(z.read(name))
    return {
        "archive": {"file": "adult.zip", "sha256": sha256(zpath), "bytes": zpath.stat().st_size},
        "files": {
            n: {"sha256": sha256(RAW_DIR / n), "bytes": (RAW_DIR / n).stat().st_size}
            for n in ("adult.data", "adult.test", "adult.names")
        },
    }


def _read(path: Path, skip_first: bool) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()[1:] if skip_first else text.splitlines()
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=COLUMNS,
                     skipinitialspace=True, na_values="?")
    return df.dropna(how="all")


def load_clean(log: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned (train, test) frames. Appends human-readable steps to ``log``."""
    log = log if log is not None else []
    train = _read(RAW_DIR / "adult.data", skip_first=False)
    test = _read(RAW_DIR / "adult.test", skip_first=True)   # first line is a comment
    log.append(f"Loaded raw files: train={len(train)} rows, test={len(test)} rows, "
               f"{len(COLUMNS)} columns")

    for name, df in (("train", train), ("test", test)):
        df["income"] = df["income"].str.strip().str.rstrip(".")
        miss = int(df.isna().any(axis=1).sum())
        per_col = {c: int(v) for c, v in df.isna().sum().items() if v}
        log.append(f"{name}: {miss} rows with missing values ('?') in {per_col}")

    n_tr, n_te = len(train), len(test)
    train, test = train.dropna().copy(), test.dropna().copy()
    log.append(f"Dropped incomplete rows: train {n_tr}->{len(train)}, test {n_te}->{len(test)}")

    dup = int(train.duplicated().sum())
    log.append(f"Exact duplicate rows in train: {dup} (kept: duplicates are distinct "
               f"census respondents with identical attributes)")

    for df in (train, test):
        df.drop(columns=DROPPED, inplace=True)
        df[TARGET] = (df[TARGET] == ">50K").astype(int)
        for c in NUMERIC:
            df[c] = df[c].astype(float)
    log.append(f"Dropped columns {DROPPED}; binarised target income>50K -> 1")
    log.append(f"Positive rate: train {train[TARGET].mean():.4f}, test {test[TARGET].mean():.4f}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


@dataclass
class Splits:
    public: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame


def make_splits(train: pd.DataFrame, test: pd.DataFrame, public_frac: float = 0.10,
                seed: int = 0) -> Splits:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    n_pub = int(round(public_frac * len(train)))
    return Splits(public=train.iloc[idx[:n_pub]].reset_index(drop=True),
                  train=train.iloc[idx[n_pub:]].reset_index(drop=True),
                  test=test.copy())


class TabularEncoder:
    """Standardise numeric columns and one-hot encode categoricals.

    Categories are taken from the union of all splits' *domains* (the public
    attribute schema in ``adult.names``), never from private statistics.
    Generalised (k-anonymised) records can be encoded as fractional multi-hot
    vectors by passing a dict ``{column: list_of_values}`` per cell.
    """

    def __init__(self, domains: dict[str, list[str]]):
        self.domains = domains
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        for c in NUMERIC:
            self.mean_[c] = float(df[c].mean())
            self.std_[c] = float(df[c].std()) or 1.0
        return self

    @property
    def feature_names(self) -> list[str]:
        names = list(NUMERIC)
        for c in CATEGORICAL:
            names += [f"{c}={v}" for v in self.domains[c]]
        return names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = [((df[NUMERIC].to_numpy(float) - np.array([self.mean_[c] for c in NUMERIC]))
                   / np.array([self.std_[c] for c in NUMERIC]))]
        for c in CATEGORICAL:
            dom = self.domains[c]
            pos = {v: i for i, v in enumerate(dom)}
            block = np.zeros((len(df), len(dom)))
            for r, val in enumerate(df[c].to_numpy()):
                if isinstance(val, (set, frozenset, list, tuple)):   # generalised cell
                    for v in val:
                        block[r, pos[v]] = 1.0 / len(val)
                else:
                    block[r, pos[val]] = 1.0
            blocks.append(block)
        return np.hstack(blocks).astype(np.float32)


def domains_from(*dfs: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(set().union(*[set(d[c].unique()) for d in dfs])) for c in CATEGORICAL}


@dataclass
class Prepared:
    splits: Splits
    encoder: TabularEncoder
    X_pub: np.ndarray
    X_tr: np.ndarray
    X_te: np.ndarray
    y_pub: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    s_pub: np.ndarray      # sensitive attribute (1 = Male)
    s_tr: np.ndarray
    s_te: np.ndarray


def prepare(seed: int = 0) -> Prepared:
    """Load processed CSVs (run scripts/01_prepare_data.py first) and encode."""
    train = pd.read_csv(PROCESSED_DIR / "adult_train_clean.csv")
    test = pd.read_csv(PROCESSED_DIR / "adult_test_clean.csv")
    sp = make_splits(train, test, seed=seed)
    enc = TabularEncoder(domains_from(train, test)).fit(sp.public)
    sens = lambda d: (d[SENSITIVE] == "Male").to_numpy(int)
    return Prepared(sp, enc,
                    enc.transform(sp.public), enc.transform(sp.train), enc.transform(sp.test),
                    sp.public[TARGET].to_numpy(int), sp.train[TARGET].to_numpy(int),
                    sp.test[TARGET].to_numpy(int),
                    sens(sp.public), sens(sp.train), sens(sp.test))


def write_provenance(prov: dict, stats: dict, log: list[str]) -> None:
    record = {
        "dataset": "Adult (Census Income)",
        "source_url": "https://archive.ics.uci.edu/dataset/2/adult",
        "download_url": UCI_URL,
        "doi": "10.24432/C5XW20",
        "creators": "Barry Becker, Ronny Kohavi (extracted from the 1994 US Census database)",
        "publisher": "UCI Machine Learning Repository, University of California, Irvine",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "original_collection": "1994 Current Population Survey (US Census Bureau); "
                               "extraction by Becker (1996) with filters "
                               "AGI>100, AFNLWGT>1, AHRSWK>0",
        "version": "UCI static release id 2 (donated 1996-04-30)",
        "retrieved_on": date.today().isoformat(),
        "retrieval_method": "HTTPS download of the official UCI archive by "
                            "scripts/01_prepare_data.py (urllib)",
        "hashes": prov,
        "sizes": stats,
        "preprocessing_log": log,
    }
    (ROOT / "data" / "provenance.json").write_text(json.dumps(record, indent=2))
