"""Dataset integrity checks (skipped until scripts/01_prepare_data.py has been run)."""

import json

import numpy as np
import pytest

from src.privml import data as D

PROV = D.ROOT / "data" / "provenance.json"
pytestmark = pytest.mark.skipif(not PROV.exists(), reason="run scripts/01_prepare_data.py first")


def test_hashes_match_provenance_record():
    # Raw files may be absent in a submission bundle (re-downloadable via 01_prepare_data.py);
    # every file that is present must match its recorded hash.
    rec = json.loads(PROV.read_text())
    checked = 0
    for name, meta in rec["hashes"]["files"].items():
        path = (D.RAW_DIR if name.startswith("adult.") else D.PROCESSED_DIR) / name
        if path.exists():
            assert D.sha256(path) == meta["sha256"], name
            checked += 1
    assert checked > 0


def test_prepared_shapes_and_encoding():
    P = D.prepare()
    assert len(P.X_pub) + len(P.X_tr) == 30162 and len(P.X_te) == 15060
    assert P.X_tr.shape[1] == len(P.encoder.feature_names) == 87
    cat = P.X_tr[:, len(D.NUMERIC):]
    assert np.allclose(cat.sum(1), len(D.CATEGORICAL))          # one value per categorical
    assert abs(P.X_pub[:, :len(D.NUMERIC)].mean(0)).max() < 1e-6  # scaler fitted on public


def test_public_and_private_partition_the_official_train_set():
    import pandas as pd
    P = D.prepare()
    full = pd.read_csv(D.PROCESSED_DIR / "adult_train_clean.csv")
    rows = lambda df: sorted(map(tuple, df.astype(str).to_numpy()))
    assert len(P.splits.public) == 3016 and len(P.splits.train) == 27146
    assert rows(pd.concat([P.splits.public, P.splits.train])) == rows(full)
