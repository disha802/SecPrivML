"""
Step 1 — Dataset acquisition, cleaning and provenance record.

Usage:  python scripts/01_prepare_data.py [--force-download]

Outputs
    data/raw/adult.{zip,data,test,names}
    data/processed/adult_{train,test}_clean.csv
    data/provenance.json      (source, licence, hashes, sizes, preprocessing log)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.privml import data as D  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    prov = D.download_raw(force=args.force_download)
    print("SHA-256 adult.zip :", prov["archive"]["sha256"])

    log: list[str] = []
    train, test = D.load_clean(log)
    D.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(D.PROCESSED_DIR / "adult_train_clean.csv", index=False)
    test.to_csv(D.PROCESSED_DIR / "adult_test_clean.csv", index=False)
    for n in ("adult_train_clean.csv", "adult_test_clean.csv"):
        prov["files"][n] = {"sha256": D.sha256(D.PROCESSED_DIR / n),
                            "bytes": (D.PROCESSED_DIR / n).stat().st_size}

    sp = D.make_splits(train, test)
    enc = D.TabularEncoder(D.domains_from(train, test))
    stats = {
        "original": {"train_rows": 32561, "test_rows": 16281, "total_rows": 48842,
                     "attributes": 14, "label": "income (<=50K, >50K)"},
        "final": {"train_rows": len(train), "test_rows": len(test),
                  "total_rows": len(train) + len(test),
                  "features_used": D.FEATURES, "n_features_raw": len(D.FEATURES),
                  "n_features_encoded": len(enc.feature_names)},
        "splits": {"public": len(sp.public), "private_train": len(sp.train),
                   "test": len(sp.test), "seed": 0},
        "class_balance": {"train_pos_rate": round(float(train[D.TARGET].mean()), 4),
                          "test_pos_rate": round(float(test[D.TARGET].mean()), 4)},
        "sensitive_attribute": {"name": D.SENSITIVE,
                                "train_male_rate": round(float((train.sex == "Male").mean()), 4)},
    }
    log.append(f"Split official train into public={len(sp.public)} / private={len(sp.train)} "
               f"(seed 0); official test kept as test={len(sp.test)}")
    log.append(f"Encoded feature dimension: {len(enc.feature_names)} "
               f"({len(D.NUMERIC)} standardised numeric + one-hot categoricals)")
    D.write_provenance(prov, stats, log)

    print("\nPreparation log")
    for line in log:
        print("  -", line)


if __name__ == "__main__":
    main()
