"""Shared helpers for runner scripts: path setup, logging to console + file, CSV output."""

import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")      # silences a joblib warning on Windows

import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


class Run:
    def __init__(self, out_dir: Path):
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_f = open(self.out / "run_log.txt", "w", encoding="utf-8")
        self.t0 = time.time()
        import scipy
        import sklearn
        env = {"python": platform.python_version(), "platform": platform.platform(),
               "numpy": np.__version__, "pandas": pd.__version__,
               "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        (self.out / "environment.json").write_text(json.dumps(env, indent=2))
        self.log(f"Environment: {env}")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def save(self, name: str, rows) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df.to_csv(self.out / f"{name}.csv", index=False)
        self.log(f"saved {name}.csv ({len(df)} rows)")
        return df

    def stage(self, title: str, fn, *args, **kw):
        self.log(f"=== {title} ===")
        t = time.time()
        out = fn(*args, **kw)
        self.log(f"--- {title} done in {time.time() - t:.1f}s")
        return out
