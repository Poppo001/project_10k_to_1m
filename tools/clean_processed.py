#!/usr/bin/env python3
from pathlib import Path

def keep_latest(dir_path: Path, pattern: str, keep: int = 1):
    files = sorted(dir_path.glob(pattern))
    for f in files[:-keep]:
        print(f"Removing {f.name}")
        f.unlink()

if __name__ == "__main__":
    proc_root = Path("project_10k_to_1m_data/processed/USDJPY/M5")
    keep_latest(proc_root, "feat_USDJPY_M5_*.csv", keep=1)
    keep_latest(proc_root, "labeled_USDJPY_M5_*.csv", keep=1)
    keep_latest(proc_root, "selfeat_USDJPY_M5_*.csv", keep=1)
