#!/usr/bin/env python3
# src/data/label_gen.py

import argparse
import pandas as pd
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="feat CSV path")
    parser.add_argument("--tp", type=float, required=True, help="take-profit (pips)")
    parser.add_argument("--sl", type=float, required=True, help="stop-loss (pips)")
    parser.add_argument("--exclude_before_release", type=bool, default=False)
    parser.add_argument("--release_exclude_window_mins", type=int, default=0)
    parser.add_argument("--out", required=True, help="output labeled CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.file, parse_dates=["time"])
    df = df.copy()
    # future_returnとして単純に次のclose-現在closeをpips換算
    pip_unit = 0.0001
    df["future_return"] = (df["close"].shift(-1) - df["close"]) / pip_unit
    df["label"] = (df["future_return"] > 0).astype(int)
    # ※発表前除外等は後日追加

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.dropna().to_csv(args.out, index=False)
    print(f"[INFO] Labeled data saved: {args.out}")

if __name__ == "__main__":
    main()
