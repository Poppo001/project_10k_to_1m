#!/usr/bin/env python3
# src/data/label_gen.py

import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--tp", type=float, required=True)
    parser.add_argument("--sl", type=float, required=True)
    parser.add_argument("--exclude_before_release", type=bool, default=False)
    parser.add_argument("--release_exclude_window_mins", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.file, parse_dates=["time"]).copy()
    pip_unit = 0.0001
    df["future_return"] = (df["close"].shift(-1) - df["close"]) / pip_unit
    df["label"] = (df["future_return"] > 0).astype(int)
    # TODO: exclude_before_release のロジック実装

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.dropna().to_csv(args.out, index=False)
    print(f"[INFO] Labeled data saved: {args.out}")

if __name__ == "__main__":
    main()
