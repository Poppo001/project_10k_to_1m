#!/usr/bin/env python3
# src/data/feature_gen.py

import argparse
import pandas as pd
from pathlib import Path

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ma"] = df["close"].rolling(window=14).mean()
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    avg_up = up.rolling(14).mean()
    avg_dn = dn.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + avg_up / avg_dn))
    df["atr"] = (df["high"].combine(df["close"].shift(), max) -
                 df["low"].combine(df["close"].shift(), min))
    return df.dropna().reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["time"])
    feat_df = generate_features(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(args.out, index=False)
    print(f"[INFO] Features saved: {args.out}")

if __name__ == "__main__":
    main()
