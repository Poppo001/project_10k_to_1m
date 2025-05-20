"""
src/data/feature_gen.py
-----------------------
raw CSV → テクニカル指標／価格派生を付与して
data/processed/ に保存。
"""
from pathlib import Path
import argparse

import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index("time", inplace=True)

    df["ret_1"] = df["close"].pct_change()
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.sma(length=20, append=True, talib=False)

    df.dropna(inplace=True)
    return df.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="raw CSV filename in data/raw/")
    args = ap.parse_args()

    src = RAW_DIR / args.file
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src, parse_dates=["time"])
    df_feat = build_features(df)

    out = PROC_DIR / f"feat_{args.file}"
    df_feat.to_csv(out, index=False)
    print(f"[+] Saved features: {out.relative_to(ROOT)}  rows={len(df_feat)}")


if __name__ == "__main__":
    main()
