# src/data/generate_features.py

import sys
from pathlib import Path

# プロジェクトルートの src をインポートパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.common import load_config, resolve_data_root

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from utils.common import load_config, resolve_data_root

def add_features(df):
    df["return"] = df["close"].pct_change()
    df["ma"] = df["close"].rolling(10).mean()
    df["rsi"] = 100 - 100 / (1 + df["return"].rolling(14).mean())
    return df.dropna()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--bars", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_data_root(cfg)

    # 入力
    raw_dir = root / "raw" / args.symbol / args.timeframe
    input_csv = sorted(raw_dir.glob(f"{args.symbol}_{args.timeframe}_{args.bars}_*.csv"))[-1]
    df = pd.read_csv(input_csv, parse_dates=["time"])

    # 特徴量生成
    df_feat = add_features(df)

    # 出力
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "processed" / args.symbol / args.timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"selfeat_{args.symbol}_{args.timeframe}_{args.bars}_{ts}.csv"
    df_feat.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✔ Saved: {out_path}")

if __name__ == "__main__":
    main()
