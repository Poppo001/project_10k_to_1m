#!/usr/bin/env python3
# src/data/fetch_mt5_ohlcv.py

import argparse
import pandas as pd
from pathlib import Path
import datetime
import sys
from src.utils.common import load_config, resolve_path

def main():
    parser = argparse.ArgumentParser(
        description="Fetch raw OHLCV from MT5 and save with date prefix"
    )
    parser.add_argument("--symbol",    required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--bars",      type=int, required=True)
    parser.add_argument("--out",       help="省略時は config の mt5_data_dir から自動生成")
    args = parser.parse_args()

    cfg = load_config()
    raw_cfg = cfg.get("raw_data", {})
    date_fmt = raw_cfg.get("date_format", "%Y%m%d")
    today_str = datetime.datetime.now().strftime(date_fmt)

    # 出力先ディレクトリ
    raw_dir = resolve_path(cfg["mt5_data_dir"], cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"{today_str}_{args.symbol}_{args.timeframe}_{args.bars}.csv"
    out_path = Path(args.out) if args.out else raw_dir / default_name

    # 実際の MT5 取得ロジック (アンコメントして使用)
    # import MetaTrader5 as mt5
    # mt5.initialize(path=cfg["mt5"]["path"],
    #                login=cfg["mt5"]["login"],
    #                password=cfg["mt5"]["password"])
    # rates = mt5.copy_rates_from_pos(
    #     args.symbol,
    #     getattr(mt5, f"TIMEFRAME_{args.timeframe}"),
    #     0,
    #     args.bars
    # )
    # df = pd.DataFrame(rates)
    # df["time"] = pd.to_datetime(df["time"], unit="s")
    # df.to_csv(out_path, index=False)

    # プレースホルダ
    print(f"[INFO] Placeholder fetch, would save to: {out_path}")

if __name__ == "__main__":
    main()
