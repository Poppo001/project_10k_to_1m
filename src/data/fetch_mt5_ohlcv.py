#!/usr/bin/env python3
# src/data/fetch_mt5_ohlcv.py

import argparse
import pandas as pd
from pathlib import Path
import datetime
import sys

# ── プロジェクトルートを sys.path に追加 ────────────────────
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
sys.path.insert(0, str(project_root))

from src.utils.common import load_config
# Uncomment if using the MetaTrader5 package:
# import MetaTrader5 as mt5

def main():
    parser = argparse.ArgumentParser(
        description="Fetch raw OHLCV from MT5 and save with date prefix"
    )
    parser.add_argument("--symbol",    required=True, help="通貨ペア (例: USDJPY)")
    parser.add_argument("--timeframe", required=True, help="時間軸 (例: M5, H1)")
    parser.add_argument("--bars",      type=int, required=True, help="取得本数")
    parser.add_argument(
        "--out",
        required=False,
        help="出力先 (省略時は config の mt5_data_dir＋日付付きファイル名)"
    )
    args = parser.parse_args()

    # config.yaml 読み込み
    cfg = load_config()
    raw_cfg = cfg.get("raw_data", {})
    date_fmt = raw_cfg.get("date_format", "%Y%m%d")
    today_str = datetime.datetime.now().strftime(date_fmt)

    # 出力ディレクトリ & ファイル名
    raw_dir = Path(cfg["mt5_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"{today_str}_{args.symbol}_{args.timeframe}_{args.bars}.csv"
    out_path = Path(args.out) if args.out else raw_dir / default_name

    # MT5 初期化（アンコメント時）
    # mt5.initialize(
    #     path=cfg["mt5"]["path"],
    #     login=cfg["mt5"]["login"],
    #     password=cfg["mt5"]["password"]
    # )
    # rates = mt5.copy_rates_from_pos(
    #     args.symbol,
    #     getattr(mt5, f"TIMEFRAME_{args.timeframe}"),
    #     0,
    #     args.bars
    # )
    # df = pd.DataFrame(rates)
    # df["time"] = pd.to_datetime(df["time"], unit="s")
    # df.to_csv(out_path, index=False)

    # プレスホルダーログ
    print(f"[INFO] (MT5 fetch placeholder) Will save to: {out_path}")
    # 例: df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
