#!/usr/bin/env python3
# src/data/fetch_mt5_ohlcv.py

import sys
from pathlib import Path
import argparse
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# ── プロジェクト root/src をパスに追加 ──
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # project_root/
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# ── 共通ユーティリティ読み込み ──
from utils.common import load_config, resolve_path

def main():
    # 1) 設定ファイルからデフォルト値を読み込む
    cfg = load_config()
    default_symbol    = cfg.get("symbol")
    default_timeframe = cfg.get("timeframe")
    default_bars      = cfg.get("bars")

    # 2) argparse で上書き可能に。省略時は config の値を使う
    parser = argparse.ArgumentParser(description="Fetch OHLCV from MT5")
    parser.add_argument("--symbol",    default=default_symbol,
                        help=f"通貨ペア (例: USDJPY; default: {default_symbol})")
    parser.add_argument("--timeframe", default=default_timeframe,
                        help=f"時間足 (例: M5, H1; default: {default_timeframe})")
    parser.add_argument("--bars",      type=int, default=default_bars,
                        help=f"取得本数 (default: {default_bars})")
    args = parser.parse_args()

    symbol    = args.symbol
    timeframe = args.timeframe
    bars      = args.bars

    # 3) MT5 初期化（config の mt5.path を優先）
    mt5_path = cfg.get("mt5", {}).get("path")
    if mt5_path:
        mt5.initialize(path=mt5_path)
    else:
        mt5.initialize()

    # 4) 設定ファイルの mt5_data_dir ("${data_base}/raw") を展開
    base_raw_dir = resolve_path(cfg["mt5_data_dir"], cfg)
    # 5) 通貨ペア／時間足サブフォルダを作成
    raw_dir = base_raw_dir / symbol / timeframe
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 6) ファイル名にタイムスタンプを付与
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = f"{symbol}_{timeframe}_{bars}_{ts}.csv"
    out_path = raw_dir / out_fname

    # 7) timeframe マッピング
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,   "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,   "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,   "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1
    }
    tf = tf_map.get(timeframe.upper())
    if tf is None:
        print(f"[ERROR] Unknown timeframe: {timeframe}")
        mt5.shutdown()
        sys.exit(1)

    # 8) データ取得
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"[ERROR] Failed to fetch data for {symbol} {timeframe}")
        mt5.shutdown()
        sys.exit(1)

    # 9) DataFrame 化して CSV 出力
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    df.to_csv(out_path, index=False)

    print(f"[INFO] Fetched and saved: {out_path}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
