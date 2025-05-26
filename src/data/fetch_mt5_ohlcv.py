# src/data/fetch_mt5_ohlcv.py
# 実行例
# bash
#python src/data/fetch_mt5_ohlcv.py --symbol USDJPY --timeframe H1 --bars 100000 --ou

import MetaTrader5 as mt5
import pandas as pd
import argparse
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="通貨ペア例: USDJPY")
    parser.add_argument("--timeframe", type=str, default="H1", help="時間足例: H1")
    parser.add_argument("--bars", type=int, default=20000, help="取得本数")
    parser.add_argument("--out", type=str, required=True, help="出力CSVパス")
    args = parser.parse_args()

    # MT5接続
    mt5.initialize()
    tf = getattr(mt5, args.timeframe)
    utc_to = datetime.now()
    rates = mt5.copy_rates_from(args.symbol, tf, utc_to, args.bars)
    mt5.shutdown()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.to_csv(args.out, index=False)
    print(f"[INFO] MT5データ保存: {args.out}")

if __name__ == "__main__":
    main()
