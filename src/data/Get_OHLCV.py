import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime

def fetch_ohlcv(symbol, timeframe, n_bars, output_path="data/processed"):
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to get data for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    os.makedirs(output_path, exist_ok=True)

    # 時間足を文字列で識別（H1などをファイル名に）
    timeframe_str = timeframe_to_str(timeframe)
    filename = f"{output_path}/{symbol}_{timeframe_str}.csv"
    df.to_csv(filename, index=False)

    print(f"✅ {symbol} {timeframe_str}: {len(df)} rows saved to {filename}")

    mt5.shutdown()
    return df

# 時間足を文字列に変換する関数（独自定義）
def timeframe_to_str(tf):
    mapping = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1"
    }
    return mapping.get(tf, "UNKNOWN")

if __name__ == "__main__":
    # 🔧 ここでシンボル・時間足・本数を自由に設定
    symbol = "USDJPY"
    timeframe = mt5.TIMEFRAME_H1
    n_bars = 1000000

    fetch_ohlcv(symbol, timeframe, n_bars)
