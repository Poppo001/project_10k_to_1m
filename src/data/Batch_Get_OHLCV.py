import MetaTrader5 as mt5
import pandas as pd
import os
from datetime import datetime

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

def fetch_and_save(symbol, timeframe, n_bars, output_path="data/processed"):
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        print(f"⚠️ No data for {symbol} {timeframe_to_str(timeframe)} ({mt5.last_error()})")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    os.makedirs(output_path, exist_ok=True)

    filename = f"{output_path}/{symbol}_{timeframe_to_str(timeframe)}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ {symbol} {timeframe_to_str(timeframe)}: {len(df)} rows → {filename}")

    mt5.shutdown()

if __name__ == "__main__":
    # 🔁 ここで一括指定（必要に応じて変更可能）
    symbols = ["EURJPY", "GBPJPY", "EURUSD"]
    timeframes = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4]
    n_bars = 100000

    for symbol in symbols:
        for tf in timeframes:
            fetch_and_save(symbol, tf, n_bars)
