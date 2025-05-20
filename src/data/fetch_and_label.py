
#!/usr/bin/env python
"""
fetch_and_label.py
------------------
Fetch USDJPY (or any symbol) H1 data from MetaTrader 5, label each bar by
whether a fixed-distance Take‑Profit (TP) or Stop‑Loss (SL) would be hit first,
and save the result as a CSV. Optionally upload to Google Drive so it can be
picked up from Google Colab.

Requirements
------------
pip install -r requirements.txt

Usage
-----
# basic (fetch 20k bars, label with 30 pips TP/SL, save locally)
python fetch_and_label.py --symbol USDJPY --bars 20000 --tp 30 --sl 30 --csv usd_h1_tp30sl30.csv

# upload the CSV to your Google Drive root folder after creation
python fetch_and_label.py --drive True

Notes
-----
* You must have the MT5 terminal running and logged in on the same machine.
* Point value is obtained from MT5 to convert pips → price.
* Label definition:
    1  : TP reached before SL
    0  : SL reached before TP
   -1  : Neither TP nor SL hit within `lookahead` bars (default 48 = 2 days)
"""

import argparse
import datetime as dt
import os
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Optional Google Drive upload
try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
except ModuleNotFoundError:
    GoogleAuth = None
    GoogleDrive = None

def parse_args():
    p = argparse.ArgumentParser(description="Fetch MT5 data and label TP/SL events")
    p.add_argument("--symbol", default="USDJPY", help="Trading symbol")
    p.add_argument("--timeframe", default="H1", choices=["M1","M5","M15","M30","H1","H4","D1"],
                   help="Time‑frame to fetch")
    p.add_argument("--bars", type=int, default=20000, help="Number of bars to fetch (most recent)")
    p.add_argument("--tp", type=float, default=30, help="TP distance in pips")
    p.add_argument("--sl", type=float, default=30, help="SL distance in pips")
    p.add_argument("--lookahead", type=int, default=48, help="Number of future bars to scan for TP/SL hit")
    p.add_argument("--csv", default="labeled_data.csv", help="Output CSV path")
    p.add_argument("--drive", type=lambda s: s.lower() in {"true","1","yes"}, default=False,
                   help="Upload result to Google Drive (requires first‑time OAuth in browser)")
    return p.parse_args()

def timeframe_to_mt5(tf_str):
    return {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }[tf_str]

def ensure_mt5():
    if not mt5.initialize():
        print(f"[E] MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)

def fetch_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        print(f"[E] No data returned for {symbol}")
        sys.exit(1)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def label_rows(df, tp_pips, sl_pips, lookahead):
    point = mt5.symbol_info(df_symbol)['point']
    tp_d = tp_pips * point * 10  # pips to price (JPY pairs: 1 pip=0.01)
    sl_d = sl_pips * point * 10

    labels = np.full(len(df), -1, dtype=int)

    high = df['high'].values
    low = df['low'].values
    openp = df['open'].values

    for i in range(len(df) - 1):
        tp_price = openp[i] + tp_d
        sl_price = openp[i] - sl_d
        future_high = high[i+1:i+1+lookahead]
        future_low = low[i+1:i+1+lookahead]

        hit_tp = np.where(future_high >= tp_price)[0]
        hit_sl = np.where(future_low <= sl_price)[0]

        if hit_tp.size and hit_sl.size:
            labels[i] = 1 if hit_tp[0] < hit_sl[0] else 0
        elif hit_tp.size:
            labels[i] = 1
        elif hit_sl.size:
            labels[i] = 0
        # else remains -1
    df['label'] = labels
    return df

def upload_to_drive(local_path):
    if GoogleAuth is None:
        print("[W] PyDrive not installed; skip Drive upload.")
        return
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    f = drive.CreateFile({'title': os.path.basename(local_path)})
    f.SetContentFile(local_path)
    f.Upload()
    print(f"[+] Uploaded to Google Drive: {f['alternateLink']}")
    return f['alternateLink']

if __name__ == "__main__":
    args = parse_args()
    ensure_mt5()
    df_symbol = mt5.symbol_info(args.symbol)
    if df_symbol is None:
        print(f"[E] Symbol {args.symbol} not found in MT5")
        mt5.shutdown()
        sys.exit(1)

    tf = timeframe_to_mt5(args.timeframe)
    df = fetch_data(args.symbol, tf, args.bars)
    df = label_rows(df, args.tp, args.sl, args.lookahead)
    df.to_csv(args.csv, index=False)
    print(f"[+] Saved labeled data to {args.csv} (rows={len(df)})")

    if args.drive:
        upload_to_drive(args.csv)

    mt5.shutdown()
