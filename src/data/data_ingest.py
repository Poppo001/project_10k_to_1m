"""
src/data/data_ingest.py
-----------------------
MT5 から OHLCV を取得し、TP/SL 到達をラベル化して
data/raw/ に CSV 保存する最小モジュール
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]     # project_10k_to_1m/
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------- util -------------------------------------------------------------

def tf_map(tf_str: str):
    """文字列 → MT5 タイムフレーム定数"""
    return {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }[tf_str]


def fetch_rates(symbol: str, timeframe, bars: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        sys.exit(f"[E] MT5 returned no data for {symbol}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


# ---------- TP/SL ラベル付け -------------------------------------------------

def label_tp_sl(
    df: pd.DataFrame,
    point: float,
    tp_pips: float,
    sl_pips: float,
    lookahead: int = 48,
) -> pd.DataFrame:
    """
    point : シンボルの point 値 (例 USDJPY=0.001)
    tp_pips/sl_pips : pips 指定 (30 -> 30pips)
    """
    tp_dist = tp_pips * point * 10   # pips → price
    sl_dist = sl_pips * point * 10

    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    labels = np.full(len(df), -1, dtype=int)

    for i in range(len(df) - 1):
        tgt_tp = open_[i] + tp_dist
        tgt_sl = open_[i] - sl_dist

        fut_hi = high[i + 1 : i + 1 + lookahead]
        fut_lo = low[i + 1 : i + 1 + lookahead]

        hit_tp = np.where(fut_hi >= tgt_tp)[0]
        hit_sl = np.where(fut_lo <= tgt_sl)[0]

        if hit_tp.size and hit_sl.size:
            labels[i] = 1 if hit_tp[0] < hit_sl[0] else 0
        elif hit_tp.size:
            labels[i] = 1
        elif hit_sl.size:
            labels[i] = 0
        # else: -1 (未到達)

    df["label"] = labels
    return df


# ---------- CLI --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="USDJPY")
    ap.add_argument("--timeframe", default="H1")
    ap.add_argument("--bars", type=int, default=20_000)
    ap.add_argument("--tp", type=float, default=30)   # pips
    ap.add_argument("--sl", type=float, default=30)   # pips
    args = ap.parse_args()

    # MT5 初期化
    if not mt5.initialize():
        sys.exit(f"[E] MT5 init failed: {mt5.last_error()}")

    tf_const = tf_map(args.timeframe)
    df_rates = fetch_rates(args.symbol, tf_const, args.bars)

    sym_info = mt5.symbol_info(args.symbol)
    if sym_info is None:
        mt5.shutdown()
        sys.exit(f"[E] Symbol {args.symbol} not found in MT5")

    df_labeled = label_tp_sl(
        df_rates,
        point=sym_info.point,
        tp_pips=args.tp,
        sl_pips=args.sl,
    )

    mt5.shutdown()

    out_file = RAW_DIR / f"{args.symbol}_{args.timeframe}_{args.bars}.csv"
    df_labeled.to_csv(out_file, index=False)
    print(f"[+] Saved {out_file.relative_to(ROOT)}  rows={len(df_labeled)}")


if __name__ == "__main__":
    main()
