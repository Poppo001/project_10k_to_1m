#!/usr/bin/env python3
# src/data/backtest.py

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.family'] = 'MS Gothic'

def main():
    parser = argparse.ArgumentParser(description="リアルTP/SL対応バックテスト")
    parser.add_argument("--csv",       required=True, help="入力CSV（ラベル付）")
    parser.add_argument("--model",     required=True, help="学習済モデル.pkl")
    parser.add_argument("--report",    required=True, help="出力レポートJSON")
    parser.add_argument("--curve_out", required=True, help="出力損益曲線PNG")
    parser.add_argument("--tp_pips",    type=float, default=30.0, help="TP (pips)")
    parser.add_argument("--sl_pips",    type=float, default=30.0, help="SL (pips)")
    parser.add_argument("--spread",     type=float, default=0.2,  help="往復スプレッド (pips)")
    parser.add_argument("--commission", type=float, default=0.1,  help="往復手数料 (pips)")
    parser.add_argument("--slippage",   type=float, default=0.5,  help="スリッページ (pips)")
    args = parser.parse_args()

    # Paths
    inp     = Path(args.csv)
    mpath   = Path(args.model)
    rpath   = Path(args.report)
    curve_p = Path(args.curve_out)

    # Checks
    for p in [inp, mpath]:
        if not p.exists():
            print(f"[ERROR] ファイルが見つかりません: {p}")
            return
    rpath.parent.mkdir(parents=True, exist_ok=True)
    curve_p.parent.mkdir(parents=True, exist_ok=True)

    # Load data & model
    df = pd.read_csv(inp)
    model = joblib.load(mpath)

    # Load feature list
    feat_json = mpath.with_name(mpath.stem + "_feature_cols.json")
    with open(feat_json, 'r', encoding='utf-8') as f:
        features = json.load(f)

    # Prepare containers
    equity = []
    cum = 0.0

    # For each row, simulate trade based on TP/SL
    for idx in range(len(df)-1):
        entry_price = df.at[idx, "open"]  # 次バーの実オープン、または df.at[idx,"close"]
        tp_price  = entry_price + args.tp_pips * 0.0001
        sl_price  = entry_price - args.sl_pips * 0.0001

        # Look ahead until TP or SL is hit
        hit = None
        for j in range(idx+1, len(df)):
            high = df.at[j, "high"]
            low  = df.at[j, "low"]
            if high >= tp_price:
                hit = args.tp_pips
                break
            if low <= sl_price:
                hit = -args.sl_pips
                break
        if hit is None:
            # Neither hit: close at last close
            exit_price = df.at[len(df)-1, "close"]
            hit = (exit_price - entry_price) / 0.0001

        # Adjust for spread, commission, slippage
        net_pips = hit - args.spread - args.commission - args.slippage
        cum += net_pips
        equity.append(cum)

    # Save equity curve plot
    plt.figure(figsize=(8,4))
    plt.plot(equity, linewidth=1)
    plt.title("資産曲線 (実TP/SL+スプレッド/手数料/スリッページ考慮)")
    plt.xlabel("トレード番号")
    plt.ylabel("累積獲得 pips")
    plt.tight_layout()
    plt.savefig(curve_p)
    plt.close()

    # Compute metrics
    cumulative_pips = cum
    running_max     = np.maximum.accumulate(equity)
    drawdowns       = running_max - equity
    max_dd          = float(np.max(drawdowns))

    print(f"[INFO] 累積獲得pips: {cumulative_pips:.2f}")
    print(f"[INFO] 最大ドローダウン: {max_dd:.2f}")

    # Report JSON
    report = {
        "cumulative_pips": cumulative_pips,
        "max_drawdown":    max_dd,
        "n_trades":        len(equity),
        "tp_pips":         args.tp_pips,
        "sl_pips":         args.sl_pips,
        "spread":          args.spread,
        "commission":      args.commission,
        "slippage":        args.slippage
    }
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] バックテストレポートを保存: {rpath}")

if __name__ == "__main__":
    main()
