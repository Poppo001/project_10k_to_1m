#!/usr/bin/env python3
# src/models/backtest.py

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Colab / ローカル共通で src/utils を参照できるように追加
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
sys.path.insert(0, str(project_root / "src"))

from utils.common import load_config, resolve_data_root, get_latest_file


def calc_equity_curve(df: pd.DataFrame, model_dict: dict) -> pd.DataFrame:
    X      = df.drop(columns=["time","label","future_return"])
    raw    = model_dict["model"].predict_proba(X)[:,1]
    proba  = model_dict["ir"].transform(raw)
    pred   = (proba > model_dict["threshold"]).astype(int)

    dfc           = df.copy()
    dfc["pred"]   = pred
    dfc["correct"]= (pred == dfc["label"]).astype(int)
    dfc["ret"]    = np.where(pred==1, dfc["future_return"], 0)
    dfc["equity"] = dfc["ret"].cumsum()
    return dfc


def plot_equity_curve(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10,5))
    plt.plot(df["time"], df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Pips")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--bars",      required=True)
    args = parser.parse_args()

    print(f"▶ symbol={args.symbol}, timeframe={args.timeframe}, bars={args.bars}")

    cfg       = load_config()
    data_root = resolve_data_root(cfg)

    proc_dir = data_root / "processed" / args.symbol / args.timeframe

    # 1) feature CSV
    pat       = f"selfeat_{args.symbol}_{args.timeframe}_{args.bars}*.csv"
    feat_path = get_latest_file(proc_dir, pat)
    if feat_path is None:
        print(f"[ERROR] No feature CSV matching '{pat}' in {proc_dir}")
        return
    df = pd.read_csv(feat_path, parse_dates=["time"])

    # 2) model & feature cols
    model_dir = data_root / "processed" / "models"
    pkl_pat   = f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}.pkl"
    json_pat  = f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_features.json"
    model_path= get_latest_file(model_dir, pkl_pat)
    feats_path= get_latest_file(model_dir, json_pat)
    if model_path is None or feats_path is None:
        print("[ERROR] Model or feature JSON not found")
        return
    model_dict = joblib.load(model_path)
    features   = json.load(feats_path.open(encoding="utf-8"))

    df = df[["time","label"] + features + ["future_return"]]

    # 3) equity curve
    df_res = calc_equity_curve(df, model_dict)
    final_eq  = df_res["equity"].iloc[-1]
    win_rate  = df_res["correct"].mean()
    max_dd    = (df_res["equity"].cummax() - df_res["equity"]).max()
    sharpe    = (df_res["ret"].mean() / df_res["ret"].std()
                 if df_res["ret"].std()!=0 else 0)

    print(f"✔ Final Equity : {final_eq:.2f} pips")
    print(f"✔ Win Rate     : {win_rate:.2%}")
    print(f"✔ Max Drawdown : {max_dd:.2f} pips")
    print(f"✔ Sharpe Ratio : {sharpe:.2f}")

    # 4) save
    base    = f"{args.symbol}_{args.timeframe}_{args.bars}"
    png_out = proc_dir / f"equity_{base}.png"
    json_out= proc_dir / f"report_{base}.json"
    plot_equity_curve(df_res, png_out)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "final_equity":  final_eq,
            "win_rate":      win_rate,
            "max_drawdown":  max_dd,
            "sharpe_ratio":  sharpe
        }, f, indent=2)

    print(f"✔ Saved curve: {png_out}")
    print(f"✔ Saved report: {json_out}")


if __name__=="__main__":
    main()
