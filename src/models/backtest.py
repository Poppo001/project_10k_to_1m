#!/usr/bin/env python3
# src/models/backtest.py

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.common import load_config, resolve_data_root, get_latest_file

plt.rcParams['font.family'] = 'MS Gothic'


def calc_equity_curve(df: pd.DataFrame, model_dict: dict) -> pd.DataFrame:
    """
    DataFrame (time, label, future_return, features...) と
    model_dict = {"model": XGBClassifier, "ir": IsotonicRegression, "threshold": float}
    を受け取り、Equity 曲線用の DataFrame を返す
    """
    X = df.drop(columns=["time", "label", "future_return"])
    y = df["label"].values
    # 予測確率 → 補正 → 閾値判定
    raw_proba = model_dict["model"].predict_proba(X)[:, 1]
    proba     = model_dict["ir"].transform(raw_proba)
    pred      = (proba > model_dict["threshold"]).astype(int)

    dfr                = df.copy()
    dfr["pred"]        = pred
    dfr["correct"]     = (pred == y).astype(int)
    dfr["ret"]         = np.where(pred == 1, dfr["future_return"], 0)
    dfr["equity"]      = dfr["ret"].cumsum()
    return dfr


def plot_equity_curve(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Pips (累積)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    required=True, help="通貨ペア")
    parser.add_argument("--timeframe", required=True, help="時間足")
    parser.add_argument("--bars",      required=True, help="バー数")
    args = parser.parse_args()

    print(f"\n▶ symbol={args.symbol}, timeframe={args.timeframe}, bars={args.bars}")

    # config & data root
    cfg       = load_config()
    data_root = resolve_data_root(cfg)

    # processed ディレクトリ
    proc_dir = data_root / "processed" / args.symbol / args.timeframe

    # ── 1) selfeat CSV 読み込み（上書き運用：タイムスタンプなし or 最新付き両対応）
    patterns = [f"selfeat_{args.symbol}_{args.timeframe}_{args.bars}*.csv"]
    feat_path = get_latest_file(proc_dir, patterns)
    if feat_path is None:
        print(f"[ERROR] No feature CSV in {proc_dir} matching {patterns}")
        return

    df = pd.read_csv(feat_path, parse_dates=["time"])

    # ── 2) モデル＆特徴量 JSON 読み込み
    model_dir = data_root / "processed" / "models"
    pkl_pat   = f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}.pkl"
    json_pat  = f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_features.json"

    model_path = get_latest_file(model_dir, pkl_pat)
    feats_path = get_latest_file(model_dir, json_pat)
    if model_path is None or feats_path is None:
        print("[ERROR] Model or features JSON not found")
        return

    model_dict = joblib.load(model_path)
    with open(feats_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    # 必要カラムだけ抽出
    df = df[["time", "label"] + features + ["future_return"]]

    # ── 3) Equity 曲線計算
    df_result = calc_equity_curve(df, model_dict)

    # 指標計算
    final_equity = df_result["equity"].iloc[-1]
    win_rate     = df_result["correct"].mean()
    max_dd       = (df_result["equity"].cummax() - df_result["equity"]).max()
    sharpe       = (
        df_result["ret"].mean() / df_result["ret"].std()
        if df_result["ret"].std() != 0 else 0
    )

    print(f"✔ Final Equity : {final_equity:.2f} pips")
    print(f"✔ Win Rate     : {win_rate:.2%}")
    print(f"✔ Max Drawdown : {max_dd:.2f} pips")
    print(f"✔ Sharpe Ratio : {sharpe:.2f}\n")

    # ── 4) 結果ファイル出力（processed 上書き運用）
    out_base = f"{args.symbol}_{args.timeframe}_{args.bars}"
    png_out  = proc_dir / f"equity_{out_base}.png"
    json_out = proc_dir / f"report_{out_base}.json"

    plot_equity_curve(df_result, png_out)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "final_equity":  final_equity,
            "win_rate":      win_rate,
            "max_drawdown":  max_dd,
            "sharpe_ratio":  sharpe
        }, f, indent=2)

    print(f"✔ Saved equity curve: {png_out}")
    print(f"✔ Saved report      : {json_out}")


if __name__ == "__main__":
    main()
