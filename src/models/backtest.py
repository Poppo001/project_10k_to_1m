# src/models/backtest.py

import pandas as pd
import argparse, pickle, json
from pathlib import Path
import matplotlib.pyplot as plt
from utils.common import load_config, resolve_data_root

plt.rcParams["font.family"] = "MS Gothic"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--bars", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_data_root(cfg)

    # 入力ファイル探索
    proc_dir = root / "processed" / args.symbol / args.timeframe
    selfeat = sorted(proc_dir.glob(f"selfeat_{args.symbol}_{args.timeframe}_{args.bars}_*.csv"))[-1]
    ts = selfeat.stem.split("_")[-1]
    model_pkl = proc_dir / f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_{ts}.pkl"
    feats_json = proc_dir / f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_{ts}_features.json"

    # 読み込み
    df = pd.read_csv(selfeat, parse_dates=["time"])
    with open(model_pkl, "rb") as f:
        model = pickle.load(f)
    with open(feats_json, "r") as f:
        features = json.load(f)

    # 推論＆シミュレーション
    df["pred"] = model.predict_proba(df[features])[:, 1]
    df["pos"] = (df["pred"] > 0.5).astype(int)
    df["ret"] = df["return"] * df["pos"]
    df["equity"] = df["ret"].cumsum()

    # 出力
    out_png = proc_dir / f"backtest_{args.symbol}_{args.timeframe}_{args.bars}_{ts}.png"
    df[["time", "equity"]].plot(x="time", title="Equity Curve")
    plt.savefig(out_png)
    print(f"✔ Backtest chart saved to: {out_png}")

if __name__ == "__main__":
    main()
