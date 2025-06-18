# src/models/backtest.py

import sys
from pathlib import Path

# ✅ src フォルダをパスに追加（common.py を見つけるため）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.common import load_config, resolve_data_root

# src/models/backtest.py

import sys
from pathlib import Path

# ✅ パス解決
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import load_config, resolve_data_root

import pandas as pd
import argparse, pickle, json
import matplotlib.pyplot as plt

# ✅ 日本語フォント（Windows環境用）
plt.rcParams["font.family"] = "MS Gothic"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--bars", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_data_root(cfg)

    # === データ読み込み ===
    proc_dir = root / "processed" / args.symbol / args.timeframe

    # 最新の特徴量CSVを探す
    selfeat = sorted(proc_dir.glob(f"selfeat_{args.symbol}_{args.timeframe}_{args.bars}_*.csv"))[-1]
    ts = selfeat.stem.split("_")[-1]

    # モデルと特徴量列ファイル
    model_pkl = proc_dir / f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_{ts}.pkl"
    feats_json = proc_dir / f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_{ts}_features.json"

    # 読み込み
    df = pd.read_csv(selfeat, parse_dates=["time"])
    with open(model_pkl, "rb") as f:
        model = pickle.load(f)
    with open(feats_json, "r") as f:
        features = json.load(f)

    # === 推論・バックテストロジック ===
    df["pred_prob"] = model.predict_proba(df[features])[:, 1]
    df["pos"] = (df["pred_prob"] > 0.5).astype(int)
    df["ret"] = df["return"] * df["pos"]
    df["equity"] = df["ret"].cumsum()

    # === グラフ出力 ===
    out_png = proc_dir / f"backtest_{args.symbol}_{args.timeframe}_{args.bars}_{ts}.png"
    df[["time", "equity"]].plot(x="time", title=f"Equity Curve: {args.symbol}-{args.timeframe}")
    plt.savefig(out_png)
    plt.close()

    print(f"✔ Backtest chart saved to: {out_png}")

if __name__ == "__main__":
    main()
