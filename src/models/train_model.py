# src/models/train_model.py

import pandas as pd
import argparse, json, pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from pathlib import Path
from utils.common import load_config, resolve_data_root

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--bars", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config()
    root = resolve_data_root(cfg)

    # 特徴量CSV取得
    csv_dir = root / "processed" / args.symbol / args.timeframe
    input_csv = sorted(csv_dir.glob(f"selfeat_{args.symbol}_{args.timeframe}_{args.bars}_*.csv"))[-1]
    df = pd.read_csv(input_csv, parse_dates=["time"])
    df = df.dropna()

    # 学習
    X = df.drop(columns=["time", "label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # 保存
    ts = input_csv.stem.split("_")[-1]
    out_dir = csv_dir
    model_path = out_dir / f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_{ts}.pkl"
    feats_path = out_dir / f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_{ts}_features.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(feats_path, "w") as f:
        json.dump(list(X.columns), f)

    print(f"✔ Model saved to: {model_path}")
    print(f"✔ Features saved to: {feats_path}")

if __name__ == "__main__":
    main()
