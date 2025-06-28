#!/usr/bin/env python3
# src/models/train_model.py

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── Colab / ローカル共通で src/utils を参照できるように追加
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]   # project_root/src/models → project_root
sys.path.insert(0, str(project_root / "src"))

from utils.common import load_config, resolve_data_root, get_latest_file


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--symbol",    required=True, help="通貨ペア (例: USDJPY)")
    parser.add_argument("--timeframe", required=True, help="時間足 (例: M5)")
    parser.add_argument("--bars",      required=True, help="バー数 (例: 100000)")
    args = parser.parse_args()

    # config.yaml 読み込み & データルート解決
    cfg       = load_config()
    data_root = resolve_data_root(cfg)

    proc_dir  = data_root / "processed" / args.symbol / args.timeframe
    model_dir = data_root / "processed" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1) selfeat CSV の取得（タイムスタンプなし／付き 両対応）
    patterns  = [f"selfeat_{args.symbol}_{args.timeframe}_{args.bars}*.csv"]
    input_csv = get_latest_file(proc_dir, patterns)
    if input_csv is None:
        print(f"[ERROR] No selfeat CSV in {proc_dir} matching {patterns}")
        sys.exit(1)

    # 2) 出力ファイル名（上書き運用：タイムスタンプなし）
    base_name = f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}"
    model_out = model_dir / f"{base_name}.pkl"
    feat_out  = model_dir / f"{base_name}_features.json"

    # 3) CSV 読み込み
    df    = pd.read_csv(input_csv, parse_dates=["time"])
    feats = [c for c in df.columns if c not in ["time", "label", "future_return"]]
    X, y  = df[feats], df["label"]

    # 4) 学習/検証分割
    test_size = min(2000, len(df)//10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 5) モデル学習
    print(f"[INFO] Training XGBoost ({len(feats)} features, test_size={test_size})")
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # 6) Isotonic 補正
    prob_train = model.predict_proba(X_train)[:,1]
    ir         = IsotonicRegression(out_of_bounds="clip")
    ir.fit(prob_train, y_train)

    # 7) 閾値最適化（EV最大化）
    prob_test = ir.transform(model.predict_proba(X_test)[:,1])
    returns   = df["future_return"].iloc[-len(prob_test):].values
    best_ev, best_thr = -np.inf, 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        pred = (prob_test > thr).astype(int)
        ev   = np.sum(returns * pred)
        if ev > best_ev:
            best_ev, best_thr = ev, thr

    # 8) 保存
    joblib.dump({"model":model, "ir":ir, "threshold":best_thr}, model_out)
    with open(feat_out, "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Model saved        : {model_out}")
    print(f"[INFO] Feature cols saved : {feat_out}")
    print(f"[INFO] Optimal threshold  : {best_thr:.3f}, EV: {best_ev:.2f}")


if __name__ == "__main__":
    main()
