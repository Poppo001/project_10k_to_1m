#!/usr/bin/env python3
# src/models/train_model.py

import argparse
from pathlib import Path
import joblib
import yaml
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

def load_config():
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def get_latest_file(dir_path: Path, patterns: list):
    """
    patterns のいずれかにマッチするファイルを glob で探し、
    ソートして最新を返す。なければ None。
    """
    files = []
    for pat in patterns:
        files.extend(dir_path.glob(pat))
    return sorted(files)[-1] if files else None

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--symbol",    required=True, help="通貨ペア")
    parser.add_argument("--timeframe", required=True, help="時間足")
    parser.add_argument("--bars",      required=True, help="バー数")
    args = parser.parse_args()

    cfg = load_config()
    data_root = Path(cfg["data_base_local"] if "data_base_local" in cfg else cfg["data_base"])
    proc_dir = data_root / "processed" / args.symbol / args.timeframe
    model_dir = data_root / "processed" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1) selfeat CSV の取得（タイムスタンプ付き or なし 両対応）
    patterns = [
        f"selfeat_{args.symbol}_{args.timeframe}_{args.bars}_*.csv",
        f"selfeat_{args.symbol}_{args.timeframe}_{args.bars}.csv"
    ]
    input_csv = get_latest_file(proc_dir, patterns)
    if input_csv is None:
        print(f"[ERROR] No selfeat CSV found with patterns {patterns} in {proc_dir}")
        return

    # 2) モデル出力先パス
    timestamp = input_csv.stem.split("_")[-1] if "_" in input_csv.stem else ""
    base_name = f"xgb_model_{args.symbol}_{args.timeframe}_{args.bars}"
    model_out = model_dir / f"{base_name}{'_' + timestamp if timestamp else ''}.pkl"
    feat_out  = model_dir / f"{base_name}{'_' + timestamp if timestamp else ''}_features.json"

    # 3) データ読み込み
    df = pd.read_csv(input_csv, parse_dates=["time"])
    feats = [c for c in df.columns if c not in ["time", "label", "future_return"]]
    X = df[feats]
    y = df["label"]

    # 4) 学習／テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=min(2000, len(df)//10), random_state=42
    )

    # 5) モデル学習
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # 6) Isotonic 補正
    prob_train = model.predict_proba(X_train)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(prob_train, y_train)

    # 7) 阈値 θ の最適化（EV最大化）
    prob_tests = ir.transform(model.predict_proba(X_test)[:, 1])
    returns = df["future_return"].values[-len(prob_tests):]
    best_ev = -np.inf
    best_thr = 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        pred = (prob_tests > thr).astype(int)
        ev = np.sum(returns * pred)
        if ev > best_ev:
            best_ev, best_thr = ev, thr

    # 8) モデル＆特徴量リスト保存
    joblib.dump({"model": model, "ir": ir, "threshold": best_thr}, model_out)
    with open(feat_out, "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Model saved: {model_out}")
    print(f"[INFO] Features saved: {feat_out}")
    print(f"[INFO] Optimal threshold: {best_thr:.3f}, EV: {best_ev:.2f}")

if __name__ == "__main__":
    main()
