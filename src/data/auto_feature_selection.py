#!/usr/bin/env python3
# src/data/auto_feature_selection.py

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap

# ── 環境に応じて tqdm を使い分け
# Colab の notebook セッションならウィジェットバー、
# それ以外（CLI 実行など）はターミナルバーを使う
if "ipykernel" in sys.modules:
    try:
        from tqdm.notebook import tqdm
    except ImportError:
        from tqdm import tqdm
else:
    from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         required=True, help="labeled CSV path")
    parser.add_argument("--out_dir",     required=True, help="output directory")
    parser.add_argument("--out",         required=True, help="selfeat CSV path")
    parser.add_argument("--window_size", type=int,   default=5000, help="chunk size")
    parser.add_argument("--step",        type=int,   default=500,  help="step size")
    parser.add_argument("--top_k",       type=int,   default=10,   help="select top K")
    parser.add_argument("--sample_frac", type=float, default=1.0,  help="sampling fraction")
    args = parser.parse_args()

    # 1) データロード
    df = pd.read_csv(args.csv, parse_dates=["time"])
    print(f"[INFO] Loading data from: {args.csv}")
    print(f"[INFO] Original rows: {len(df)}")

    # 2) 任意サンプリング
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampling fraction: {args.sample_frac}")
        print(f"[INFO] After sampling: {len(df)} rows")

    # 特徴量列とラベル
    feature_cols = [c for c in df.columns if c not in ["time", "label", "future_return"]]
    X = df[feature_cols]
    y = df["label"].values

    # 3) RF 学習
    print("[INFO] Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    rf.fit(X, y)
    print("[INFO] RF trained.")

    # 4) SHAP 用配列初期化
    total_rows = len(df)
    n_features = len(feature_cols)
    shap_vals = np.zeros((total_rows, n_features), dtype=float)

    # 5) SHAP 計算
    explainer = shap.TreeExplainer(rf)
    for start in tqdm(
        range(0, total_rows, args.step),
        desc="SHAP calc chunks",
        leave=True,
        mininterval=0.5
    ):
        end = min(start + args.window_size, total_rows)
        batch_X = X.iloc[start:end]

        # SHAP 値取得
        batch_shap = explainer.shap_values(batch_X)

        # リスト／3次元 ndarray の両方に対応し、必ず (n_samples, n_features) に
        if isinstance(batch_shap, list):
            batch_shap = np.array(batch_shap[1])          # list[neg, pos]
        elif isinstance(batch_shap, np.ndarray) and batch_shap.ndim == 3:
            batch_shap = batch_shap[1]                    # shape (2, n, m)

        shap_vals[start:end, :] = batch_shap

    # 6) 重要度ランキング
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    top_idx       = np.argsort(-mean_abs_shap)[: args.top_k]
    selected_feats = [feature_cols[i] for i in top_idx]

    # 7) 結果を書き出し
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_sel = df[["time", "label"] + selected_feats + ["future_return"]]
    df_sel.to_csv(args.out, index=False)

    print(f"[INFO] Selected top {args.top_k} features: {selected_feats}")
    print(f"[INFO] Selected features saved: {args.out}")


if __name__ == "__main__":
    main()
