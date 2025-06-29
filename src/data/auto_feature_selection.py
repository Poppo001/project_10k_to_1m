#!/usr/bin/env python3
# src/data/auto_feature_selection.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap

# ── Notebook／スクリプト両対応の tqdm インポート
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          required=True, help="labeled CSV path")
    parser.add_argument("--out_dir",      required=True, help="output directory")
    parser.add_argument("--out",          required=True, help="selfeat CSV path")
    parser.add_argument("--window_size",  type=int,   default=5000, help="chunk size for SHAP")
    parser.add_argument("--step",         type=int,   default=500,  help="step size between chunks")
    parser.add_argument("--top_k",        type=int,   default=10,   help="number of features to select")
    parser.add_argument("--sample_frac",  type=float, default=1.0,  help="fraction to sample (for speed test)")
    args = parser.parse_args()

    # 1) データ読み込み
    df = pd.read_csv(args.csv, parse_dates=["time"])
    print(f"[INFO] Loading data from: {args.csv}")
    print(f"[INFO] Original rows: {len(df)}")

    # 2) サンプリング（任意）
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampling fraction: {args.sample_frac}")
        print(f"[INFO] After sampling: {len(df)} rows")

    # 特徴量列とラベル列に分離
    feature_cols = [c for c in df.columns if c not in ["time", "label", "future_return"]]
    X = df[feature_cols]
    y = df["label"].values

    # 3) RF 学習
    print("[INFO] Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    rf.fit(X, y)
    print("[INFO] RF trained.")

    # 4) SHAP 配列初期化
    total_rows  = len(df)
    n_features  = len(feature_cols)
    shap_vals   = np.zeros((total_rows, n_features), dtype=float)

    # 5) SHAP 計算
    explainer   = shap.TreeExplainer(rf)
    chunk_size  = args.window_size

    for start in tqdm(
        range(0, total_rows, args.step),
        desc="SHAP calc chunks",
        leave=True,        # ループ後もバーを残す
        mininterval=0.5    # 更新間隔：0.5秒
    ):
        end       = min(start + chunk_size, total_rows)
        batch_X   = X.iloc[start:end]
        batch_shap = explainer.shap_values(batch_X)

        # 二分類時は正クラス側のみ抽出
        if isinstance(batch_shap, list):
            batch_shap = np.array(batch_shap[1])

        shap_vals[start:end, :] = batch_shap

    # 6) 平均絶対 SHAP で重要度ランキング
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    idx_sorted    = np.argsort(-mean_abs_shap)[: args.top_k]
    selected_feats = [feature_cols[i] for i in idx_sorted]

    # 7) 結果出力
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_sel = df[["time", "label"] + selected_feats + ["future_return"]]
    df_sel.to_csv(args.out, index=False)

    print(f"[INFO] Selected top {args.top_k} features: {selected_feats}")
    print(f"[INFO] Selected features saved: {args.out}")


if __name__ == "__main__":
    main()
