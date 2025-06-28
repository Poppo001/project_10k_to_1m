#!/usr/bin/env python3
# src/data/auto_feature_selection.py

import argparse
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
from tqdm import tqdm  # プログレスバー用

def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto Feature Selection via RandomForest + SHAP"
    )
    parser.add_argument("--csv",         required=True,
                        help="ラベル付き CSV ファイルパス")
    parser.add_argument("--out_dir",     required=True,
                        help="出力ディレクトリ")
    parser.add_argument("--out",         required=True,
                        help="選択特徴量付き CSV 出力パス")
    parser.add_argument("--window_size", type=int, default=5000,
                        help="チャンク学習ウィンドウサイズ")
    parser.add_argument("--step",        type=int, default=500,
                        help="チャンクステップ（SHAP描画用）")
    parser.add_argument("--top_k",       type=int, default=10,
                        help="選択する特徴量数")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                        help="サンプリング比率（0<α≤1）")
    return parser.parse_args()

def main():
    args     = parse_args()
    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"[INFO] Rows loaded: {total_rows}")

    # 特徴量行列とラベル
    if "time" in df.columns:
        df_feat = df.drop(columns=["time"])
    else:
        df_feat = df.copy()
    y = df_feat["label"]
    X = df_feat.drop(columns=["label", "future_return"])

    # サンプリング
    if args.sample_frac < 1.0:
        print(f"[INFO] Sampling fraction: {args.sample_frac}")
        X, _, y, _ = train_test_split(
            X, y,
            train_size=args.sample_frac,
            stratify=y,
            random_state=42
        )
        total_rows = len(X)
        print(f"[INFO] After sampling: {total_rows} rows")

    # 学習/検証分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=min(2000, total_rows//10), random_state=42
    )

    # モデル学習
    print("[INFO] Training RandomForestClassifier...")
    t0    = time.time()
    model = RandomForestClassifier(n_estimators=100,
                                   random_state=42,
                                   n_jobs=-1)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[INFO] RF trained in {elapsed:.1f}s")

    # SHAP 計算（tqdmでプログレスバー表示）
    print("[INFO] Starting SHAP calculation...")
    explainer  = shap.TreeExplainer(model)
    n_features = X.shape[1]
    shap_vals  = np.zeros((total_rows, n_features), dtype=float)
    chunk_size = args.window_size

    for start in tqdm(range(0, total_rows, chunk_size),
                      desc="SHAP calc chunks"):
        end    = min(start + chunk_size, total_rows)
        batch  = X.iloc[start:end]

        # raw_shap の取り出し
        raw_shap = explainer.shap_values(batch)

        # positive クラスに対応する SHAP 値を取得
        if isinstance(raw_shap, list):
            # list [class0, class1]
            batch_shap = raw_shap[1]
        else:
            # numpy array
            if raw_shap.ndim == 3:
                # shape == (n_rows, n_features, n_classes)
                batch_shap = raw_shap[:, :, 1]
            else:
                # shape == (n_rows, n_features)
                batch_shap = raw_shap

        # 形状チェック
        rows = end - start
        if batch_shap.shape != (rows, n_features):
            raise ValueError(
                f"[ERROR] SHAP array has wrong shape {batch_shap.shape}, "
                f"expected ({rows}, {n_features})"
            )

        shap_vals[start:end, :] = batch_shap

    # 特徴量重要度の算出
    mean_abs_shap   = np.mean(np.abs(shap_vals), axis=0)
    feat_importance = pd.Series(mean_abs_shap, index=X.columns)
    selected_feats  = feat_importance.nlargest(args.top_k).index.tolist()
    print(f"[INFO] Selected top {args.top_k} features: {selected_feats}")

    # 出力データフレーム作成および保存
    df_out  = df[["time", "label"] + selected_feats + ["future_return"]]
    out_csv = Path(args.out).resolve()
    df_out.to_csv(out_csv, index=False)
    print(f"[INFO] Selected features saved: {out_csv}")

if __name__ == "__main__":
    main()
