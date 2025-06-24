#!/usr/bin/env python3
# src/data/auto_feature_selection.py

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import shap
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Auto feature selection with SHAP")
    parser.add_argument("--csv", required=True, help="labeled CSV path")
    parser.add_argument("--out", required=True, help="output selfeat CSV path")
    parser.add_argument("--window_size", type=int, default=5000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample_frac", type=float, default=1.0,
                        help="Data sampling fraction for speed (0<fr<=1)")
    args = parser.parse_args()

    print("[INFO] Loading data...", file=sys.stderr)
    df = pd.read_csv(args.csv, parse_dates=["time"])
    print(f"[INFO] Rows loaded: {len(df)}", file=sys.stderr)

    # 必要な列だけ
    features = [c for c in df.columns if c not in ("time", "label", "future_return")]
    X_full = df[features]
    y_full = df["label"].values

    # サンプル抽出
    if 0 < args.sample_frac < 1.0:
        n_sample = int(len(df) * args.sample_frac)
        print(f"[INFO] Sampling {n_sample} rows ({args.sample_frac*100:.1f}%) for speed", file=sys.stderr)
        sample_idx = np.random.choice(len(df), size=n_sample, replace=False)
        X = X_full.iloc[sample_idx]
        y = y_full[sample_idx]
    else:
        X = X_full
        y = y_full
    print(f"[INFO] Data shape for model: {X.shape}", file=sys.stderr)

    # ランダムフォレスト学習
    print("[INFO] Training RandomForestClassifier...", file=sys.stderr)
    start = time.time()
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X, y)
    print(f"[INFO] RF trained in {time.time()-start:.1f}s", file=sys.stderr)

    # SHAP 値計算
    print("[INFO] Calculating SHAP values...", file=sys.stderr)
    start = time.time()
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_vals = explainer.shap_values(X)[1]
    print(f"[INFO] SHAP calculated in {time.time()-start:.1f}s", file=sys.stderr)

    # 重要度算出
    print("[INFO] Computing feature importances...", file=sys.stderr)
    importances = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(importances)[-args.top_k:]
    sel_feats = [features[i] for i in idx]
    print(f"[INFO] Selected top {args.top_k} features: {sel_feats}", file=sys.stderr)

    # 出力 DataFrame 作成
    out_df = df[["time", "label"] + sel_feats + ["future_return"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[INFO] Selected features saved: {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
