# src/data/auto_feature_selection_v2_fixed.py

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap

# tqdmを環境に合わせて読み込み
def get_tqdm():
    if "ipykernel" in sys.modules:
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm
    else:
        from tqdm import tqdm
    return tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window_size", type=int, default=5000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    args = parser.parse_args()

    # データ読み込み
    df = pd.read_csv(args.csv)
    print(f"[INFO] Loading data: {args.csv} ({len(df)} rows)")

    # サンプリング
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled: {len(df)} rows (frac={args.sample_frac})")

    # 特徴量・ラベル設定
    feature_cols = [c for c in df.columns if c not in ["time", "label", "future_return"]]
    X = df[feature_cols]
    y = df["label"].values

    # モデル学習
    print("[INFO] Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    rf.fit(X, y)
    print("[INFO] RF trained.")

    total_rows = len(df)
    n_features = len(feature_cols)
    n_classes = rf.n_classes_
    shap_vals = np.zeros((total_rows, n_features), dtype=float)

    # SHAP Explainer初期化（approximate=True時はmodel_output='raw'のみ対応）
    print("[INFO] Initializing SHAP TreeExplainer (approximate=True, model_output='raw')...")
    explainer = shap.TreeExplainer(
        rf,
        model_output="raw",  # raw出力（approximate=Trueと組み合わせ）
        approximate=True
    )

    tqdm = get_tqdm()
    # チャンク自動切替
    if total_rows <= args.window_size:
        starts = [0]
        print(f"[INFO] SHAP run as single chunk ({total_rows} rows)")
    else:
        starts = list(range(0, total_rows, args.step))
        print(f"[INFO] SHAP run in {len(starts)} chunks (window={args.window_size}, step={args.step})")

    # SHAP計算ループ
    for start in tqdm(starts, desc="SHAP chunks", mininterval=0.5):
        end = min(start + args.window_size, total_rows)
        batch = X.iloc[start:end]
        sv = explainer.shap_values(batch)

        # 形状を(batch, features)に整形
        if isinstance(sv, list):
            arr = np.asarray(sv[1])
        elif isinstance(sv, np.ndarray):
            if sv.ndim == 3:
                if sv.shape[0] == n_classes:
                    arr = sv[1]
                elif sv.shape[1] == n_classes:
                    arr = sv[:, 1, :]
                else:
                    arr = sv[:, :, 1]
            elif sv.ndim == 2:
                if sv.shape == (end - start, n_features):
                    arr = sv
                elif sv.shape == (n_features, n_classes):
                    arr = np.repeat(sv[:, 1][np.newaxis, :], end - start, axis=0)
                elif sv.shape == (n_classes, n_features):
                    arr = np.repeat(sv[1][np.newaxis, :], end - start, axis=0)
                else:
                    raise ValueError(f"[ERROR] SHAP shape: {sv.shape}")
            else:
                raise ValueError(f"[ERROR] Unsupported SHAP ndim: {sv.ndim}")
        else:
            raise ValueError(f"[ERROR] Unsupported SHAP output type: {type(sv)}")

        shap_vals[start:end, :] = arr

    # 特徴量選出
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    top_idx = np.argsort(-mean_abs)[:args.top_k]
    selected = [feature_cols[i] for i in top_idx]

    # 結果出力
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cols_to_output = ["time", "label"] + selected
    if "future_return" in df.columns:
        cols_to_output.append("future_return")

    df_sel = df[cols_to_output]
    df_sel.to_csv(args.out, index=False)

    print(f"[INFO] Top {args.top_k} features: {selected}")
    print(f"[INFO] Output saved: {args.out}")


if __name__ == "__main__":
    main()
