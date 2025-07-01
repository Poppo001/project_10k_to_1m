#!/usr/bin/env python3
# src/data/auto_feature_selection.py

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap

# Colab/Notebook vs CLI で tqdm を切り替え
if "ipykernel" in sys.modules:
    try:
        from tqdm.notebook import tqdm
    except ImportError:
        from tqdm import tqdm
else:
    from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Labeled CSV path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--out", required=True, help="Selfeat CSV path")
    parser.add_argument(
        "--window_size", type=int, default=5000,
        help="SHAPバッチサイズ（デフォルト:5000）"
    )
    parser.add_argument(
        "--step", type=int, default=500,
        help="SHAPステップ幅（デフォルト:500）"
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="選択する特徴量数（デフォルト:10）"
    )
    parser.add_argument(
        "--sample_frac", type=float, default=1.0,
        help="サンプリング率（テスト用, デフォルト全量）"
    )
    args = parser.parse_args()

    # 1) データ読み込み
    # time 列は日付として使用しないため、parse_datesを削除
    df = pd.read_csv(args.csv)
    print(f"[INFO] Loading data from: {args.csv}")
    print(f"[INFO] Original rows: {len(df)}")

    # 2) 任意サンプリング
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampling fraction: {args.sample_frac}")
        print(f"[INFO] After sampling: {len(df)} rows")

    # 特徴量列とラベル列
    # future_return が存在しない可能性を考慮し、feature_cols から除外
    feature_cols = [c for c in df.columns if c not in ["time", "label", "future_return"]]
    X = df[feature_cols]
    y = df["label"].values

    # 3) RF 学習
    print("[INFO] Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    rf.fit(X, y)
    print("[INFO] RF trained.")

    total_rows = len(df)
    n_features = len(feature_cols)
    n_classes = rf.n_classes_
    shap_vals = np.zeros((total_rows, n_features), dtype=float)

    # 4) SHAP 計算
    explainer = shap.TreeExplainer(rf)

    for start in tqdm(
        range(0, total_rows, args.step),
        desc="SHAP calc chunks",
        leave=True,
        mininterval=0.5
    ):
        end = min(start + args.window_size, total_rows)
        batch = X.iloc[start:end]

        sv = explainer.shap_values(batch)

        # ── いかなる形状も (batch_size, n_features) に整形 ──
        if isinstance(sv, list):
            # list[class0, class1]
            arr = np.asarray(sv[1])  # 正クラス側

        elif isinstance(sv, np.ndarray):
            if sv.ndim == 3:
                s0, s1, s2 = sv.shape
                # (classes, samples, features)
                if s0 == n_classes and s1 == (end - start) and s2 == n_features:
                    arr = sv[1]  # sv[クラス1] → shape=(batch, features)
                # (samples, classes, features)
                elif s0 == (end - start) and s1 == n_classes and s2 == n_features:
                    arr = sv[:, 1, :]  # sv[:,クラス1,:]
                # (samples, features, classes)
                elif s0 == (end - start) and s1 == n_features and s2 == n_classes:
                    arr = sv[:, :, 1]  # shape=(batch, features)
                else:
                    raise ValueError(
                        f"[ERROR] Unexpected 3D SHAP shape: {sv.shape}. "
                        f"Expected (n_classes, batch_size, n_features), (batch_size, n_classes, n_features), or (batch_size, n_features, n_classes)."
                    )

            elif sv.ndim == 2:
                s0, s1 = sv.shape
                # (samples, features) - 最も一般的な二項分類の出力
                if s0 == (end - start) and s1 == n_features:
                    arr = sv
                # (features, classes) - 特定のexplainerやshapバージョンで発生する可能性
                elif s0 == n_features and s1 == n_classes:
                    rep = sv[:, 1]
                    arr = np.repeat(rep[np.newaxis, :], end - start, axis=0)
                # (classes, features) - 特定のexplainerやshapバージョンで発生する可能性
                elif s0 == n_classes and s1 == n_features:
                    rep = sv[1]
                    arr = np.repeat(rep[np.newaxis, :], end - start, axis=0)
                else:
                    raise ValueError(
                        f"[ERROR] Unexpected 2D SHAP shape: {sv.shape}. "
                        f"Expected (batch_size, n_features), (n_features, n_classes), or (n_classes, n_features)."
                    )

            else:
                raise ValueError(f"[ERROR] Unsupported SHAP ndim: {sv.ndim}")

        else:
            raise ValueError(f"[ERROR] Unsupported SHAP output type: {type(sv)}")

        shap_vals[start:end, :] = arr

    # 5) 平均絶対値で重要度算出→上位 K
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    top_idx = np.argsort(-mean_abs)[: args.top_k]
    selected = [feature_cols[i] for i in top_idx]

    # 6) CSV 出力
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # future_return 列の存在を確認し、存在する場合のみ追加
    cols_to_output = ["time", "label"] + selected
    if "future_return" in df.columns:
        cols_to_output.append("future_return")

    df_sel = df[cols_to_output]
    df_sel.to_csv(args.out, index=False)

    print(f"[INFO] Selected top {args.top_k} features: {selected}")
    print(f"[INFO] Saved selfeat CSV: {args.out}")


if __name__ == "__main__":
    main()