# src/data/auto_feature_selection_v3_parallel.py

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map

# tqdmを環境に合わせて取得
def get_tqdm():
    if "ipykernel" in sys.modules:
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm
    else:
        from tqdm import tqdm
    return tqdm

# SHAP出力を(batch_size, n_features)に整形する
def reshape_shap(sv, n_classes, n_features, batch_size):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    if not isinstance(sv, np.ndarray):
        raise ValueError(f"Unsupported SHAP type: {type(sv)}")
    # sv.ndim == 3 or 2
    if sv.ndim == 3:
        #  (classes, samples, features) or (samples, classes, features)
        if sv.shape[0] == n_classes:
            return sv[1]
        return sv[:, 1, :]
    if sv.ndim == 2:
        if sv.shape == (batch_size, n_features):
            return sv
        if sv.shape == (n_classes, n_features):
            return np.repeat(sv[1][np.newaxis, :], batch_size, axis=0)
    raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

# 各チャンクを計算するワーカー
def compute_chunk(args):
    start, end, X_np, n_classes, n_features = args
    batch_size = end - start
    sv = explainer.shap_values(X_np[start:end])
    arr = reshape_shap(sv, n_classes, n_features, batch_size)
    return start, arr

# ワーカープロセス初期化 (TreeExplainerを共有)
def init_worker(rf):
    global explainer
    explainer = shap.TreeExplainer(rf, model_output="raw", approximate=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window_size", type=int, default=5000)
    parser.add_argument("--step", type=int, default=None,
                        help="チャンクステップ幅。未指定時はwindow_sizeを使用")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    args = parser.parse_args()

    # データロード
    df = pd.read_csv(args.csv)
    print(f"[INFO] Loading data: {args.csv} ({len(df)} rows)")

    # サンプリング
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled: {len(df)} rows (frac={args.sample_frac})")

    feature_cols = [c for c in df.columns if c not in ["time","label","future_return"]]
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

    # 並列ワーカー数設定
    n_workers = max(cpu_count() - 1, 1)
    print(f"[INFO] CPU cores for SHAP parallel: {n_workers}")

    # 動的window_sizeとstep設定
    optimal_window = max(args.window_size, total_rows // (n_workers * 2))
    step = args.step or optimal_window
    if total_rows <= optimal_window:
        starts = [0]
        optimal_window = total_rows
        print(f"[INFO] Single chunk mode: window={optimal_window}")
    else:
        starts = list(range(0, total_rows, step))
        print(f"[INFO] Parallel SHAP: window={optimal_window}, step={step}, chunks={len(starts)}")

    # numpy配列化（forkによるメモリ共有想定）
    X_np = X.values

    # プロセスプールでSHAP計算
    print("[INFO] Starting parallel SHAP computation...")
    tqdm = get_tqdm()
    # init_workerでexplainerを各プロセスにセット
    results = process_map(
        compute_chunk,
        [(s, min(s + optimal_window, total_rows), X_np, n_classes, n_features) for s in starts],
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(rf,),
        desc="SHAP parallel chunks"
    )

    # 結果統合
    for start, arr in results:
        shap_vals[start:start + arr.shape[0], :] = arr

    # 上位K特徴量選出
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    top_idx = np.argsort(-mean_abs)[: args.top_k]
    selected = [feature_cols[i] for i in top_idx]

    # 出力
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