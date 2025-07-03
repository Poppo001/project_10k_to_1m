# src/data/auto_feature_selection_v4_fast.py

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
from multiprocessing import cpu_count, Pool
from tqdm import tqdm

# tqdmを環境に合わせて取得
def get_tqdm():
    if "ipykernel" in sys.modules:
        try:
            from tqdm.notebook import tqdm as nb_tqdm
        except ImportError:
            nb_tqdm = tqdm
        return nb_tqdm
    return tqdm

# SHAP出力を(batch_size, n_features)に整形
def reshape_shap(sv, n_classes, n_features, batch_size):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    if not isinstance(sv, np.ndarray):
        raise ValueError(f"Unsupported SHAP type: {type(sv)}")
    if sv.ndim == 3:
        return sv[1] if sv.shape[0] == n_classes else sv[:,1,:]
    if sv.ndim == 2:
        if sv.shape == (batch_size, n_features):
            return sv
        if sv.shape == (n_classes, n_features):
            return np.repeat(sv[1][np.newaxis,:], batch_size, axis=0)
    raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

# ワーカー初期化
def init_worker(rf):
    global explainer
    explainer = shap.TreeExplainer(rf, model_output="raw")

# 各チャンク計算ワーカー
def compute_chunk(args):
    start, end, X_np, n_classes, n_features = args
    sv = explainer.shap_values(X_np[start:end])
    return start, reshape_shap(sv, n_classes, n_features, end-start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window_size", type=int, default=5000)
    parser.add_argument("--step", type=int, default=None,
                        help="未指定でwindow_sizeを使用")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[INFO] Loading: {len(df)} rows from {args.csv}")
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled: {len(df)} rows (frac={args.sample_frac})")

    feature_cols = [c for c in df.columns if c not in ["time","label","future_return"]]
    X = df[feature_cols]
    y = df["label"].values

    print("[INFO] Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    rf.fit(X, y)
    print("[INFO] RF trained.")

    # sample_frac大きい場合はPermutation Importanceで高速処理
    if args.sample_frac >= 0.1:
        print("[INFO] sample_frac>=0.1: using permutation importance fallback")
        perm = permutation_importance(rf, X, y, n_repeats=3, n_jobs=-1, random_state=42)
        imp_mean = perm.importances_mean
        top_idx = np.argsort(-imp_mean)[:args.top_k]
        selected = [feature_cols[i] for i in top_idx]
    else:
        # SHAPで厳密計算（小サンプル向け）
        total = len(X)
        n_classes = rf.n_classes_
        n_feat = len(feature_cols)
        # 動的チャンク設定
        workers = max(cpu_count()-1,1)
        window = args.window_size
        step = args.step or window
        starts = list(range(0, total, step))
        print(f"[INFO] SHAP chunks: window={window}, step={step}, count={len(starts)}")
        X_np = X.values
        # 並列 or 逐次
        results = []
        tqdm_cls = get_tqdm()
        if workers>1 and len(starts)>1:
            print("[INFO] Parallel SHAP computation...")
            args_list = [(s, min(s+window,total), X_np, n_classes, n_feat) for s in starts]
            with Pool(workers, initializer=init_worker, initargs=(rf,)) as pool:
                for start, arr in tqdm_cls(pool.imap(compute_chunk,args_list), total=len(args_list), desc="SHAP"):
                    results.append((start,arr))
        else:
            print("[INFO] Sequential SHAP computation...")
            explainer = shap.TreeExplainer(rf, model_output="raw")
            for start in tqdm_cls(starts, total=len(starts), desc="SHAP"):
                end = min(start+window, total)
                sv = explainer.shap_values(X_np[start:end])
                arr = reshape_shap(sv, n_classes, n_feat, end-start)
                results.append((start,arr))
        # 結果合成
        _, first = results[0]
        shap_vals = np.zeros((total, first.shape[1]))
        for s,arr in results:
            shap_vals[s:s+arr.shape[0],:] = arr
        mean_abs = np.mean(np.abs(shap_vals),axis=0)
        top_idx = np.argsort(-mean_abs)[:args.top_k]
        selected = [feature_cols[i] for i in top_idx]

    # 出力
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    cols = ["time","label"]+selected
    if "future_return" in df.columns: cols.append("future_return")
    df[cols].to_csv(args.out,index=False)
    print(f"[INFO] Selected features: {selected}")

if __name__ == "__main__":
    main()
