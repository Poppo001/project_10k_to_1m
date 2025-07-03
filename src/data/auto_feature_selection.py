# src/data/auto_feature_selection_v7_fastfallback.py

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

# tqdm取得
def get_tqdm():
    if "ipykernel" in sys.modules:
        try:
            from tqdm.notebook import tqdm as nb_tqdm
        except ImportError:
            nb_tqdm = tqdm
        return nb_tqdm
    return tqdm

# SHAP array reshape
def reshape_sv(sv, batch_size, n_features, n_classes):
    if isinstance(sv, list):
        arr = np.asarray(sv[1])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            arr = arr[1] if arr.shape[0] == n_classes else arr[:,1,:]
    if arr.shape != (batch_size, n_features):
        raise ValueError(f"Unexpected SHAP shape: {arr.shape}, expected ({batch_size},{n_features})")
    return arr

# Init worker
def init_worker(rf, n_features, n_classes):
    global explainer, _n_features, _n_classes
    explainer = shap.TreeExplainer(rf)
    _n_features = n_features
    _n_classes = n_classes

# Chunk worker
def compute_chunk(args):
    start, end, X_np = args
    batch_size = end - start
    sv = explainer.shap_values(X_np[start:end])
    arr = reshape_sv(sv, batch_size, _n_features, _n_classes)
    return start, arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.csv}")

    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled {len(df)} rows ({args.sample_frac*100:.1f}%)")

    exclude = ['time','label','future_return','open','high','low','close']
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols]
    y = df['label'].values

    print("[INFO] Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    rf.fit(X, y)
    print("[INFO] RF trained.")

    total = len(X)
    n_features = X.shape[1]
    n_classes = rf.n_classes_

    # Fallback to permutation importance for small frac
    if args.sample_frac >= 0.05:
        print("[INFO] sample_frac>=0.05: using permutation importance")
        perm = permutation_importance(rf, X, y, n_repeats=5, n_jobs=-1, random_state=42)
        imp = perm.importances_mean
        top_idx = np.argsort(-imp)[:args.top_k]
        selected = [feature_cols[i] for i in top_idx]
    else:
        # SHAP chunk settings
        workers = args.n_workers or cpu_count()
        window = args.window_size or total
        if window > total:
            window = total
        step = window
        starts = list(range(0, total, step))
        print(f"[INFO] SHAP chunks: window={window}, chunks={len(starts)}, workers={workers}")

        X_np = X.values
        tqdm_cls = get_tqdm()
        results = []
        with Pool(workers, initializer=init_worker, initargs=(rf, n_features, n_classes)) as pool:
            args_list = [(s, min(s+window, total), X_np) for s in starts]
            for start, arr in tqdm_cls(pool.imap_unordered(compute_chunk, args_list), total=len(args_list), desc="SHAP"):
                results.append((start, arr))

        shap_vals = np.zeros((total, n_features))
        for start, arr in results:
            shap_vals[start:start+arr.shape[0],:] = arr
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        top_idx = np.argsort(-mean_abs)[:args.top_k]
        selected = [feature_cols[i] for i in top_idx]

    # Save
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cols_out = ['time','label'] + selected
    if 'future_return' in df.columns:
        cols_out.append('future_return')
    df[cols_out].to_csv(args.out, index=False)
    print(f"[INFO] Selected top {args.top_k} features: {selected}")

if __name__=='__main__':
    main()