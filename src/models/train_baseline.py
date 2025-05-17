"""
src/models/train_baseline.py
----------------------------
・加工済み CSV を読み込み
・Purged TimeSeriesSplit でロジスティック回帰（ベースライン）
・AUC と EV を表示
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"


def purged_split(n_splits=5, purge_size=24):
    """TimeSeriesSplit + 後続 purge"""
    tss = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tss.split(np.arange(len(df))):
        if len(train_idx) > purge_size:
            train_idx = train_idx[:-purge_size]
        yield train_idx, test_idx


def expected_value(y_true, proba, tp_pips=30, sl_pips=30):
    p_tp = proba
    ev = (p_tp * tp_pips) - ((1 - p_tp) * sl_pips)
    decided = ev > 0
    return ev[decided].mean() if decided.any() else 0.0


def main(csv_name: str):
    global df  # purged_split で参照
    df = pd.read_csv(PROC_DIR / csv_name, parse_dates=["time"])
    df = df[df["label"] != -1]

    X = df.drop(columns=["time", "label"])
    y = df["label"]

    aucs, evs = [], []
    for train_idx, test_idx in purged_split():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, solver="lbfgs"),
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]

        aucs.append(roc_auc_score(y.iloc[test_idx], proba))
        evs.append(expected_value(y.iloc[test_idx].values, proba))

    print(f"AUC  mean: {np.mean(aucs):.3f}")
    print(f"EV   mean: {np.mean(evs):.3f}  pips/取引")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="processed CSV filename")
    args = ap.parse_args()
    main(args.file)
