#!/usr/bin/env python3
# src/data/auto_feature_selection.py

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier
import shap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="labeled CSV path")
    parser.add_argument("--out", required=True, help="output selfeat CSV path")
    parser.add_argument("--window_size", type=int, default=5000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["time"])
    features = [c for c in df.columns if c not in ("time","label","future_return")]
    X = df[features]
    y = df["label"]

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)[1]
    importances = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(importances)[-args.top_k:]
    sel_feats = [features[i] for i in idx]

    out_df = df[["time","label"]+sel_feats+["future_return"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[INFO] Selected features saved: {args.out}")

if __name__ == "__main__":
    main()
