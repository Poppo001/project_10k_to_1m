#!/usr/bin/env python3
# src/models/train_baseline.py

import argparse
import json
import subprocess
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import sys

# ── プロジェクトルートをパスに追加 ─────────────────────────
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]   # src/models → src → プロジェクトルート
sys.path.insert(0, str(project_root))

# ── 共通ユーティリティ読み込み ─────────────────────────────
from src.utils.common import load_config

def expected_value(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   tp: float,
                   sl: float) -> float:
    """
    Expected Value (EV) の簡易計算。
    - wins   = 予測確率で「勝ち」と判断したものの合計 × TP
    - losses = (1 - 予測確率) で「負け」と判断したものの合計 × SL
    - EV = wins - losses
    """
    wins   = y_proba[y_true == 1].sum() * tp
    losses = (1 - y_proba[y_true == 0]).sum() * sl
    return wins - losses

def main():
    # ── 引数定義 ─────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Phase2 Baseline: LogisticRegression + TimeSeriesSplit 評価"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="特徴量CSV または ラベル付きCSV のパス"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="出力先 JSON パス（未指定時は CSV フォルダ直下に baseline_result.json）"
    )
    args = parser.parse_args()

    # ── config.yaml から TP/SL を取得 ─────────────────────────
    cfg     = load_config()
    tp_pips = float(cfg.get("tp", 30.0))
    sl_pips = float(cfg.get("sl", 30.0))

    # ── 入力CSV読み込み（特徴量CSV or ラベル付きCSV） ─────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV が見つかりません: {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=["time"])

    # ── ラベル付きCSVでなければ label_gen.py を実行 ────────────────
    if "label" not in df.columns or "future_return" not in df.columns:
        labeled_path = csv_path.parent / f"labeled_{csv_path.stem}.csv"
        print(f"[INFO] ラベル付きCSVを自動生成: {labeled_path}")
        subprocess.run([
            sys.executable,
            str(project_root / "src" / "data" / "label_gen.py"),
            "--file", str(csv_path),
            "--tp",   str(tp_pips),
            "--sl",   str(sl_pips),
            "--out",  str(labeled_path)
        ], check=True)
        csv_path = labeled_path
        df = pd.read_csv(csv_path, parse_dates=["time"])

    # ── 必須列チェック ─────────────────────────────────────
    if "label" not in df.columns or "future_return" not in df.columns:
        print("[ERROR] 'label' または 'future_return' 列がありません")
        return

    # ── 特徴量とラベルの切り出し ─────────────────────────────
    features = [c for c in df.columns if c not in ("time", "label", "future_return")]
    X = df[features]
    y = df["label"].values

    # ── 時系列分割クロスバリデーション ─────────────────────
    tss = TimeSeriesSplit(n_splits=5)
    auc_scores = []
    ev_scores  = []

    for fold, (train_idx, valid_idx) in enumerate(tss.split(X), start=1):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[valid_idx], y[valid_idx]

        # ロジスティック回帰モデル学習
        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr, y_tr)

        # 予測確率と評価指標
        proba = model.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, proba)
        ev    = expected_value(y_val, proba, tp_pips, sl_pips)

        auc_scores.append(auc)
        ev_scores.append(ev)
        print(f"Fold {fold}: AUC={auc:.3f}, EV={ev:.2f}")

    # ── 結果集計 ───────────────────────────────────────────
    result = {
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std":  float(np.std(auc_scores)),
        "ev_mean":  float(np.mean(ev_scores)),
        "ev_std":   float(np.std(ev_scores)),
        "tp_pips":  tp_pips,
        "sl_pips":  sl_pips
    }

    # ── JSON 出力 ───────────────────────────────────────────
    output_path = Path(args.output) if args.output else csv_path.parent / "baseline_result.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[INFO] Baseline result saved to: {output_path}")

if __name__ == "__main__":
    main()
