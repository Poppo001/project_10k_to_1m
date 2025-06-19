#!/usr/bin/env python3
# src/models/train_baseline.py

import argparse
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path
import sys

# ── プロジェクトルートを sys.path に追加 ───────────────────
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]   # src/models → src → プロジェクトルート
sys.path.insert(0, str(project_root))

# ── 共通ユーティリティ読み込み ───────────────────────────────
from src.utils.common import load_config

# ── Purged TimeSeries Split（後ほど実装/詳細化） ────────────
# ここでは仮に外部モジュールからインポートする想定です
try:
    from src.utils.purged_tss import PurgedTimeSeriesSplit
except ImportError:
    PurgedTimeSeriesSplit = None

def expected_value(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   tp: float,
                   sl: float) -> float:
    """
    Expected Value (EV) の簡易計算。
    wins: 各予測確率の合計 × TP
    losses: 各 (1 - 予測確率) の合計 × SL
    EV = wins - losses
    """
    wins   = y_proba[y_true == 1].sum() * tp
    losses = (1 - y_proba[y_true == 0]).sum() * sl
    return wins - losses

def main():
    # ── 引数定義 ─────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Phase2 Baseline: LogisticRegression + PurgedTSSplit 評価"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="labeled CSV path (e.g. data/processed/USDJPY/M5/labeled_USDJPY_M5_100000.csv)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="出力結果 JSON パス (未指定なら CSV 階層に baseline_result.json を出力)"
    )
    args = parser.parse_args()

    # ── config 読み込み ─────────────────────────────────────
    cfg = load_config()
    tp_pips = float(cfg.get("tp", 30.0))
    sl_pips = float(cfg.get("sl", 30.0))

    # ── データ読み込み ─────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path, parse_dates=["time"])

    # 特徴量カラムとラベル・未来リターン
    if "label" not in df.columns or "future_return" not in df.columns:
        print("[ERROR] 'label' または 'future_return' 列がありません")
        return

    features = [c for c in df.columns
                if c not in ("time", "label", "future_return")]
    X = df[features]
    y = df["label"].values

    # ── PurgedTSSplit のチェック ────────────────────────────
    if PurgedTimeSeriesSplit is None:
        print("[ERROR] PurgedTimeSeriesSplit が見つかりません。")
        print("→ src/utils/purged_tss.py に実装するか、外部ライブラリを導入してください。")
        return

    # ── 学習／検証ループ ────────────────────────────────────
    pts = PurgedTimeSeriesSplit(n_splits=5, purge=50, embargo=10)
    auc_scores = []
    ev_scores  = []

    for fold, (train_idx, valid_idx) in enumerate(pts.split(X, y, groups=df["time"])):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[valid_idx], y[valid_idx]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        ev  = expected_value(y_val, proba, tp_pips, sl_pips)

        auc_scores.append(auc)
        ev_scores.append(ev)
        print(f"Fold {fold+1}: AUC={auc:.3f}, EV={ev:.2f}")

    # ── 結果集計 ────────────────────────────────────────────
    result = {
        "auc_mean": np.mean(auc_scores),
        "auc_std":  np.std(auc_scores),
        "ev_mean":  np.mean(ev_scores),
        "ev_std":   np.std(ev_scores),
        "tp_pips": tp_pips,
        "sl_pips": sl_pips
    }

    # ── 結果出力 ────────────────────────────────────────────
    output_path = Path(args.output) if args.output else csv_path.parent / "baseline_result.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Baseline result saved to: {output_path}")

if __name__ == "__main__":
    main()
