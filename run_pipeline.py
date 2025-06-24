#!/usr/bin/env python3
# run_pipeline.py

import argparse
import subprocess
import sys
from pathlib import Path
from src.utils.common import load_config, resolve_path

def run(cmd: list):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run Phases 1–4")
    parser.add_argument(
        "--phase",
        default=None,
        help="実行するフェーズ範囲を指定 (例: 1, 1-3, 2-4；未指定は全フェーズ)"
    )
    args = parser.parse_args()

    cfg = load_config()
    # ディレクトリパスを config.yaml から解決
    raw_dir    = resolve_path(cfg["mt5_data_dir"], cfg)
    proc_dir   = resolve_path(cfg["processed_dir"], cfg)
    model_dir  = resolve_path(cfg["model_dir"], cfg)
    report_dir = resolve_path(cfg["report_dir"], cfg)

    symbol    = cfg["symbol"]
    timeframe = cfg["timeframe"]
    bars      = cfg["bars"]

    # 実行フェーズの決定
    if args.phase:
        if "-" in args.phase:
            start, end = map(int, args.phase.split("-"))
            phases = list(range(start, end+1))
        else:
            phases = [int(x) for x in args.phase.split(",")]
    else:
        phases = cfg.get("run_phases", [1,2,3,4])

    # Phase1: 特徴量生成 → ラベル付与 → 特徴量選択
    if 1 in phases:
        # 生データCSVをタイムスタンプ付きで最新取得
        raw_candidates = list(raw_dir.glob(f"*_{symbol}_{timeframe}_{bars}.csv"))
        raw_candidates.sort()
        if not raw_candidates:
            print(f"[ERROR] No raw CSV found matching pattern *_{symbol}_{timeframe}_{bars}.csv in {raw_dir}")
            sys.exit(1)
        raw_csv = raw_candidates[-1]
        print(f"[INFO] Using raw CSV: {raw_csv}")

        # 1-1) feature_gen.py
        feat = proc_dir / symbol / timeframe / f"feat_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable, "src/data/feature_gen.py",
            "--csv", str(raw_csv),
            "--out", str(feat)
        ])

        # 1-2) label_gen.py
        lab = proc_dir / symbol / timeframe / f"labeled_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable, "src/data/label_gen.py",
            "--file", str(feat),
            "--tp", str(cfg["label_gen"]["tp"]),
            "--sl", str(cfg["label_gen"]["sl"]),
            "--exclude_before_release", str(cfg["label_gen"]["exclude_before_release"]),
            "--release_exclude_window_mins", str(cfg["label_gen"]["release_exclude_window_mins"]),
            "--out", str(lab)
        ])

        # 1-3) auto_feature_selection.py
        sel = proc_dir / symbol / timeframe / f"selfeat_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable, "src/data/auto_feature_selection.py",
            "--csv", str(lab),
            "--out", str(sel),
            "--window_size", "5000",
            "--step", "500",
            "--top_k", "10"
        ])

    # Phase2: ベースライン構築 (Logistic Regression)
    if 2 in phases:
        lab = proc_dir / symbol / timeframe / f"labeled_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable, "src/models/train_baseline.py",
            "--csv", str(lab)
        ])

    # Phase3: ブースト木モデル学習＋キャリブレーション
    if 3 in phases:
        sel = proc_dir / symbol / timeframe / f"selfeat_{symbol}_{timeframe}_{bars}.csv"
        model_out = model_dir / f"xgb_model_{symbol}_{timeframe}_{bars}.pkl"
        feat_cols = model_dir / f"xgb_model_{symbol}_{timeframe}_{bars}_features.json"
        run([
            sys.executable, "src/models/train_model.py",
            "--file", str(sel),
            "--model_out", str(model_out),
            "--feature_cols_out", str(feat_cols)
        ])

    # Phase4: 簡易バックテストによる評価
    if 4 in phases:
        run([
            sys.executable, "src/models/backtest.py",
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--bars", str(bars)
        ])

if __name__ == "__main__":
    main()
