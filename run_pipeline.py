#!/usr/bin/env python3
# run_pipeline.py

import argparse
import subprocess
import sys
from src.utils.common import load_config, resolve_path

def run(cmd: list):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    cfg = load_config()
    # 主要ディレクトリ
    raw_dir   = resolve_path(cfg["mt5_data_dir"], cfg)
    proc_dir  = resolve_path(cfg["processed_dir"], cfg)
    model_dir = resolve_path(cfg["model_dir"], cfg)
    report_dir= resolve_path(cfg["report_dir"], cfg)

    symbol    = cfg["symbol"]
    timeframe = cfg["timeframe"]
    bars      = cfg["bars"]

    phases = cfg.get("run_phases", [1,2,3,4])

    # Phase1: 特徴量・ラベル・自動特徴選択
    if 1 in phases:
        feat = proc_dir / symbol / timeframe / f"feat_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable, "src/data/feature_gen.py",
            "--csv", str(raw_dir / f"{symbol}_{timeframe}_{bars}.csv"),
            "--out", str(feat)
        ])
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
        sel = proc_dir / symbol / timeframe / f"selfeat_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable, "src/data/auto_feature_selection.py",
            "--csv", str(lab),
            "--out", str(sel),
            "--window_size", "5000",
            "--step", "500",
            "--top_k", "10"
        ])

    # Phase2: ベースライン構築
    if 2 in phases:
        lab = proc_dir / symbol / timeframe / f"labeled_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable, "src/models/train_baseline.py",
            "--csv", str(lab)
        ])

    # Phase3: モデル学習
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

    # Phase4: バックテスト
    if 4 in phases:
        run([
            sys.executable, "src/models/backtest.py",
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--bars", str(bars)
        ])

if __name__ == "__main__":
    main()
