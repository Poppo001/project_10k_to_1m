#!/usr/bin/env python3
# run_pipeline.py

import argparse
import subprocess
import sys
import os
from pathlib import Path

def is_colab() -> bool:
    """Colab上（Driveマウント直下）で動作しているか判定。"""
    return os.getcwd().startswith("/content/drive/MyDrive")

def run(cmd: list):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run Phases 1–4")
    parser.add_argument(
        "--phase",
        default=None,
        help="実行するフェーズ(例:1,1-3,2-4;未指定は全フェーズ)"
    )
    args = parser.parse_args()

    # まず常に config.yaml を読み込んでパラメータ取得
    from src.utils.common import load_config, resolve_path
    cfg = load_config()

    # パラメータ
    symbol    = cfg["symbol"]
    timeframe = cfg["timeframe"]
    bars      = cfg["bars"]
    # label_gen パラメータ
    tp    = cfg["label_gen"]["tp"]
    sl    = cfg["label_gen"]["sl"]
    excl  = cfg["label_gen"]["exclude_before_release"]
    excl_w = cfg["label_gen"]["release_exclude_window_mins"]

    # 実行フェーズのデフォルト
    if args.phase:
        if "-" in args.phase:
            s, e = args.phase.split("-")
            phases = list(range(int(s), int(e) + 1))
        else:
            phases = [int(x) for x in args.phase.split(",")]
    else:
        phases = cfg.get("run_phases", [1,2,3,4])

    # 実行環境でコード＆データのルートを決定
    if is_colab():
        DRIVE     = Path("/content/drive/MyDrive")
        CODE_DIR  = DRIVE / "project_10k_to_1m"
        DATA_DIR  = DRIVE / "project_10k_to_1m_data"
        raw_dir    = DATA_DIR / "raw"
        proc_dir   = DATA_DIR / "processed"
        model_dir  = DATA_DIR / "processed" / "models"
        report_dir = DATA_DIR / "processed" / "reports"
    else:
        raw_dir    = resolve_path(cfg["mt5_data_dir"],   cfg)
        proc_dir   = resolve_path(cfg["processed_dir"],  cfg)
        model_dir  = resolve_path(cfg["model_dir"],      cfg)
        report_dir = resolve_path(cfg["report_dir"],     cfg)
        CODE_DIR   = Path().resolve()

    # ── Phase1: 生データ取得 → 特徴量生成 → ラベル生成 → 特徴量選択 ──
    if 1 in phases:
        # タイムスタンプ付き or なし 両対応で CSV 検索
        candidates = sorted(raw_dir.glob(f"*_{symbol}_{timeframe}_{bars}.csv"))
        if not candidates:
            candidates = sorted(raw_dir.glob(f"{symbol}_{timeframe}_{bars}.csv"))
        if not candidates:
            print(f"[ERROR] No raw CSV in {raw_dir} matching")
            sys.exit(1)
        raw_csv = candidates[-1]
        print(f"[INFO] Using raw CSV: {raw_csv}")

        # 1-1) 特徴量生成
        feat = proc_dir / symbol / timeframe / f"feat_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable,
            str(CODE_DIR/"src"/"data"/"feature_gen.py"),
            "--csv", str(raw_csv),
            "--out", str(feat)
        ])

        # 1-2) ラベル生成
        lab = proc_dir / symbol / timeframe / f"labeled_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable,
            str(CODE_DIR/"src"/"data"/"label_gen.py"),
            "--file", str(feat),
            "--tp",   str(tp),
            "--sl",   str(sl),
            "--exclude_before_release",  str(excl),
            "--release_exclude_window_mins", str(excl_w),
            "--out",  str(lab)
        ])

        # 1-3) 自動特徴量選択
        sel = proc_dir / symbol / timeframe / f"selfeat_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable,
            str(CODE_DIR/"src"/"data"/"auto_feature_selection.py"),
            "--csv", str(lab),
            "--out", str(sel),
            "--window_size", "5000",
            "--step", "500",
            "--top_k", "10"
        ])

    # ── Phase2: ベースライン構築 (Logistic Regression) ──
    if 2 in phases:
        lab = proc_dir / symbol / timeframe / f"labeled_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable,
            str(CODE_DIR/"src"/"models"/"train_baseline.py"),
            "--csv", str(lab)
        ])

    # ── Phase3: ブースト木モデル学習 ──
    if 3 in phases:
        sel = proc_dir / symbol / timeframe / f"selfeat_{symbol}_{timeframe}_{bars}.csv"
        model_out = model_dir / f"xgb_model_{symbol}_{timeframe}_{bars}.pkl"
        feat_cols = model_dir / f"xgb_model_{symbol}_{timeframe}_{bars}_features.json"
        run([
            sys.executable,
            str(CODE_DIR/"src"/"models"/"train_model.py"),
            "--file", str(sel),
            "--model_out", str(model_out),
            "--feature_cols_out", str(feat_cols)
        ])

    # ── Phase4: バックテスト ──
    if 4 in phases:
        run([
            sys.executable,
            str(CODE_DIR/"src"/"models"/"backtest.py"),
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--bars", str(bars)
        ])

    print("\n--- All done! ---")

if __name__ == "__main__":
    main()
