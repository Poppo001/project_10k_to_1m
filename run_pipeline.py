#!/usr/bin/env python3
# run_pipeline.py

import argparse
import subprocess
import sys
import os
from pathlib import Path

def is_colab() -> bool:
    """
    Colab上（Driveマウント直下）で動作しているか判定。
    """
    return os.getcwd().startswith("/content/drive/MyDrive")

def run(cmd: list):
    """
    外部スクリプトを実行するユーティリティ
    """
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # 1) 引数パース
    parser = argparse.ArgumentParser(description="Run Phases 1–4")
    parser.add_argument(
        "--phase",
        default=None,
        help="実行するフェーズ(例:1,1-3,2-4; 未指定はconfig.yamlのrun_phases)"
    )
    args = parser.parse_args()

    # 2) 設定ファイル読み込み
    from src.utils.common import load_config, resolve_path
    cfg = load_config()

    # 3) パラメータ取得
    symbol    = cfg["symbol"]
    timeframe = cfg["timeframe"]
    bars      = cfg["bars"]
    tp        = cfg["label_gen"]["tp"]
    sl        = cfg["label_gen"]["sl"]
    excl      = cfg["label_gen"]["exclude_before_release"]
    excl_w    = cfg["label_gen"]["release_exclude_window_mins"]
    default_phases = cfg.get("run_phases", [1,2,3,4])

    # 4) フェーズ指定を解釈
    if args.phase:
        if "-" in args.phase:
            s, e = args.phase.split("-")
            phases = list(range(int(s), int(e) + 1))
        else:
            phases = [int(x) for x in args.phase.split(",")]
    else:
        phases = default_phases

    # 5) 実行環境によるパス決定
    if is_colab():
        DRIVE     = Path("/content/drive/MyDrive")
        CODE_DIR  = DRIVE / "project_10k_to_1m"
        DATA_DIR  = DRIVE / "project_10k_to_1m_data"
        raw_base   = DATA_DIR / "raw"
        proc_dir   = DATA_DIR / "processed"
        model_dir  = DATA_DIR / "processed" / "models"
        report_dir = DATA_DIR / "processed" / "reports"
    else:
        raw_base   = resolve_path(cfg["mt5_data_dir"],   cfg)
        proc_dir   = resolve_path(cfg["processed_dir"],  cfg)
        model_dir  = resolve_path(cfg["model_dir"],      cfg)
        report_dir = resolve_path(cfg["report_dir"],     cfg)
        CODE_DIR   = Path().resolve()

    # ── raw_dir をサブフォルダ化 ──
    raw_dir   = raw_base   / symbol / timeframe
    feat_dir  = proc_dir   / symbol / timeframe

    # Phase1：生データ取得→特徴量→ラベル→特徴量選択
    if 1 in phases:
        # 1-1) 生データ取得は省略（fetch_mt5_ohlcv.py 等を別途実行）
        # 1-2) 特徴量生成
        candidates = sorted(raw_dir.glob(f"*_{symbol}_{timeframe}_{bars}.csv"))
        if not candidates:
            candidates = sorted(raw_dir.glob(f"{symbol}_{timeframe}_{bars}_*.csv"))
        if not candidates:
            print(f"[ERROR] No raw CSV in {raw_dir}")
            sys.exit(1)
        raw_csv = candidates[-1]
        print(f"[INFO] Using raw CSV: {raw_csv}")

        # 1-3) feature_gen.py
        feat_out = feat_dir / f"feat_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable,
            str(CODE_DIR/"src"/"data"/"feature_gen.py"),
            "--csv", str(raw_csv),
            "--out", str(feat_out)
        ])

        # 1-4) label_gen.py
        lab_out = feat_dir / f"labeled_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable,
            str(CODE_DIR/"src"/"data"/"label_gen.py"),
            "--file", str(feat_out),
            "--tp",   str(tp),
            "--sl",   str(sl),
            "--exclude_before_release",      str(excl),
            "--release_exclude_window_mins", str(excl_w),
            "--out",  str(lab_out)
        ])

        # 1-5) auto_feature_selection.py
        sel_out = feat_dir / f"selfeat_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable,
            str(CODE_DIR/"src"/"data"/"auto_feature_selection.py"),
            "--csv",     str(lab_out),
            "--out_dir", str(feat_dir),
            "--out",     str(sel_out),
            "--window_size", "5000",
            "--step",        "500",
            "--top_k",       "10"
        ])

    # Phase2：ベースライン構築
    if 2 in phases:
        lab_csv = feat_dir / f"labeled_{symbol}_{timeframe}_{bars}.csv"
        run([
            sys.executable,
            str(CODE_DIR/"src"/"models"/"train_baseline.py"),
            "--csv", str(lab_csv)
        ])

    # Phase3：ブースト木モデル学習
    if 3 in phases:
        sel_csv   = feat_dir / f"selfeat_{symbol}_{timeframe}_{bars}.csv"
        model_out = model_dir / f"xgb_model_{symbol}_{timeframe}_{bars}.pkl"
        feat_json = model_dir / f"xgb_model_{symbol}_{timeframe}_{bars}_features.json"
        run([
            sys.executable,
            str(CODE_DIR/"src"/"models"/"train_model.py"),
            "--file",             str(sel_csv),
            "--model_out",        str(model_out),
            "--feature_cols_out", str(feat_json)
        ])

    # Phase4：バックテスト
    if 4 in phases:
        run([
            sys.executable,
            str(CODE_DIR/"src"/"models"/"backtest.py"),
            "--symbol",    symbol,
            "--timeframe", timeframe,
            "--bars",      str(bars)
        ])

    print("\n--- All done! ---")

if __name__ == "__main__":
    main()
