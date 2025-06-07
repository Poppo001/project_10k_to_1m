#!/usr/bin/env python3
# run_pipeline_multi.py

import argparse
import yaml
import subprocess
import sys
import datetime
from pathlib import Path

def load_config():
    """
    Load paths and defaults from config.yaml, converting ${…} placeholders to absolute Paths.
    """
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    script_dir = Path(__file__).parent

    def make_abspath(rel_path: str) -> Path:
        text = rel_path.replace("${data_base}", cfg["data_base"])
        text = text.replace("${project_base}", cfg["project_base"])
        return (script_dir / text).resolve()

    project_base   = make_abspath(cfg["project_base"])
    data_base      = make_abspath(cfg["data_base"])
    mt5_data_dir   = make_abspath(cfg["mt5_data_dir"])
    processed_dir  = make_abspath(cfg["processed_dir"])
    model_dir      = make_abspath(cfg["model_dir"])
    report_dir     = make_abspath(cfg["report_dir"])
    tools_dir      = project_base / "tools"

    default_symbol    = cfg.get("symbol")
    default_timeframe = cfg.get("timeframe")
    default_bars      = cfg.get("bars")
    default_tp        = cfg.get("tp", 30)
    default_sl        = cfg.get("sl", 30)

    return {
        "PROJECT_BASE": project_base,
        "DATA_BASE": data_base,
        "MT5_DATA_DIR": mt5_data_dir,
        "PROCESSED_DIR": processed_dir,
        "MODEL_DIR": model_dir,
        "REPORT_DIR": report_dir,
        "TOOLS_DIR": tools_dir,
        "DEFAULT_SYMBOL": default_symbol,
        "DEFAULT_TIMEFRAME": default_timeframe,
        "DEFAULT_BARS": default_bars,
        "DEFAULT_TP": default_tp,
        "DEFAULT_SL": default_sl,
    }

def main():
    # 1. Load config.yaml
    cfg = load_config()
    PROJECT_BASE   = cfg["PROJECT_BASE"]
    MT5_DATA_DIR   = cfg["MT5_DATA_DIR"]
    PROCESSED_DIR  = cfg["PROCESSED_DIR"]
    MODEL_DIR      = cfg["MODEL_DIR"]
    REPORT_DIR     = cfg["REPORT_DIR"]
    TOOLS_DIR      = cfg["TOOLS_DIR"]

    default_symbol    = cfg["DEFAULT_SYMBOL"]
    default_timeframe = cfg["DEFAULT_TIMEFRAME"]
    default_bars      = cfg["DEFAULT_BARS"]
    default_tp        = cfg["DEFAULT_TP"]
    default_sl        = cfg["DEFAULT_SL"]

    # 2. Parse CLI args (optional overrides)
    parser = argparse.ArgumentParser(description="Batch FX pipeline executor")
    parser.add_argument("--symbols",    help="Comma-separated symbols (override config.yaml)")
    parser.add_argument("--timeframes", help="Comma-separated timeframes (override config.yaml)")
    parser.add_argument("--bars_list",  help="Comma-separated bars counts (override config.yaml)")
    parser.add_argument("--tp",   type=int, help=f"TP pips (default {default_tp})")
    parser.add_argument("--sl",   type=int, help=f"SL pips (default {default_sl})")
    args = parser.parse_args()

    symbols    = args.symbols.split(",")    if args.symbols    else [default_symbol]
    timeframes = args.timeframes.split(",") if args.timeframes else [default_timeframe]
    bars_list  = [int(x) for x in args.bars_list.split(",")] if args.bars_list else [default_bars]
    tp = args.tp if args.tp is not None else default_tp
    sl = args.sl if args.sl is not None else default_sl

    # 3. Ensure output folders exist
    for d in [PROCESSED_DIR, MODEL_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 4. Main loop
    for symbol in symbols:
        for timeframe in timeframes:
            for bars in bars_list:
                raw_csv = MT5_DATA_DIR / f"{symbol}_{timeframe}_{bars}.csv"
                if not raw_csv.exists():
                    print(f"⚠️ Raw data not found: {raw_csv}")
                    continue

                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"{symbol}_{timeframe}_{bars}_{now}"

                # Step 1: Feature generation
                feat_csv = PROCESSED_DIR / f"feat_{base_name}.csv"
                if not feat_csv.exists():
                    print(f"[INFO] Generating features: {feat_csv}")
                    subprocess.run([
                        sys.executable,
                        str(PROJECT_BASE / "src" / "data" / "feature_gen_full.py"),
                        "--csv", str(raw_csv),
                        "--out", str(feat_csv)
                    ], check=True)
                else:
                    print(f"[INFO] SKIP: features exist: {feat_csv}")

                # Step 2: Label generation
                label_csv = PROCESSED_DIR / f"labeled_{base_name}.csv"
                if not label_csv.exists():
                    print(f"[INFO] Generating labels: {label_csv}")
                    subprocess.run([
                        sys.executable,
                        str(PROJECT_BASE / "src" / "data" / "label_gen.py"),
                        "--file", str(feat_csv),
                        "--tp", str(tp),
                        "--sl", str(sl),
                        "--out", str(label_csv)
                    ], check=True)
                else:
                    print(f"[INFO] SKIP: labels exist: {label_csv}")

                # Step 3: Auto feature selection
                sel_feat_csv = PROCESSED_DIR / f"selfeat_{base_name}.csv"
                if not sel_feat_csv.exists():
                    print(f"[INFO] Selecting features: {sel_feat_csv}")
                    subprocess.run([
                        sys.executable,
                        str(PROJECT_BASE / "src" / "data" / "auto_feature_selection.py"),
                        "--csv", str(label_csv),
                        "--out_dir", str(PROCESSED_DIR),
                        "--out", str(sel_feat_csv),
                        "--window_size", "5000",
                        "--step", "500",
                        "--top_k", "10"
                    ], check=True)
                else:
                    print(f"[INFO] SKIP: selected features exist: {sel_feat_csv}")

                # Step 4: Train model
                model_pkl = MODEL_DIR / f"xgb_model_{base_name}.pkl"
                feature_cols_json = MODEL_DIR / f"xgb_model_{base_name}_feature_cols.json"
                if not model_pkl.exists():
                    print(f"[INFO] Training model: {model_pkl}")
                    subprocess.run([
                        sys.executable,
                        str(PROJECT_BASE / "src" / "train_model.py"),
                        "--file", str(sel_feat_csv),
                        "--model_out", str(model_pkl),
                        "--feature_cols_out", str(feature_cols_json),
                        "--test_size", "2000"
                    ], check=True)
                else:
                    print(f"[INFO] SKIP: model exists: {model_pkl}")

                # Step 5: Evaluate model
                eval_report = REPORT_DIR / f"eval_report_{base_name}.json"
                if not eval_report.exists():
                    print(f"[INFO] Evaluating model: {eval_report}")
                    subprocess.run([
                        sys.executable,
                        str(PROJECT_BASE / "src" / "evaluate_model.py"),
                        "--csv", str(sel_feat_csv),
                        "--model", str(model_pkl),
                        "--out", str(eval_report),
                        "--test_size", "2000"
                    ], check=True)
                else:
                    print(f"[INFO] SKIP: evaluation report exists: {eval_report}")

                # Step 6: Backtest
                bt_report = REPORT_DIR / f"backtest_{base_name}.json"
                bt_curve  = REPORT_DIR / f"backtest_curve_{base_name}.png"
                if not bt_report.exists() or not bt_curve.exists():
                    print(f"[INFO] Running backtest: {bt_report}, {bt_curve}")
                    subprocess.run([
                        sys.executable,
                        str(PROJECT_BASE / "src" / "backtest.py"),
                        "--csv", str(sel_feat_csv),
                        "--model", str(model_pkl),
                        "--report", str(bt_report),
                        "--curve_out", str(bt_curve)
                    ], check=True)
                else:
                    print(f"[INFO] SKIP: backtest exists: {bt_report}, {bt_curve}")

                # Step 7: Auto-organize
                organize_log = REPORT_DIR / f"organize_{base_name}.log"
                if not organize_log.exists():
                    print("[INFO] Auto-organizing outputs")
                    subprocess.run([
                        sys.executable,
                        str(TOOLS_DIR / "auto_organize.py"),
                    ], check=True)
                    organize_log.write_text("OK")
                else:
                    print(f"[INFO] SKIP: auto-organize log exists: {organize_log}")

                print(f"✅ Completed pipeline for {symbol} {timeframe} {bars}")

    print("\n--- All pipelines completed! ---")

if __name__ == "__main__":
    main()
