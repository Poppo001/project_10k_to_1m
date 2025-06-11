#!/usr/bin/env python3
# run_pipeline_multi.py

import argparse
import yaml
import subprocess
import sys
import datetime
from pathlib import Path

def load_config():
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    script_dir = Path(__file__).parent

    def abspath(p):
        return (script_dir / p.replace("${data_base}", cfg["data_base"])
                              .replace("${project_base}", cfg["project_base"])
               ).resolve()

    return {
        "PROJECT_BASE":  abspath(cfg["project_base"]),
        "MT5_DATA_DIR":  abspath(cfg["mt5_data_dir"]),
        "PROCESSED_DIR": abspath(cfg["processed_dir"]),
        "MODEL_DIR":     abspath(cfg["model_dir"]),
        "REPORT_DIR":    abspath(cfg["report_dir"]),
        "TOOLS_DIR":     abspath(cfg["project_base"]) / "tools",
        # デフォルトTP/SLをここで読み込む
        "TP":            cfg.get("tp", 30),
        "SL":            cfg.get("sl", 30),
        "SPREAD":        cfg.get("spread", 0.2),
        "COMMISSION":    cfg.get("commission", 0.1),
        "SLIPPAGE":      cfg.get("slippage", 0.5),
    }

def main():
    cfg = load_config()
    PB = cfg["PROJECT_BASE"]
    RAW_DIR   = cfg["MT5_DATA_DIR"]
    PROC_DIR  = cfg["PROCESSED_DIR"]
    MODEL_DIR = cfg["MODEL_DIR"]
    REP_DIR   = cfg["REPORT_DIR"]
    TOOLS     = cfg["TOOLS_DIR"]

    # コマンドラインで override できるようにしてもよいですが、ここは config.yaml 固定
    tp = cfg["TP"]
    sl = cfg["SL"]
    spread     = cfg["SPREAD"]
    commission = cfg["COMMISSION"]
    slippage   = cfg["SLIPPAGE"]

    # フォルダ作成
    for d in [PROC_DIR, MODEL_DIR, REP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    symbols    = [cfg.get("symbol")]
    timeframes = [cfg.get("timeframe")]
    bars_list  = [cfg.get("bars")]

    for symbol in symbols:
        for timeframe in timeframes:
            for bars in bars_list:
                raw_csv = RAW_DIR / f"{symbol}_{timeframe}_{bars}.csv"
                if not raw_csv.exists():
                    print(f"⚠️ Raw not found: {raw_csv}")
                    continue

                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base = f"{symbol}_{timeframe}_{bars}_{ts}"

                # 1) Feature gen
                feat = PROC_DIR / f"feat_{base}.csv"
                if not feat.exists():
                    print(f"[INFO] Generating features: {feat}")
                    subprocess.run([
                        sys.executable, str(PB/"src"/"data"/"feature_gen_full.py"),
                        "--csv", str(raw_csv),
                        "--out", str(feat)
                    ], check=True)

                # 2) Label gen
                lab = PROC_DIR / f"labeled_{base}.csv"
                if not lab.exists():
                    print(f"[INFO] Generating labels (TP={tp}, SL={sl}): {lab}")
                    subprocess.run([
                        sys.executable, str(PB/"src"/"data"/"label_gen.py"),
                        "--file", str(feat),
                        "--tp", str(tp), "--sl", str(sl),
                        "--out", str(lab)
                    ], check=True)

                # 3) Auto feature selection
                sel = PROC_DIR / f"selfeat_{base}.csv"
                if not sel.exists():
                    print(f"[INFO] Selecting features: {sel}")
                    subprocess.run([
                        sys.executable, str(PB/"src"/"data"/"auto_feature_selection.py"),
                        "--csv", str(lab),
                        "--out_dir", str(PROC_DIR),
                        "--out", str(sel),
                        "--window_size", "5000",
                        "--step", "500",
                        "--top_k", "10"
                    ], check=True)

                # 4) Train model
                model_pkl = MODEL_DIR / f"xgb_model_{base}.pkl"
                feat_json = MODEL_DIR / f"xgb_model_{base}_feature_cols.json"
                if not model_pkl.exists():
                    print(f"[INFO] Training model: {model_pkl}")
                    subprocess.run([
                        sys.executable, str(PB/"src"/"data"/"train_model.py"),
                        "--file", str(sel),
                        "--model_out", str(model_pkl),
                        "--feature_cols_out", str(feat_json),
                        "--test_size", "2000"
                    ], check=True)

                # 5) Evaluate model
                eval_json = REP_DIR / f"eval_report_{base}.json"
                if not eval_json.exists():
                    print(f"[INFO] Evaluating model: {eval_json}")
                    subprocess.run([
                        sys.executable, str(PB/"src"/"data"/"evaluate_model.py"),
                        "--csv", str(sel),
                        "--model", str(model_pkl),
                        "--out", str(eval_json),
                        "--test_size", "2000"
                    ], check=True)

                # 6) Backtest
                bt_json  = REP_DIR / f"backtest_{base}.json"
                bt_curve = REP_DIR / f"backtest_curve_{base}.png"
                if not bt_json.exists() or not bt_curve.exists():
                    print(f"[INFO] Running backtest: TP={tp}, SL={sl}, Spread={spread}, Comm={commission}, Slip={slippage}")
                    subprocess.run([
                        sys.executable, str(PB/"src"/"backtest.py"),
                        "--csv", str(sel),
                        "--model", str(model_pkl),
                        "--report", str(bt_json),
                        "--curve_out", str(bt_curve),
                        "--tp_pips",   str(tp),
                        "--sl_pips",   str(sl),
                        "--spread",    str(spread),
                        "--commission",str(commission),
                        "--slippage",  str(slippage)
                    ], check=True)

                # 7) Organize
                log = REP_DIR / f"organize_{base}.log"
                if not log.exists():
                    print("[INFO] Auto-organizing outputs")
                    subprocess.run([
                        sys.executable, str(TOOLS / "auto_organize.py"),
                    ], check=True)
                    log.write_text("OK")

                print(f"✅ {symbol} {timeframe} {bars}: Completed")

    print("\n--- All done! ---")

if __name__ == "__main__":
    main()
