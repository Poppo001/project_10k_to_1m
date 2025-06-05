# run_pipeline_multi.py

import argparse
import yaml
import subprocess
import sys
import datetime
from pathlib import Path

def load_config():
    """
    project_base, data_base, default symbol/timeframe/bars/tp/sl を
    相対パス→絶対パス化して読み込む。
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
        "DEFAULT_SL": default_sl
    }

def main():
    # 1. 設定読み込み（config.yaml）
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

    # 2. コマンドライン引数（すべてオプション化）
    parser = argparse.ArgumentParser(description="複数シンボル・時間足の一括パイプライン実行")
    parser.add_argument("--symbols", help="カンマ区切りの通貨ペアリスト（未指定時は config.yaml の symbol）")
    parser.add_argument("--timeframes", help="カンマ区切りの時間足リスト（未指定時は config.yaml の timeframe）")
    parser.add_argument("--bars_list", help="カンマ区切りのバー数リスト（未指定時は config.yaml の bars）")
    parser.add_argument("--tp", type=int, help=f"テイクプロフィット(pips)、未指定時は config.yaml の tp={default_tp}")
    parser.add_argument("--sl", type=int, help=f"ストップロス(pips)、未指定時は config.yaml の sl={default_sl}")
    args = parser.parse_args()

    # 3. 引数 or config の値を最終決定
    symbols    = args.symbols.split(",") if args.symbols else [default_symbol]
    timeframes = args.timeframes.split(",") if args.timeframes else [default_timeframe]
    bars_list  = [int(x) for x in args.bars_list.split(",")] if args.bars_list else [default_bars]
    tp = args.tp if args.tp is not None else default_tp
    sl = args.sl if args.sl is not None else default_sl

    # 4. 出力フォルダがなければ作成
    for d in [PROCESSED_DIR, MODEL_DIR, REPORT_DIR]:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)

    # 5. 各シンボル・時間足・バー数ごとにパイプラインを回す
    for symbol in symbols:
        for timeframe in timeframes:
            for bars in bars_list:
                raw_csv = MT5_DATA_DIR / f"{symbol}_{timeframe}_{bars}.csv"
                if not raw_csv.exists():
                    print(f"⚠️ データファイルが見つかりません: {raw_csv}")
                    continue

                now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = f"{symbol}_{timeframe}_{bars}_{now}"

                # ステップ1: 特徴量生成
                feat_csv = PROCESSED_DIR / f"feat_{base_name}.csv"
                subprocess.run([
                    sys.executable,
                    str(PROJECT_BASE / "src" / "data" / "feature_gen_full.py"),
                    "--csv", str(raw_csv),
                    "--out", str(feat_csv)
                ], check=True)

                # ステップ2: ラベル生成
                label_csv = PROCESSED_DIR / f"labeled_{base_name}.csv"
                subprocess.run([
                    sys.executable,
                    str(PROJECT_BASE / "src" / "data" / "label_gen.py"),
                    "--file", str(feat_csv),
                    "--tp", str(tp),
                    "--sl", str(sl),
                    "--out", str(label_csv)
                ], check=True)

                # ステップ3: 動的特徴量選択
                sel_feat_csv = PROCESSED_DIR / f"selfeat_{base_name}.csv"
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

                # ステップ4: モデル学習
                model_pkl = MODEL_DIR / f"xgb_model_{base_name}.pkl"
                feature_cols_json = MODEL_DIR / f"xgb_model_{base_name}_feature_cols.json"
                subprocess.run([
                    sys.executable,
                    str(PROJECT_BASE / "src" / "train_model.py"),
                    "--file", str(sel_feat_csv),
                    "--model_out", str(model_pkl),
                    "--feature_cols_out", str(feature_cols_json),
                    "--test_size", "2000"
                ], check=True)

                # ステップ5: モデル評価
                eval_report_json = REPORT_DIR / f"eval_report_{base_name}.json"
                subprocess.run([
                    sys.executable,
                    str(PROJECT_BASE / "src" / "evaluate_model.py"),
                    "--csv", str(sel_feat_csv),
                    "--model", str(model_pkl),
                    "--out", str(eval_report_json),
                    "--test_size", "2000"
                ], check=True)

                # ステップ6: バックテスト
                bt_report_json = REPORT_DIR / f"backtest_{base_name}.json"
                bt_curve_png   = REPORT_DIR / f"backtest_curve_{base_name}.png"
                subprocess.run([
                    sys.executable,
                    str(PROJECT_BASE / "src" / "backtest.py"),
                    "--csv", str(sel_feat_csv),
                    "--model", str(model_pkl),
                    "--report", str(bt_report_json),
                    "--curve_out", str(bt_curve_png)
                ], check=True)

                # ステップ7: 成果物自動整理
                subprocess.run([
                    sys.executable,
                    str(TOOLS_DIR / "auto_organize.py"),
                ], check=True)

                print(f"✅ {symbol} {timeframe} {bars}: 全処理完了！")

    print("\n--- 全シンボル・時間足のパイプライン実行が完了しました！ ---")

if __name__ == "__main__":
    main()
