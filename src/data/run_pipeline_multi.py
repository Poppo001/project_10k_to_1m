import subprocess
import sys
import datetime
from pathlib import Path

# 設定リスト（例）
symbols    = ["USDJPY", "EURUSD", "GBPJPY"]
timeframes = ["H1", "M15"]
bars_list  = [100000]

tp = 30
sl = 30

mt5_data_dir = Path("data/MT5_OHLCV")
proc_dir = Path("data/processed")
model_dir = Path("models")
report_dir = Path("reports")
tools_dir = Path("tools")

for symbol in symbols:
    for timeframe in timeframes:
        for bars in bars_list:
            raw_csv = mt5_data_dir / f"{symbol}_{timeframe}_{bars}.csv"
            if not raw_csv.exists():
                print(f"⚠️ データファイルが見つかりません: {raw_csv}")
                continue

            now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"{symbol}_{timeframe}_{bars}_{now}"

            # ---- 以降は個別パイプライン ----
            feat_csv = proc_dir / f"feat_{base_name}.csv"
            subprocess.run([
                sys.executable, "src/feature_gen_full.py",
                "--csv", str(raw_csv),
                "--out", str(feat_csv)
            ], check=True)

            label_csv = proc_dir / f"labeled_{base_name}.csv"
            subprocess.run([
                sys.executable, "src/label_gen.py",
                "--csv", str(feat_csv),
                "--tp", str(tp), "--sl", str(sl),
                "--out", str(label_csv)
            ], check=True)

            sel_feat_csv = proc_dir / f"selfeat_{base_name}.csv"
            selected_feats_json = proc_dir / f"selected_feats_{base_name}.json"
            subprocess.run([
                sys.executable, "src/auto_feature_selection.py",
                "--csv", str(label_csv),
                "--out", str(sel_feat_csv),
                "--feat_out", str(selected_feats_json)
            ], check=True)

            model_pkl = model_dir / f"xgb_model_{base_name}.pkl"
            feature_cols_json = model_dir / f"xgb_model_{base_name}_feature_cols.json"
            model_report_json = report_dir / f"model_report_{base_name}.json"
            subprocess.run([
                sys.executable, "src/train_model.py",
                "--file", str(sel_feat_csv),
                "--model_out", str(model_pkl),
                "--feature_cols_out", str(feature_cols_json),
                "--report", str(model_report_json)
            ], check=True)

            eval_report_json = report_dir / f"eval_report_{base_name}.json"
            subprocess.run([
                sys.executable, "src/evaluate_model.py",
                "--csv", str(sel_feat_csv),
                "--model", str(model_pkl),
                "--out", str(eval_report_json)
            ], check=True)

            bt_report_json = report_dir / f"backtest_{base_name}.json"
            bt_curve_png = report_dir / f"backtest_curve_{base_name}.png"
            subprocess.run([
                sys.executable, "src/backtest.py",
                "--csv", str(sel_feat_csv),
                "--model", str(model_pkl),
                "--report", str(bt_report_json),
                "--curve_out", str(bt_curve_png)
            ], check=True)

            subprocess.run([
                sys.executable, str(tools_dir / "auto_organize.py"),
            ], check=True)

            print(f"✅ {symbol} {timeframe} {bars}: 全処理完了！")

print("\n--- 全シンボル・時間足のパイプライン実行が完了しました！ ---")
