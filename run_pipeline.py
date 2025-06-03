import subprocess
import sys
import datetime
from pathlib import Path

# ========= 設定値 =========
# MT5 からダウンロード済みの CSV をそのまま使う場合のパス
mt5_data_dir = Path("data/MT5_OHLCV")
symbol     = "USDJPY"
timeframe  = "H1"
bars       = 100000  # 実際に存在するファイル名に合わせてください

# 例: "USDJPY_H1_100000.csv"
raw_csv = mt5_data_dir / f"{symbol}_{timeframe}_{bars}.csv"
if not raw_csv.exists():
    print(f"⚠️ 入力データが見つかりません: {raw_csv}")
    sys.exit(1)

now       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
base_name = f"{symbol}_{timeframe}_{bars}_{now}"

# 各出力先ディレクトリ（既に存在するものとする）
proc_dir   = Path("data/processed")
model_dir  = Path("models")
report_dir = Path("reports")
tools_dir  = Path("tools")

# TP/SL 値（お好みで変更可）
tp = 30
sl = 30

# ====================================================
# ステップ1: 特徴量生成
# ====================================================
feat_csv = proc_dir / f"feat_{base_name}.csv"
print(f"[INFO] 特徴量生成: {raw_csv} → {feat_csv}")
subprocess.run([
    sys.executable, "src/feature_gen_full.py",
    "--csv", str(raw_csv),
    "--out", str(feat_csv)
], check=True)

# ====================================================
# ステップ2: ラベル生成
#   ※ ここを --csv ではなく --file に変更
# ====================================================
label_csv = proc_dir / f"labeled_{base_name}.csv"
print(f"[INFO] ラベル生成: {feat_csv} → {label_csv}")
subprocess.run([
    sys.executable, "src/label_gen.py",
    "--file", str(feat_csv),
    "--tp", str(tp),
    "--sl", str(sl),
    "--out", str(label_csv)
], check=True)

# ====================================================
# ステップ3: 動的特徴量選択
# ====================================================
sel_feat_csv = proc_dir / f"selfeat_{base_name}.csv"
selected_feats_json = proc_dir / f"selected_feats_{base_name}.json"
print(f"[INFO] 動的特徴量選択: {label_csv} → {sel_feat_csv} / {selected_feats_json}")
subprocess.run([
    sys.executable, "src/auto_feature_selection.py",
    "--csv", str(label_csv),
    "--out", str(sel_feat_csv),
    "--feat_out", str(selected_feats_json)
], check=True)

# ====================================================
# ステップ4: モデル学習
# ====================================================
model_pkl         = model_dir / f"xgb_model_{base_name}.pkl"
feature_cols_json = model_dir / f"xgb_model_{base_name}_feature_cols.json"
model_report_json = report_dir / f"model_report_{base_name}.json"
print(f"[INFO] モデル学習: {sel_feat_csv} → {model_pkl} / {feature_cols_json} / {model_report_json}")
subprocess.run([
    sys.executable, "src/train_model.py",
    "--file", str(sel_feat_csv),
    "--model_out", str(model_pkl),
    "--feature_cols_out", str(feature_cols_json),
    "--report", str(model_report_json)
], check=True)

# ====================================================
# ステップ5: モデル評価
# ====================================================
eval_report_json = report_dir / f"eval_report_{base_name}.json"
print(f"[INFO] モデル評価: {sel_feat_csv}, {model_pkl} → {eval_report_json}")
subprocess.run([
    sys.executable, "src/evaluate_model.py",
    "--csv", str(sel_feat_csv),
    "--model", str(model_pkl),
    "--out", str(eval_report_json)
], check=True)

# ====================================================
# ステップ6: バックテスト
# ====================================================
bt_report_json = report_dir / f"backtest_{base_name}.json"
bt_curve_png   = report_dir / f"backtest_curve_{base_name}.png"
print(f"[INFO] バックテスト: {sel_feat_csv}, {model_pkl} → {bt_report_json} / {bt_curve_png}")
subprocess.run([
    sys.executable, "src/backtest.py",
    "--csv", str(sel_feat_csv),
    "--model", str(model_pkl),
    "--report", str(bt_report_json),
    "--curve_out", str(bt_curve_png)
], check=True)

# ====================================================
# ステップ7: 成果物自動整理
# ====================================================
print(f"[INFO] 成果物自動整理を実行: {tools_dir / 'auto_organize.py'}")
subprocess.run([
    sys.executable, str(tools_dir / "auto_organize.py"),
], check=True)

print("\n✅ 全パイプライン完了！")
print(f"  ・最新成果物: {bt_report_json} などを `{report_dir}/` 以下に出力")
print("  ・データ・モデル・レポートも自動で整理されました")
