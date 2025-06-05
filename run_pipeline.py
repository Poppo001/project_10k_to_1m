import yaml
import os
import subprocess
import sys
import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# 1. config.yaml の読み込み
# ─────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ─────────────────────────────────────────────
# 2. パスを Path() に渡すだけ
# ─────────────────────────────────────────────
PROJECT_BASE   = Path(cfg["project_base"])
DATA_BASE      = Path(cfg["data_base"])
MT5_DATA_DIR   = Path(cfg["mt5_data_dir"])
PROCESSED_DIR  = Path(cfg["processed_dir"])
MODEL_DIR      = Path(cfg["model_dir"])
REPORT_DIR     = Path(cfg["report_dir"])
TOOLS_DIR      = PROJECT_BASE / "tools"

symbol    = cfg.get("symbol", "USDJPY")
timeframe = cfg.get("timeframe", "H1")
bars      = cfg.get("bars", 100000)
tp        = cfg.get("tp", 30)
sl        = cfg.get("sl", 30)

# ─────────────────────────────────────────────
# 3. 必要なフォルダが存在するかチェック／作成
# ─────────────────────────────────────────────
for folder in [MT5_DATA_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR]:
    if not folder.exists():
        print(f"[WARNING] フォルダが存在しないため作成します: {folder}")
        folder.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 4. パイプライン実行
# ─────────────────────────────────────────────
now       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
base_name = f"{symbol}_{timeframe}_{bars}_{now}"

# (1) MT5生データ
raw_csv = MT5_DATA_DIR / f"{symbol}_{timeframe}_{bars}.csv"
if not raw_csv.exists():
    print(f"[ERROR] 入力データが見つかりません: {raw_csv}")
    sys.exit(1)

# (2) 特徴量生成
feat_csv = PROCESSED_DIR / f"feat_{base_name}.csv"
print(f"[INFO] 特徴量生成: {raw_csv} → {feat_csv}")
subprocess.run([
    sys.executable,
    str(PROJECT_BASE / "src" / "feature_gen_full.py"),
    "--csv", str(raw_csv),
    "--out", str(feat_csv)
], check=True)

# (3) ラベル生成
label_csv = PROCESSED_DIR / f"labeled_{base_name}.csv"
print(f"[INFO] ラベル生成: {feat_csv} → {label_csv}")
subprocess.run([
    sys.executable,
    str(PROJECT_BASE / "src" / "label_gen.py"),
    "--file", str(feat_csv),
    "--tp", str(tp),
    "--sl", str(sl),
    "--out", str(label_csv)
], check=True)

# (4) 動的特徴量選択
sel_feat_csv       = PROCESSED_DIR / f"selfeat_{base_name}.csv"
selected_feats_json = PROCESSED_DIR / f"selected_feats_{base_name}.json"
print(f"[INFO] 動的特徴量選択: {label_csv} → {sel_feat_csv} / {selected_feats_json}")
subprocess.run([
    sys.executable,
    str(PROJECT_BASE / "src" / "auto_feature_selection.py"),
    "--csv", str(label_csv),
    "--out", str(sel_feat_csv),
    "--feat_out", str(selected_feats_json)
], check=True)

# (5) モデル学習
model_pkl         = MODEL_DIR / f"xgb_model_{base_name}.pkl"
feature_cols_json = MODEL_DIR / f"xgb_model_{base_name}_feature_cols.json"
model_report_json = REPORT_DIR / f"model_report_{base_name}.json"
print(f"[INFO] モデル学習: {sel_feat_csv} → {model_pkl} / {feature_cols_json} / {model_report_json}")
subprocess.run([
    sys.executable,
    str(PROJECT_BASE / "src" / "train_model.py"),
    "--file", str(sel_feat_csv),
    "--model_out", str(model_pkl),
    "--feature_cols_out", str(feature_cols_json),
    "--report", str(model_report_json)
], check=True)

# (6) モデル評価
eval_report_json = REPORT_DIR / f"eval_report_{base_name}.json"
print(f"[INFO] モデル評価: {sel_feat_csv}, {model_pkl} → {eval_report_json}")
subprocess.run([
    sys.executable,
    str(PROJECT_BASE / "src" / "evaluate_model.py"),
    "--csv", str(sel_feat_csv),
    "--model", str(model_pkl),
    "--out", str(eval_report_json)
], check=True)

# (7) バックテスト
bt_report_json = REPORT_DIR / f"backtest_{base_name}.json"
bt_curve_png   = REPORT_DIR / f"backtest_curve_{base_name}.png"
print(f"[INFO] バックテスト: {sel_feat_csv}, {model_pkl} → {bt_report_json} / {bt_curve_png}")
subprocess.run([
    sys.executable,
    str(PROJECT_BASE / "src" / "backtest.py"),
    "--csv", str(sel_feat_csv),
    "--model", str(model_pkl),
    "--report", str(bt_report_json),
    "--curve_out", str(bt_curve_png)
], check=True)

# (8) 成果物整理
print(f"[INFO] 成果物自動整理を実行: {TOOLS_DIR / 'auto_organize.py'}")
subprocess.run([
    sys.executable,
    str(TOOLS_DIR / "auto_organize.py"),
], check=True)

print("\n✅ 全パイプライン完了！")
print(f"  ・最新成果物: {bt_report_json} を `{REPORT_DIR}` 以下に出力")
print("  ・データ・モデル・レポートも自動で整理されました")
