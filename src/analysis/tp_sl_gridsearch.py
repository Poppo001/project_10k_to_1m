import os
import subprocess
import pandas as pd

# --- パラメータ候補リスト ---
TP_list = [10, 20, 30]
SL_list = [10, 20, 30]

# --- ファイルパス ---
FEATURE_CSV = "/content/drive/MyDrive/project_10k_to_1m_data/processed/feat_USDJPY_H1_1000000bars_TECH.csv"
LABEL_DIR   = "/content/drive/MyDrive/project_10k_to_1m_data/processed/"
RESULTS = []

for tp in TP_list:
    for sl in SL_list:
        label_csv = f"{LABEL_DIR}labeled_USDJPY_H1_1000000bars_TP{tp}SL{sl}.csv"
        # --- ラベル生成 ---
        cmd_label = [
            "python", "src/data/label_gen.py",
            "--file", FEATURE_CSV,
            "--tp", str(tp),
            "--sl", str(sl),
            "--out", label_csv
        ]
        print(f"[INFO] TP={tp} SL={sl} でラベル生成")
        subprocess.run(cmd_label, check=True)

        # --- 学習 & バックテスト ---
        cmd_train = [
            "python", "src/models/train_model.py",
            "--file", label_csv,
            "--model_out", f"{LABEL_DIR}xgb_model_TP{tp}SL{sl}.pkl",
            "--report_out", f"{LABEL_DIR}report_TP{tp}SL{sl}.json"
        ]
        print(f"[INFO] TP={tp} SL={sl} で学習・検証")
        subprocess.run(cmd_train, check=True)

        # --- レポートから結果抽出 ---
        report_path = f"{LABEL_DIR}report_TP{tp}SL{sl}.json"
        if os.path.exists(report_path):
            report = pd.read_json(report_path, typ="series")
            RESULTS.append({
                "TP": tp, "SL": sl,
                "獲得pips": report.get("pips", None),
                "勝率": report.get("accuracy", None),
                "シャープレシオ": report.get("sharpe", None)
            })

# --- 結果をDataFrameで保存・表示 ---
df_res = pd.DataFrame(RESULTS)
df_res = df_res.sort_values("シャープレシオ", ascending=False)
print(df_res)
df_res.to_csv(f"{LABEL_DIR}tp_sl_gridsearch_results.csv", index=False)
print("[INFO] TP/SL自動検証 完了")
