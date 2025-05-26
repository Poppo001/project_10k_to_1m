# src/analysis/optuna_tp_sl_search.py

import optuna
import subprocess
import json
import os

# -- 共通パス/設定 --
FEATURE_CSV = "/content/drive/MyDrive/project_10k_to_1m_data/processed/feat_USDJPY_H1_1000000bars_TECH.csv"
LABEL_DIR = "/content/drive/MyDrive/project_10k_to_1m_data/processed/"

def objective(trial):
    tp = trial.suggest_int("tp", 5, 40, step=5)   # 5〜40pipsを5刻みで
    sl = trial.suggest_int("sl", 5, 40, step=5)
    label_csv = f"{LABEL_DIR}labeled_USDJPY_H1_1000000bars_TP{tp}SL{sl}.csv"

    # ラベル生成
    cmd_label = [
        "python", "src/data/label_gen.py",
        "--file", FEATURE_CSV,
        "--tp", str(tp),
        "--sl", str(sl),
        "--out", label_csv
    ]
    subprocess.run(cmd_label, check=True)

    # 学習＆評価
    model_out = f"{LABEL_DIR}xgb_model_TP{tp}SL{sl}.pkl"
    report_out = f"{LABEL_DIR}report_TP{tp}SL{sl}.json"
    cmd_train = [
        "python", "src/models/train_model.py",
        "--file", label_csv,
        "--model_out", model_out,
        "--report_out", report_out
    ]
    subprocess.run(cmd_train, check=True)

    # バリデーション結果取得
    with open(report_out, "r") as f:
        report = json.load(f)
    # ここではaccuracyで最適化（他指標も可）
    return report["accuracy"]

# --- Optunaで最適TP/SL幅探索 ---
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)   # n_trialsを必要数に変更

print("[INFO] 最適結果:", study.best_params)
print("[INFO] ベスト精度:", study.best_value)
