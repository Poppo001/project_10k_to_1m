import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import json
from pathlib import Path

# --- ファイルパスを最新フローに統一 ---
CSV_PATH = "data/processed/labeled_USDJPY_H1_FULL.csv"
MODEL_OUT_DIR = Path("data/processed/models")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- データ読み込み ---
df = pd.read_csv(CSV_PATH)

# --- 目的変数（ラベル）および除外列設定 ---
target_col = "label"
drop_cols = ["label", "win_loss", "time", "signal", "prob", "equity", "trade_pips"]
feature_cols_all = [c for c in df.columns if c not in drop_cols]

# --- ローリングウィンドウ設定 ---
window_size = 5000
step = 500
top_k = 10

results = []

for start in range(0, len(df) - window_size - step, step):
    end = start + window_size
    df_train = df.iloc[start:end]
    df_test = df.iloc[end:end+step]

    X_train, y_train = df_train[feature_cols_all], df_train[target_col]
    X_test, y_test = df_test[feature_cols_all], df_test[target_col]

    # 初回：全特徴量で学習
    model_full = XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42)
    model_full.fit(X_train, y_train)

    # SHAP値で重要度分析
    explainer = shap.TreeExplainer(model_full)
    shap_values = explainer.shap_values(X_train)

    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_abs_mean)[::-1][:top_k]
    selected_features = [feature_cols_all[i] for i in top_indices]

    # 選抜された特徴量のみで再学習
    model_selected = XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42)
    model_selected.fit(X_train[selected_features], y_train)

    # テスト期間で予測・評価
    y_pred = model_selected.predict(X_test[selected_features])
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report["accuracy"]

    results.append({
        "start_index": end,
        "selected_features": selected_features,
        "accuracy": accuracy,
    })

    # モデル・特徴量リストを保存
    model_path = MODEL_OUT_DIR / f"xgb_model_{end}.pkl"
    joblib.dump(model_selected, model_path)
    with open(model_path.with_suffix('.json'), "w") as f:
        json.dump(selected_features, f, ensure_ascii=False, indent=2)

    print(f"[INFO] {end}: 精度={accuracy:.4f}, 特徴量={selected_features}")
    print(f"[INFO] モデル保存: {model_path}")

# --- 結果CSV保存 ---
results_df = pd.DataFrame(results)
results_df.to_csv(MODEL_OUT_DIR / "auto_feature_selection_results.csv", index=False)
print("[INFO] 自動特徴量選択結果をCSV保存完了。")
