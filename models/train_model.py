import pandas as pd
import joblib
import json
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === ファイルパスを自分の環境に合わせて調整 ===
INPUT = "data\processed\feat_USDJPY_H1_FULL.csv"
MODEL_OUT = "data/processed/xgb_model_allfeats_USDJPY_H1_1000000bars.pkl"

# --- データ読込 ---
df = pd.read_csv(INPUT)

# --- 説明変数リストを自動抽出（目的変数や不要列は除外）---
drop_cols = ["label", "win_loss", "time", "signal", "prob", "equity", "trade_pips"]
feature_cols = [c for c in df.columns if c not in drop_cols]

# --- Train/Test 分割（時系列で末尾2,000件をテストに）---
train = df.iloc[:-2000]
test  = df.iloc[-2000:]

X_train, y_train = train[feature_cols], train["label"]
X_test , y_test  = test[feature_cols],  test["label"]

# --- モデル学習 ---
model = XGBClassifier(tree_method="hist", use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# --- 精度確認（任意） ---
y_pred = model.predict(X_test)
print("[INFO] モデル精度レポート")
print(classification_report(y_test, y_pred))
print("[INFO] 混同行列")
print(confusion_matrix(y_test, y_pred))

# --- モデル保存 ---
joblib.dump(model, MODEL_OUT)
print(f"[INFO] モデル保存: {MODEL_OUT}")

# --- 使用した特徴量リストも保存（再現性のためおすすめ） ---
with open(MODEL_OUT.replace(".pkl", "_feature_cols.json"), "w") as f:
    json.dump(feature_cols, f, ensure_ascii=False, indent=2)
print("[INFO] 使用した特徴量リストも保存しました")
