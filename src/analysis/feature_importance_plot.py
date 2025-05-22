import joblib
import pandas as pd
import matplotlib.pyplot as plt

# （1）パス設定
MODEL_PATH = "/content/drive/MyDrive/project_10k_to_1m_data/processed/xgb_model_labeled_USDJPY_H1_1000000bars.pkl"
CSV_PATH = "/content/drive/MyDrive/project_10k_to_1m_data/processed/labeled_USDJPY_H1_1000000bars.csv"

# （2）モデルと特徴量名のロード
model = joblib.load(MODEL_PATH)
df = pd.read_csv(CSV_PATH)
feature_names = [c for c in df.columns if c not in ["label", "win_loss", "time"]]

# （3）重要度抽出と並べ替え
importances = model.feature_importances_
indices = importances.argsort()[::-1]

# （4）グラフ表示
plt.figure(figsize=(12, 6))
plt.title("Feature Importances (XGBoost)")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
