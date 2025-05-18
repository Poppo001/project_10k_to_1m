import joblib
import pandas as pd

# モデルとデータ読み込み
model = joblib.load("models/xgb_usdjpy.pkl")
df = pd.read_csv("data/processed/feat_USDJPY_H1_20000.csv")

# -1 ラベル除外
df = df[df.label != -1]

# 予測
X = df.drop(columns=["time", "label"])
proba = model.predict_proba(X)[:, 1]

# 確率表示
print("サンプル確率:", proba[:5])
