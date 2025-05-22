import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# 設定
CSV_PATH = "/content/drive/MyDrive/project_10k_to_1m_data/processed/labeled_USDJPY_H1_1000000bars.csv"
df = pd.read_csv(CSV_PATH)
features_all = [c for c in df.columns if c not in ["label", "win_loss", "time"]]

window_size = 5000      # 直近N件で学習
top_k = 5               # 重要度上位k個の特徴量のみで判断
step = 500              # 毎にロールウィンドウを更新
results = []

for start in range(0, len(df)-window_size-step, step):
    end = start + window_size
    df_window = df.iloc[start:end]
    X_win = df_window[features_all]
    y_win = df_window["label"]

    # 重要度付きモデル学習
    model = XGBClassifier(tree_method="hist", use_label_encoder=False, eval_metric="logloss")
    model.fit(X_win, y_win)
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_k]  # 上位k特徴量

    # 次のstep分だけは選ばれた特徴量のみで判断
    X_next = df.iloc[end:end+step][[features_all[i] for i in indices]]
    y_next = df.iloc[end:end+step]["label"]
    pred = model.predict(X_next)

    # 成績保存
    acc = (pred == y_next).mean()
    results.append({
        "start": end,
        "features": [features_all[i] for i in indices],
        "accuracy": acc,
    })

    print(f"[INFO] {end}: 重要度上位={results[-1]['features']}, acc={acc:.3f}")

# 結果集計
df_res = pd.DataFrame(results)
print(df_res.head()) 