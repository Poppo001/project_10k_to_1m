import pandas as pd
import numpy as np

# テクニカル指標
import ta  # ta 0.11以降を推奨

# --- 入出力パス（Colab用例：Drive、VS Codeはローカルに読み替え）---
INPUT = "/content/drive/MyDrive/project_10k_to_1m_data/processed/labeled_USDJPY_H1_1000000bars.csv"
OUTPUT = "/content/drive/MyDrive/project_10k_to_1m_data/processed/feat_USDJPY_H1_1000000bars_TECH.csv"

# --- データ読込 ---
df = pd.read_csv(INPUT)

# --- プライスアクション ---
df["candle_size"] = df["high"] - df["low"]
df["candle_body"] = abs(df["close"] - df["open"])
df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
df["return"] = df["close"].pct_change()
df["gap"] = df["open"] - df["close"].shift(1)

# --- 移動平均 ---
df["ma_5"] = df["close"].rolling(window=5).mean()
df["ma_20"] = df["close"].rolling(window=20).mean()
df["ma_50"] = df["close"].rolling(window=50).mean()

# --- ボラティリティ系 ---
df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
df["bb_bbm"] = ta.volatility.bollinger_mavg(df["close"], window=20)
df["bb_bbh"] = ta.volatility.bollinger_hband(df["close"], window=20)
df["bb_bbl"] = ta.volatility.bollinger_lband(df["close"], window=20)
# Parkinson’s HL変動幅
df["parkinson_vol"] = (1/(4*np.log(2))) * ((np.log(df["high"]/df["low"]))**2).rolling(window=14).mean().apply(np.sqrt)

# --- モメンタム系 ---
df["roc_10"] = ta.momentum.roc(df["close"], window=10)
df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
df["stochrsi_14"] = ta.momentum.stochrsi(df["close"], window=14)
df["cci_14"] = ta.trend.cci(df["high"], df["low"], df["close"], window=14)

# --- トレンド強弱 ---
df["adx_14"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
df["macd"] = ta.trend.macd_diff(df["close"])
df["dmi_plus"] = ta.trend.adx_pos(df["high"], df["low"], df["close"], window=14)
df["dmi_minus"] = ta.trend.adx_neg(df["high"], df["low"], df["close"], window=14)

# --- 欠損値削除 ---
df = df.dropna().reset_index(drop=True)

# --- 保存 ---
df.to_csv(OUTPUT, index=False)
print(f"[INFO] 入力: {INPUT}")
print(f"[INFO] 出力: {OUTPUT}")
print(f"[INFO] 完了: {df.shape[0]}行, {df.shape[1]}列")

