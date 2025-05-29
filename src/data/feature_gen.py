import pandas as pd
import numpy as np
import pandas_ta as ta

# --- 入出力設定（適宜変更してください） ---
INPUT = "data/MT5_OHLCV/USDJPY_H1_100000.csv"
OUTPUT = "data/processed/feat_USDJPY_H1.csv"

# --- データ読込 ---
df = pd.read_csv(INPUT)

# --- 移動平均線 ---
ma_windows = [5, 10, 20, 30, 40, 50, 100, 200]
for window in ma_windows:
    df[f"ma_{window}"] = df["close"].rolling(window).mean()

# --- 移動平均乖離率 ---
for window in ma_windows:
    df[f"ma_dev_{window}"] = (df["close"] - df[f"ma_{window}"]) / df[f"ma_{window}"] * 100

# --- エンベロープ (±1%) ---
envelope_pct = 0.01
df["env_upper"] = df["ma_20"] * (1 + envelope_pct)
df["env_lower"] = df["ma_20"] * (1 - envelope_pct)

# --- パラボリック ---
df["parabolic"] = ta.trend.psar(df["high"], df["low"], df["close"])

# --- 一目均衡表 ---
ichi = ta.trend.IchimokuIndicator(high=df["high"], low=df["low"])
df["ichi_conversion"] = ichi.ichimoku_conversion_line()
df["ichi_base"] = ichi.ichimoku_base_line()
df["ichi_span_a"] = ichi.ichimoku_a()
df["ichi_span_b"] = ichi.ichimoku_b()

# --- ボリンジャーバンド (20日, ±2σ) ---
bb = ta.volatility.BollingerBands(df["close"])
df["bb_high"] = bb.bollinger_hband()
df["bb_low"] = bb.bollinger_lband()
df["bb_middle"] = bb.bollinger_mavg()

# --- RSI (14日) ---
df["rsi_14"] = ta.momentum.rsi(df["close"])

# --- ストキャスティックス ---
stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
df["stoch_k"] = stoch.stoch()
df["stoch_d"] = stoch.stoch_signal()

# --- サイコロジカルライン (12日) ---
df["psycho_line"] = df["close"].rolling(12).apply(lambda x: np.sum(np.diff(x) > 0) / 12 * 100)

# --- MACD ---
macd = ta.momentum.MACD(df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_diff"] = macd.macd_diff()

# --- RCI (順位相関指数, 9日) ---
def rci(series, period=9):
    rank_period = series.rolling(period).apply(lambda x: np.corrcoef(x, np.arange(period))[0, 1])
    return rank_period * 100

df["rci_9"] = rci(df["close"], 9)

# --- DMI (ADX, DMI+, DMI-) ---
adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
df["adx"] = adx.adx()
df["dmi_plus"] = adx.adx_pos()
df["dmi_minus"] = adx.adx_neg()

# --- モメンタム (14日) ---
df["momentum"] = ta.momentum.ao(df["high"], df["low"])

# --- ROC (変化率, 12日) ---
df["roc"] = ta.momentum.roc(df["close"], window=12)

# --- レシオケータ (Ratio Indicator, 10日) ---
df["ratio_indicator"] = (df["close"] / df["close"].shift(10)) * 100

# --- 欠損値削除 ---
df.dropna(inplace=True)

# --- 特徴量付きCSVを保存 ---
df.to_csv(OUTPUT, index=False)

print(f"[INFO] 特徴量付きCSVを保存しました: {OUTPUT}")
