# python src/data/feature_gen_full.py

import pandas as pd
import numpy as np
import ta

# 入出力設定
INPUT = "data/MT5_OHLCV/USDJPY_H1_100000.csv"
OUTPUT = "data/processed/feat_USDJPY_H1_FULL.csv"

# データ読み込み
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
ichi = ta.trend.IchimokuIndicator(df["high"], df["low"])
df["ichi_conversion"] = ichi.ichimoku_conversion_line()
df["ichi_base"] = ichi.ichimoku_base_line()
df["ichi_span_a"] = ichi.ichimoku_a()
df["ichi_span_b"] = ichi.ichimoku_b()

# --- ボリンジャーバンド (±1σ, ±2σ, ±3σ, 20日基準) ---
bb_window = 20
bb_std = df["close"].rolling(bb_window).std()
df["bb_middle"] = df["close"].rolling(bb_window).mean()
df["bb_high_1"] = df["bb_middle"] + bb_std
df["bb_low_1"] = df["bb_middle"] - bb_std
df["bb_high_2"] = df["bb_middle"] + 2 * bb_std
df["bb_low_2"] = df["bb_middle"] - 2 * bb_std
df["bb_high_3"] = df["bb_middle"] + 3 * bb_std
df["bb_low_3"] = df["bb_middle"] - 3 * bb_std

# --- RSI ---
df["rsi_14"] = ta.momentum.rsi(df["close"])

# --- ストキャスティクス ---
stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
df["stoch_k"] = stoch.stoch()
df["stoch_d"] = stoch.stoch_signal()

# --- サイコロジカルライン ---
df["psycho_line"] = df["close"].rolling(12).apply(lambda x: np.sum(np.diff(x) > 0) / 12 * 100)

# --- MACD ---
macd = ta.momentum.MACD(df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_diff"] = macd.macd_diff()

# --- RCI ---
def rci(series, period=9):
    rank_period = series.rolling(period).apply(lambda x: np.corrcoef(x, np.arange(period))[0, 1])
    return rank_period * 100
df["rci_9"] = rci(df["close"], 9)

# --- DMI ---
adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
df["adx"] = adx.adx()
df["dmi_plus"] = adx.adx_pos()
df["dmi_minus"] = adx.adx_neg()

# --- モメンタム ---
df["momentum"] = ta.momentum.ao(df["high"], df["low"])

# --- ROC ---
df["roc"] = ta.momentum.roc(df["close"], window=12)

# --- レシオケータ ---
df["ratio_indicator"] = (df["close"] / df["close"].shift(10)) * 100

# 追加推奨指標
# --- OBV ---
df["obv"] = ta.volume.on_balance_volume(df["close"], df["tick_volume"])

# --- VWAP ---
df["vwap"] = ta.volume.volume_weighted_average_price(df["high"], df["low"], df["close"], df["tick_volume"])

# --- ヒストリカルボラティリティ ---
df["hist_vol_20"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

# --- ケルトナーチャネル ---
kc = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"])
df["kc_high"] = kc.keltner_channel_hband()
df["kc_low"] = kc.keltner_channel_lband()

# --- プライスアクション系 ---
df["inside_bar"] = ((df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))).astype(int)
df["outside_bar"] = ((df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))).astype(int)

# --- 時間季節性 ---
df["hour"] = pd.to_datetime(df["time"]).dt.hour
df["weekday"] = pd.to_datetime(df["time"]).dt.weekday
df["month"] = pd.to_datetime(df["time"]).dt.month

# --- 相関指標（DXY、米債利回り等）---
# 相関データ（仮に同じDataFrame内にある場合）
# df["corr_dxy_20"] = df["close"].rolling(20).corr(df["dxy"])
# ※要外部データ取得済みの場合のみ有効化してください。

# 欠損値処理
df.dropna(inplace=True)

# CSV保存
df.to_csv(OUTPUT, index=False)
print(f"[INFO] 特徴量付き完全版CSVを保存しました: {OUTPUT}")
