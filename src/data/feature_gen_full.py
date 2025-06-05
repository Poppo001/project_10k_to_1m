import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import ta

from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, CCIIndicator, ADXIndicator, IchimokuIndicator, PSARIndicator
)
from ta.momentum import (
    RSIIndicator, StochasticOscillator, ROCIndicator, AwesomeOscillatorIndicator, KAMAIndicator
)
from ta.volatility import (
    BollingerBands, AverageTrueRange
)
from ta.volume import (
    OnBalanceVolumeIndicator, MFIIndicator
)

def main():
    parser = argparse.ArgumentParser(description="特徴量生成スクリプト")
    parser.add_argument(
        "--csv", required=True,
        help="入力の MT5 OHLCV CSV ファイルパス（例: data/MT5_OHLCV/USDJPY_H1_100000.csv）"
    )
    parser.add_argument(
        "--out", required=True,
        help="出力する特徴量付き CSV のパス（例: data/processed/feat_USDJPY_H1_100000.csv）"
    )
    args = parser.parse_args()

    input_path = Path(args.csv)
    output_path = Path(args.out)

    # 入力ファイルの存在チェック
    if not input_path.exists():
        print(f"[ERROR] 入力ファイルが見つかりません: {input_path}")
        return

    # 出力先フォルダがなければ作成
    output_dir = output_path.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- データ読込 ---
    df = pd.read_csv(input_path)

    # --- 移動平均線 ---
    for n in [5, 10, 20, 30, 40, 50, 100, 200]:
        df[f"sma_{n}"] = SMAIndicator(df["close"], window=n).sma_indicator()
        df[f"ema_{n}"] = EMAIndicator(df["close"], window=n).ema_indicator()
        df[f"sma_dev_{n}"] = (df["close"] - df[f"sma_{n}"]) / df[f"sma_{n}"]

    # --- エンベロープ ---
    for n in [20]:
        df[f"env_upper_{n}"] = df[f"sma_{n}"] * 1.03  # 3%上
        df[f"env_lower_{n}"] = df[f"sma_{n}"] * 0.97  # 3%下

    # --- パラボリックSAR ---
    psar = PSARIndicator(df["high"], df["low"], df["close"])
    df["psar"] = psar.psar()

    # --- 一目均衡表 ---
    ichimoku = IchimokuIndicator(df["high"], df["low"])
    df["ichimoku_a"] = ichimoku.ichimoku_a()
    df["ichimoku_b"] = ichimoku.ichimoku_b()
    df["ichimoku_base"] = ichimoku.ichimoku_base_line()
    df["ichimoku_conv"] = ichimoku.ichimoku_conversion_line()

    # --- ボリンジャーバンド（±1σ, 2σ, 3σ）---
    for n, sigmas in [(20, [1, 2, 3])]:
        for sigma in sigmas:
            bb = BollingerBands(df["close"], window=n, window_dev=sigma)
            df[f"bb_bbm_{sigma}"] = bb.bollinger_mavg()
            df[f"bb_bbh_{sigma}"] = bb.bollinger_hband()
            df[f"bb_bbl_{sigma}"] = bb.bollinger_lband()

    # --- RSI ---
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()

    # --- ストキャスティクス ---
    sto = StochasticOscillator(df["high"], df["low"], df["close"], window=14)
    df["stoch_k"] = sto.stoch()
    df["stoch_d"] = sto.stoch_signal()

    # --- サイコロジカルライン ---
    n = 12
    df["psy"] = df["close"].rolling(n).apply(
        lambda x: np.sum(x > x.shift(1)) / n * 100 if len(x) == n else np.nan
    )

    # --- MACD ---
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # --- RCI ---
    def rci_func(x):
        size = len(x)
        ranks = pd.Series(x).rank().values
        t = np.arange(1, size + 1)
        d = np.sum((ranks - t) ** 2)
        return (1 - 6 * d / (size * (size ** 2 - 1))) * 100

    df["rci_9"] = df["close"].rolling(window=9).apply(rci_func, raw=False)
    df["rci_26"] = df["close"].rolling(window=26).apply(rci_func, raw=False)

    # --- DMI（ADX, DMI+-, DI+-） ---
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()
    df["dmi_plus"] = adx.adx_pos()
    df["dmi_minus"] = adx.adx_neg()

    # --- モメンタム ---
    df["momentum_10"] = df["close"].diff(10)
    df["roc_10"] = ROCIndicator(df["close"], window=10).roc()

    # --- レシオケータ（KAMA） ---
    df["kama_10"] = KAMAIndicator(df["close"], window=10).kama()
    df["kama_30"] = KAMAIndicator(df["close"], window=30).kama()

    # --- ATR ---
    df["atr_14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # --- OBV（On-Balance Volume）---
    df["obv"] = OnBalanceVolumeIndicator(
        close=df["close"], volume=df["tick_volume"]
    ).on_balance_volume()

    # --- MFI（Money Flow Index）---
    df["mfi_14"] = MFIIndicator(
        high=df["high"], low=df["low"], close=df["close"], volume=df["tick_volume"], window=14
    ).money_flow_index()

    # --- Awesome Oscillator ---
    df["ao"] = AwesomeOscillatorIndicator(df["high"], df["low"]).awesome_oscillator()

    # --- Price Action（ローソク足特徴量）---
    df["candle_size"] = df["high"] - df["low"]
    df["candle_body"] = abs(df["close"] - df["open"])
    df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["gap"] = df["open"] - df["close"].shift(1)
    df["return"] = df["close"].pct_change()

    # --- 欠損値削除 ---
    df = df.dropna().reset_index(drop=True)

    # --- 保存 ---
    df.to_csv(output_path, index=False)
    print(f"[INFO] 入力: {input_path}")
    print(f"[INFO] 出力: {output_path}")
    print(f"[INFO] 完了: {df.shape[0]}行, {df.shape[1]}列")

if __name__ == "__main__":
    main()
