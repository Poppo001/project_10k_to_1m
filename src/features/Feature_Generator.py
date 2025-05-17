import pandas as pd
import pandas_ta as ta  # ta-lib を使う場合は import talib
import os


def generate_features(input_csv, output_csv=None):
    # データ読み込み
    df = pd.read_csv(input_csv)

    # 時刻をDatetime型に変換（インデックス化も可）
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # 必須カラムがあるか確認
    required_cols = {"open", "high", "low", "close", "tick_volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    # ===== 🧠 テクニカル指標の生成 =====
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["ema_20"] = ta.ema(df["close"], length=20)
    bbands = ta.bbands(df["close"], length=20)
    df = pd.concat([df, bbands], axis=1)

    df["macd"], df["macd_signal"], df["macd_hist"] = ta.macd(df["close"]).values.T

    # ===== ⚠️ 欠損値を処理 =====
    df = df.dropna().reset_index(drop=True)

    # ===== 💾 保存 =====
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"✅ Features saved to {output_csv}")

    return df


if __name__ == "__main__":
    # 🔧 使用例
    symbol = "USDJPY"
    timeframe = "H1"
    input_path = f"data/processed/{symbol}_{timeframe}.csv"
    output_path = f"data/featured/{symbol}_{timeframe}_features.csv"

    generate_features(input_csv=input_path, output_csv=output_path)
