# python src/data/get_mt5_ohlcv.py

import MetaTrader5 as mt5
import pandas as pd
import os
from pathlib import Path
import time

# MT5を初期化
if not mt5.initialize():
    print("[ERROR] MT5の初期化に失敗しました。")
    quit()

# 取得する通貨ペア・コモディティ・暗号資産のリスト
symbols = [
    # 通貨ペア
    "USDJPY", "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURJPY", "GBPJPY", "AUDJPY",
    # コモディティ
    "XAUUSD", "XAGUSD",
    # 暗号資産（代表的なもの）
    "BTCUSD", "ETHUSD"
]

# MT5の全時間足
timeframes = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1
}

bars = 100000  # 取得する本数

# 出力ディレクトリを作成
output_dir = Path("data/MT5_OHLCV")
output_dir.mkdir(parents=True, exist_ok=True)

for symbol in symbols:
    # シンボルを有効化
    selected = mt5.symbol_select(symbol, True)
    if not selected:
        print(f"[WARNING] シンボル取得不可： {symbol}")
        continue
    
    for tf_name, tf in timeframes.items():
        print(f"[INFO] {symbol} ({tf_name})のデータ取得を開始...")
        
        # OHLCVデータ取得
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        
        # データ取得に失敗した場合
        if rates is None or len(rates) == 0:
            print(f"[WARNING] データ取得に失敗またはデータがありません：{symbol} ({tf_name})")
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # ファイル名を明確に指定
        output_file = output_dir / f"{symbol}_{tf_name}_{bars}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"[INFO] 保存完了：{output_file}")
        
        # 負荷軽減のためのスリープ
        time.sleep(1)  

# MT5を終了
mt5.shutdown()
print("[INFO] 全データ取得処理が完了しました。")
