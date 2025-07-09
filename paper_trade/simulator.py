#紙上シミュレーション（simulator.py
import pandas as pd

# — シグナル読み込み
df = pd.read_csv('paper_trade/signals.csv', parse_dates=['time'])

# — スリッページ・スプレッドを反映した実約定価格
df['exec_price'] = df.apply(
    lambda r: r['price'] + (r['spread'] + r['slippage'])/10000 
        if r['signal']=='LONG'
        else r['price'] - (r['spread'] + r['slippage'])/10000,
    axis=1
)

# — 仮想決済：次バー始値で決済した場合のPnL計算
#    ※必要に応じて TP/SL もここに実装
from MetaTrader5 import TIMEFRAME_H1, copy_rates_from_pos
import MetaTrader5 as mt5
mt5.initialize()

results = []
for _, row in df.iterrows():
    # 次バー1本後の始値取得
    rates = mt5.copy_rates_from_pos(
        cfg['mt5']['symbol'],
        getattr(mt5, f"TIMEFRAME_{cfg['mt5']['timeframe']}"),
        0, 2
    )
    next_open = rates[1]['open']
    pnl = (next_open - row['exec_price']) * (1 if row['signal']=='LONG' else -1) * 10000
    results.append(pnl)

df['pnl'] = results

# — レポート集計
summary = {
    'total_trades': len(df),
    'win_rate': (df['pnl']>0).mean(),
    'avg_pnl': df['pnl'].mean(),
    'max_drawdown': df['pnl'].cumsum().min(),
    'total_pnl': df['pnl'].sum()
}
print(pd.Series(summary))
df.to_csv('paper_trade/simulation_results.csv', index=False)