#!/usr/bin/env python3

# backtest.py

# ───────────────────────────────────────────────────────────────────────────────

# ネスト化ディレクトリ構成に対応し、config.yamlのパラメータをデフォルトとして

# CLI引数(--tp\_pips等)での上書きも可能なリアルTP/SLバックテストスクリプト

# ───────────────────────────────────────────────────────────────────────────────

import argparse
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- 設定読み込み関数 ---

def load\_config():
cfg = yaml.safe\_load(Path('config.yaml').read\_text(encoding='utf-8'))
return cfg

# --- メイン ---

def main():\
# 引数パース
parser = argparse.ArgumentParser(description='Real TP/SL Backtest')
parser.add\_argument('--csv',       required=True, help='Processed feature CSV')
parser.add\_argument('--model',     required=True, help='Trained model .pkl')
parser.add\_argument('--report',    required=True, help='Output JSON report path')
parser.add\_argument('--curve\_out', required=True, help='Output equity curve PNG path')
parser.add\_argument('--tp\_pips',     type=float, help='Take Profit (pips), default from config.yaml')
parser.add\_argument('--sl\_pips',     type=float, help='Stop Loss (pips), default from config.yaml')
parser.add\_argument('--spread',      type=float, help='Spread cost (pips), default from config.yaml')
parser.add\_argument('--commission',  type=float, help='Commission cost (pips), default from config.yaml')
parser.add\_argument('--slippage',    type=float, help='Slippage cost (pips), default from config.yaml')
args = parser.parse\_args()

```
# config.yaml 読込
cfg = load_config()
# パラメータ: CLI優先、未指定時はconfigから
tp_pips     = args.tp_pips     if args.tp_pips     is not None else cfg['tp']
sl_pips     = args.sl_pips     if args.sl_pips     is not None else cfg['sl']
spread      = args.spread      if args.spread      is not None else cfg['spread']
commission  = args.commission  if args.commission  is not None else cfg['commission']
slippage    = args.slippage    if args.slippage    is not None else cfg['slippage']

# 入力ファイルチェック
csv_path   = Path(args.csv);
model_path = Path(args.model)
report_path= Path(args.report)
curve_path = Path(args.curve_out)

# 出力フォルダ作成
report_path.parent.mkdir(parents=True, exist_ok=True)
curve_path.parent.mkdir(parents=True, exist_ok=True)

# データ読み込み
df = pd.read_csv(csv_path)

# モデル & 特徴量リスト読み込み
model = joblib.load(model_path)
feat_cols_file = model_path.with_suffix('_feature_cols.json')
feature_cols = json.loads(feat_cols_file.read_text(encoding='utf-8'))

# 特徴量欠損行除去
df = df.dropna(subset=feature_cols).reset_index(drop=True)

# 確率予測 → シグナル
X = df[feature_cols]
df['prob']   = model.predict_proba(X)[:, 1]
df['signal'] = (df['prob'] > 0.5).astype(int)

# バックテスト実行
equity = []
eq = 0.0
trades = []
for _, row in df.iterrows():
    if row['signal'] == 1:
        # エントリー価格: open + half spread
        entry = row['open'] + spread / 2
        high  = row['high']
        low   = row['low']
        # TP/SL判定
        if high >= entry + tp_pips:
            profit = tp_pips
        elif low <= entry - sl_pips:
            profit = -sl_pips
        else:
            profit = row['close'] - entry
        # コスト控除
        profit = profit - commission - slippage
    else:
        profit = 0.0
    eq += profit
    trades.append(profit)
    equity.append(eq)

df['trade_pips'] = trades
df['equity']     = equity

# メトリクス計算
cumulative_pips = float(eq)
running_max     = np.maximum.accumulate(equity)
drawdowns       = running_max - equity
max_drawdown    = float(np.max(drawdowns))
n_trades        = int((df['signal'] == 1).sum())

# 資産曲線プロット
plt.figure()
plt.plot(equity)
plt.title('Equity Curve')
plt.xlabel('Trade #')
plt.ylabel('Cumulative Pips')
plt.tight_layout()
plt.savefig(curve_path)
plt.close()

# レポート保存
report = {
    'cumulative_pips': cumulative_pips,
    'max_drawdown':    max_drawdown,
    'n_trades':        n_trades,
    'tp_pips':         tp_pips,
    'sl_pips':         sl_pips,
    'spread':          spread,
    'commission':      commission,
    'slippage':        slippage
}
report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(f"[INFO] Report saved: {report_path}")
```

if **name** == '**main**':
main()
