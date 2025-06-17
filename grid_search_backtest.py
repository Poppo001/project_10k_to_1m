# grid_search_backtest.py
# ───────────────────────────────────────────────────────────────────────────────
# Colab/GDrive 環境で TP/SL のグリッドサーチを実行し、
# 最適組み合わせでバックテストレポートを作成するスクリプト
# ───────────────────────────────────────────────────────────────────────────────

import itertools
import subprocess
import sys
import json
import yaml
from pathlib import Path
import pandas as pd

# ── 1) パス設定 ─────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path('/content/drive/MyDrive/project_10k_to_1m')
DATA_ROOT        = Path('/content/project_10k_to_1m_data')
CONFIG_YAML      = PROJECT_ROOT / 'config.yaml'
PROCESSED_DIR    = DATA_ROOT    / 'processed'
MODELS_DIR       = DATA_ROOT    / 'models'
REPORTS_DIR      = DATA_ROOT    / 'reports'
BACKTEST_SCRIPT  = PROJECT_ROOT / 'src' / 'backtest.py'

# ── 2) config.yaml 読み込み & シンボル／時間足／バー数取得 ────────────────────
cfg = yaml.safe_load(CONFIG_YAML.read_text(encoding='utf-8'))
symbol    = cfg['symbol']
timeframe = cfg['timeframe']
bars      = cfg['bars']

# コスト設定も config.yaml から
SPREAD     = cfg.get('spread', 3)
COMMISSION = cfg.get('commission', 0)
SLIPPAGE   = cfg.get('slippage', 10)

print(f"Config → symbol={symbol}, timeframe={timeframe}, bars={bars}")
print(f"Costs  → spread={SPREAD}, commission={COMMISSION}, slippage={SLIPPAGE}")

# ── 3) 最新の selfeat CSV とモデルPKL を動的に取得 ──────────────────────────
csv_pattern   = f"selfeat_{symbol}_{timeframe}_{bars}_*.csv"
csv_candidates   = sorted(PROCESSED_DIR.glob(csv_pattern))
if not csv_candidates:
    raise FileNotFoundError(f"No feature CSV matching '{csv_pattern}' in {PROCESSED_DIR}")
csv_file = csv_candidates[-1]

model_pattern = f"xgb_model_{symbol}_{timeframe}_{bars}_*.pkl"
model_candidates = sorted(MODELS_DIR.glob(model_pattern))
if not model_candidates:
    raise FileNotFoundError(f"No model PKL matching '{model_pattern}' in {MODELS_DIR}")
model_file = model_candidates[-1]

print(f"Using CSV:   {csv_file.name}")
print(f"Using Model: {model_file.name}")

# ── 4) グリッドサーチ用パラメータリスト ───────────────────────────────────
tps = list(range(10, 101, 10))  # [10,20,...,100]
sls = list(range(10, 101, 10))  # [10,20,...,100]

# ── 5) グリッドサーチ実行＆結果収集 ───────────────────────────────────────
records = []
for tp, sl in itertools.product(tps, sls):
    # 5.1) config.yaml を上書き
    cfg.update({'tp': tp, 'sl': sl})
    CONFIG_YAML.write_text(yaml.dump(cfg, sort_keys=False), encoding='utf-8')

    # 5.2) 出力パス
    tag       = f"grid_{symbol}_{timeframe}_{bars}_tp{tp}_sl{sl}"
    rpt_path  = REPORTS_DIR / f'backtest_{tag}.json'
    curve_out = REPORTS_DIR / f'backtest_curve_{tag}.png'

    # 5.3) backtest.py 呼び出し
    cmd = [
        sys.executable, str(BACKTEST_SCRIPT),
        '--csv',       str(csv_file),
        '--model',     str(model_file),
        '--report',    str(rpt_path),
        '--curve_out', str(curve_out),
    ]
    print(f"Running TP={tp}, SL={sl} → {rpt_path.name}")
    subprocess.run(cmd, check=True)

    # 5.4) 結果読み込み・記録
    result = json.loads(rpt_path.read_text(encoding='utf-8'))
    result.update({
        'tp_pips':    tp,
        'sl_pips':    sl,
        'spread':     SPREAD,
        'commission': COMMISSION,
        'slippage':   SLIPPAGE,
    })
    records.append(result)

# ── 6) DataFrame 化＆Top10 抽出 ──────────────────────────────────────────────
df = pd.DataFrame(records)
df['profit_per_dd'] = df['cumulative_pips'] / df['max_drawdown']
df_top = df.sort_values(['profit_per_dd','cumulative_pips'], ascending=False).head(10)

print("\n▶ Top 10 parameter sets by profit / max_drawdown")
print(df_top[[
    'tp_pips','sl_pips','spread','commission','slippage',
    'cumulative_pips','max_drawdown','profit_per_dd'
]].to_string(index=False))

# ── 7) 最良組み合わせで再度バックテスト実行 ─────────────────────────────────
best = df_top.iloc[0]
best_tp = int(best['tp_pips'])
best_sl = int(best['sl_pips'])

print(f"\n▶ Running final Backtest with best TP={best_tp}, SL={best_sl}")

# 上書き
cfg.update({'tp': best_tp, 'sl': best_sl})
CONFIG_YAML.write_text(yaml.dump(cfg, sort_keys=False), encoding='utf-8')

final_tag   = f"best_{symbol}_{timeframe}_{bars}_tp{best_tp}_sl{best_sl}"
final_rpt   = REPORTS_DIR / f'backtest_{final_tag}.json'
final_curve = REPORTS_DIR / f'backtest_curve_{final_tag}.png'

final_cmd = [
    sys.executable, str(BACKTEST_SCRIPT),
    '--csv',       str(csv_file),
    '--model',     str(model_file),
    '--report',    str(final_rpt),
    '--curve_out', str(final_curve),
]
subprocess.run(final_cmd, check=True)

# レポート表示
final_report = json.loads(final_rpt.read_text(encoding='utf-8'))
print("\n▶ Final Backtest Report")
for k, v in final_report.items():
    print(f"  {k:15}: {v}")
