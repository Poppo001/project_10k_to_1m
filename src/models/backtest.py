#!/usr/bin/env python3
import argparse
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb

def load_model(path):
    model = xgb.Booster()
    model.load_model(path)
    return model

def calculate_sharpe(pnl_list, freq_scale=None):
    arr = np.array(pnl_list, dtype=float)
    if arr.size < 2:
        return np.nan
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return np.nan
    n = freq_scale or arr.size
    return mean / std * np.sqrt(n)

def calculate_max_drawdown(equity_curve):
    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    drawdown = eq - peak
    return drawdown.min()

def main():
    parser = argparse.ArgumentParser(description="Backtest FX strategy with long & short support", allow_abbrev=False)
    parser.add_argument('--symbol',            required=True)
    parser.add_argument('--timeframe',         required=True)
    parser.add_argument('--bars',      type=int, required=True)
    parser.add_argument('--model-path',        required=True)
    parser.add_argument('--features')
    parser.add_argument('--spread',    type=float, default=0.0)
    parser.add_argument('--slippage',  type=float, default=0.0)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--threshold-long',  type=float, default=0.5)
    parser.add_argument('--threshold-short', type=float, default=0.5)
    parser.add_argument('--tp',        type=float, default=0.0)
    parser.add_argument('--sl',        type=float, default=0.0)
    parser.add_argument('--pip-mult',  type=float, default=1.0)
    args = parser.parse_args()

    if args.threshold is not None:
        thr_long = args.threshold
        thr_short = 1.0 - args.threshold
    else:
        thr_long = args.threshold_long
        thr_short = args.threshold_short

    base = '/content/drive/MyDrive/project_10k_to_1m_data'
    data_csv = os.path.join(base, 'processed', args.symbol, args.timeframe,
                            f'labeled_{args.symbol}_{args.timeframe}_{args.bars}.csv')
    df = pd.read_csv(data_csv)

    if args.features:
        feat_json = args.features
    else:
        feat_json = os.path.join(base, 'processed', 'models',
                                 f'xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_features.json')
    with open(feat_json, 'r') as f:
        feature_cols = json.load(f)

    model = load_model(args.model_path)
    X = df[feature_cols]
    dmat = xgb.DMatrix(X)
    preds = model.predict(dmat)

    signals = np.where(preds >= thr_long,  1,
               np.where(preds <= thr_short, -1, 0))

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    wins = 0
    total_trades = 0
    pnl_list = []

    for sig, ret in zip(signals, df['future_return']):
        if sig == 1:
            capped = np.clip(ret, -args.sl, args.tp)
            pnl = capped * args.pip_mult - args.spread - args.slippage
            total_trades += 1
            if pnl > 0:
                wins += 1
        elif sig == -1:
            capped = np.clip(-ret, -args.sl, args.tp)
            pnl = capped * args.pip_mult - args.spread - args.slippage
            total_trades += 1
            if pnl > 0:
                wins += 1
        else:
            pnl = 0.0

        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        pnl_list.append(pnl)

    win_rate = (wins / total_trades * 100) if total_trades > 0 else np.nan
    sharpe = calculate_sharpe(pnl_list)
    drawdown = calculate_max_drawdown(np.cumsum(pnl_list))

    print(f"Final Equity     : {equity:.2f} pips")
    print(f"Total trades     : {total_trades}")
    print(f"Win Rate         : {win_rate:.2f}%")
    print(f"Max Drawdown     : {drawdown:.2f} pips")
    print(f"Sharpe Ratio     : {sharpe:.2f}")

if __name__ == '__main__':
    main()
