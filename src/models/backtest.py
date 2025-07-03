#!/usr/bin/env python3
"""
backtest.py

MetaTrader5 から取得した CSV データと学習済みモデルを用いてバックテストを行い、
P&L、勝率、最大ドローダウン、Sharpe Ratio を計算・出力します。
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


def calculate_sharpe(pnl_series, freq_scale=None):
    """
    Sharpe Ratio を計算します。

    Parameters:
    pnl_series : list or np.array
        各トレードの損益（pips 単位）の系列
    freq_scale : int, optional
        年率化のためのスケーリング因子（例：トレード回数、日次リターンなら252など）。
        指定がなければシリーズ長を使用。

    Returns:
    float
        Sharpe Ratio
    """
    arr = np.array(pnl_series, dtype=float)
    if arr.size < 2:
        return np.nan
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return np.nan
    n = freq_scale or arr.size
    return mean / std * np.sqrt(n)


def load_model(args):
    if args.model_path:
        model = xgb.Booster()
        model.load_model(args.model_path)
        return model, 'xgb'
    else:
        base = '/content/drive/MyDrive/project_10k_to_1m_data'
        model_path = os.path.join(base, 'processed', 'models',
                                  f'xgb_model_{args.symbol}_{args.timeframe}_{args.bars}.pkl')
        model = joblib.load(model_path)
        return model, 'joblib'


def main():
    parser = argparse.ArgumentParser(description='Backtest FX strategy')
    parser.add_argument('--symbol', required=True, help='Currency pair symbol')
    parser.add_argument('--timeframe', required=True, help='Timeframe, e.g. M5')
    parser.add_argument('--bars', type=int, required=True, help='Number of bars')
    parser.add_argument('--model-path', help='Path to XGBoost model file (.xgb)')
    parser.add_argument('--spread', type=float, default=3.4, help='Spread in pips')
    parser.add_argument('--slippage', type=float, default=10, help='Slippage in pips')
    args = parser.parse_args()

    # Data & features
    base = '/content/drive/MyDrive/project_10k_to_1m_data'
    data_csv = os.path.join(base, 'processed', args.symbol, args.timeframe,
                             f'labeled_{args.symbol}_{args.timeframe}_{args.bars}.csv')
    df = pd.read_csv(data_csv)

    feat_json = os.path.join(base, 'processed', 'models',
                             f'xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_features.json')
    with open(feat_json) as f:
        feature_cols = json.load(f)
    X = df[feature_cols]

    # Model
    model, model_type = load_model(args)
    if model_type == 'xgb':
        dmat = xgb.DMatrix(X)
        preds = model.predict(dmat)
    else:
        preds = model.predict_proba(X)[:, 1]

    # Generate binary signals
    threshold = 0.5
    signals = (preds >= threshold).astype(int)

    # Backtest loop
    equity = 0.0
    peak = 0.0
    dd = 0.0
    wins = 0
    total = 0
    pnl_list = []

    for idx, signal in enumerate(signals):
        ret = df['future_return'].iloc[idx]
        pnl = 0.0
        # Long
        if signal == 1:
            pnl = ret - args.spread - args.slippage
        # Short (if implemented by model)
        elif signal == -1:
            pnl = -ret - args.spread - args.slippage
        # No trade for signal==0

        equity += pnl
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
        if pnl > 0:
            wins += 1
        total += 1
        pnl_list.append(pnl)

    # Metrics
    win_rate = wins / total * 100 if total else 0
    sharpe = calculate_sharpe(pnl_list)

    # Output
    print(f"Final Equity : {equity:.2f} pips")
    print(f"Win Rate     : {win_rate:.2f}%")
    print(f"Max Drawdown : {dd:.2f} pips")
    print(f"Sharpe Ratio : {sharpe:.2f}")


if __name__ == '__main__':
    main()
