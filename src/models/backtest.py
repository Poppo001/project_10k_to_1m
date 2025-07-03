#!/usr/bin/env python3
import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

def load_model(args):
    if args.model_path:
        model = xgb.Booster()
        model.load_model(args.model_path)
        return model, 'xgb'
    else:
        # legacy joblib model path
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
    parser.add_argument('--spread', type=float, default=0.0, help='Spread in pips')
    parser.add_argument('--slippage', type=float, default=0.0, help='Slippage in pips')
    args = parser.parse_args()

    # Load data
    base = '/content/drive/MyDrive/project_10k_to_1m_data'
    data_csv = os.path.join(base, 'processed', args.symbol, args.timeframe,
                             f'labeled_{args.symbol}_{args.timeframe}_{args.bars}.csv')
    df = pd.read_csv(data_csv)

    # Load features list
    feat_json = os.path.join(base, 'processed', 'models',
                             f'xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_features.json')
    with open(feat_json) as f:
        feature_cols = json.load(f)
    X = df[feature_cols]

    # Load Model
    model, model_type = load_model(args)

    # Generate predictions
    if model_type == 'xgb':
        dmat = xgb.DMatrix(X)
        preds = model.predict(dmat)
    else:
        preds = model.predict_proba(X)[:,1]

    # Determine threshold (reuse auto-thresholding logic or fixed)
    # For simplicity, use 0.5
    threshold = 0.5
    signals = (preds >= threshold).astype(int)

    # Backtest logic
    equity = 0.0
    peak = 0.0
    dd = 0.0
    wins = 0
    total = 0
    for idx, signal in enumerate(signals):
        ret = df['future_return'].iloc[idx] - args.spread - args.slippage
        pnl = ret if signal == 1 else -ret
        equity += pnl
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
        wins += (pnl > 0)
        total += 1

    win_rate = wins / total * 100 if total else 0
    sharpe = np.mean(preds) / np.std(preds) if np.std(preds) != 0 else 0

    print(f"Final Equity : {equity:.2f} pips")
    print(f"Win Rate     : {win_rate:.2f}%")
    print(f"Max Drawdown : {dd:.2f} pips")
    print(f"Sharpe Ratio : {sharpe:.2f}")

if __name__ == '__main__':
    main()
