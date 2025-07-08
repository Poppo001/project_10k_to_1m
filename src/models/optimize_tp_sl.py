#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from backtest import calculate_sharpe

def load_data(csv_path, features_json):
    df = pd.read_csv(csv_path)
    with open(features_json, 'r') as f:
        features = json.load(f)
    X = df[features]
    ret = df['future_return']
    return X, ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--spread', type=float, default=3.4)
    parser.add_argument('--slippage', type=float, default=10.0)
    parser.add_argument('--thr-low', type=float, default=0.3)
    parser.add_argument('--thr-high', type=float, default=0.7)
    parser.add_argument('--tp-low', type=float, default=5.0)
    parser.add_argument('--tp-high', type=float, default=20.0)
    parser.add_argument('--sl-low', type=float, default=5.0)
    parser.add_argument('--sl-high', type=float, default=20.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    X, ret = load_data(args.csv, args.features)
    model = xgb.Booster()
    model.load_model(args.model_path)
    def objective(trial):
        th = trial.suggest_float('threshold', args.thr_low, args.thr_high)
        tp = trial.suggest_float('tp', args.tp_low, args.tp_high)
        sl = trial.suggest_float('sl', args.sl_low, args.sl_high)
        dmat = xgb.DMatrix(X)
        preds = model.predict(dmat)
        signals = (preds >= th).astype(int)
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        pnl_list = []
        for sig, r in zip(signals, ret):
            pnl = (np.clip(r, -sl, tp) - args.spread - args.slippage) if sig == 1 else 0.0
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
            pnl_list.append(pnl)
        sharpe = calculate_sharpe(pnl_list)
        final_eq = equity
        score = sharpe - args.alpha * (max_dd / (abs(final_eq) + 1e-9))
        return score if np.isfinite(score) else -np.inf
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.trials)
    best = study.best_params.copy()
    result = {'threshold': best.pop('threshold'), 'tp': best.pop('tp'), 'sl': best.pop('sl')}
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Best Score   : {study.best_value:.4f}")
    print(f"Best Params  : {result}")

if __name__ == '__main__':
    main()
