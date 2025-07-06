#!/usr/bin/env python3
"""
optimize_tp_sl.py

ステージ2：Stage1で学習・閾値確定済みのモデルを用い、TP/SLを最適化するスクリプトです。
Colab無料版でも高速に実行できるよう、Trial数を抑えています。
"""
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


def objective(trial, model, X, ret, threshold, spread, slippage):
    # TP/SL 探索範囲
    tp = trial.suggest_float('tp', 1.0, 50.0)
    sl = trial.suggest_float('sl', 1.0, 50.0)

    # 予測 & シミュレーション
    dmat = xgb.DMatrix(X)
    preds = model.predict(dmat)
    signals = (preds >= threshold).astype(int)
    pnl_list = [
        (np.clip(r, -sl, tp) - spread - slippage) if sig == 1 else 0.0
        for sig, r in zip(signals, ret)
    ]

    # Sharpe Ratio 計算
    sharpe = calculate_sharpe(pnl_list)
    # nan や inf は最悪として扱う
    if not np.isfinite(sharpe):
        return -1.0
    return sharpe


def main():
    parser = argparse.ArgumentParser(
        description='Stage2: TP/SL 最適化 (Sharpe max)'
    )
    parser.add_argument('--csv',       required=True, help='ラベル付きCSVパス')
    parser.add_argument('--features',  required=True, help='特徴量リストJSONパス')
    parser.add_argument('--model-path', required=True, help='Stage1モデル(.xgb)パス')
    parser.add_argument('--threshold', type=float, required=True, help='Stage1で最適化された閾値')
    parser.add_argument('--spread',    type=float, default=3.4, help='スプレッド(pips)')
    parser.add_argument('--slippage',  type=float, default=10.0, help='スリッページ(pips)')
    parser.add_argument('--trials',    type=int, default=15, help='Optuna試行回数')
    parser.add_argument('--output',    required=True, help='結果JSON出力パス')
    args = parser.parse_args()

    # データ読み込み
    X, ret = load_data(args.csv, args.features)

    # モデル読み込み
    model = xgb.Booster()
    model.load_model(args.model_path)

    # Optuna 実行
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(
            trial, model, X, ret,
            args.threshold, args.spread, args.slippage
        ),
        n_trials=args.trials
    )

    best = study.best_params.copy()
    best_sharpe = study.best_value

    # 出力
    result = best.copy()
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Best Sharpe : {best_sharpe:.4f}")
    print(f"Best TP/SL  : {result}")

if __name__ == '__main__':
    main()
