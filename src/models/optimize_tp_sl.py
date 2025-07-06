#!/usr/bin/env python3
"""
optimize_tp_sl.py

ステージ2：Stage1で学習・閾値確定済みのモデルを用い、TP/SLを最適化するスクリプトです。
Colab無料版でも高速に実行できるよう、Trial数を抑えつつ、nan/inf を回避する実装を含みます。
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
    # TP/SL 探索範囲を定義
    tp = trial.suggest_float('tp', 1.0, 50.0)
    sl = trial.suggest_float('sl', 1.0, 50.0)

    # 予測とシミュレーション
    dmat = xgb.DMatrix(X)
    preds = model.predict(dmat)
    signals = (preds >= threshold).astype(int)
    pnl_list = []
    for sig, r in zip(signals, ret):
        if sig == 1:
            capped = np.clip(r, -sl, tp)
            pnl = capped - spread - slippage
        else:
            pnl = 0.0
        pnl_list.append(pnl)

    # Sharpe Ratio を計算
    sharpe = calculate_sharpe(pnl_list)
    # 非有限値（nan, inf）は最低値で評価
    if not np.isfinite(sharpe):
        return -1.0
    return sharpe


def main():
    parser = argparse.ArgumentParser(
        description='Stage2: TP/SL 最適化 (Sharpe max)'
    )
    parser.add_argument('--csv',       required=True, help='ラベル付きCSVファイルのパス')
    parser.add_argument('--features',  required=True, help='特徴量リストJSONのパス')
    parser.add_argument('--model-path', required=True, help='Stage1で保存したXGBoostモデル(.xgb)のパス')
    parser.add_argument('--threshold',  type=float, required=True, help='Stage1で最適化された閾値')
    parser.add_argument('--spread',    type=float, default=3.4, help='スプレッド(pips)')
    parser.add_argument('--slippage',  type=float, default=10.0, help='スリッページ(pips)')
    parser.add_argument('--trials',    type=int, default=15, help='Optuna試行回数(デフォルト:15)')
    parser.add_argument('--output',    required=True, help='最適TP/SLを保存するJSONパス')
    args = parser.parse_args()

    # データロード
    X, ret = load_data(args.csv, args.features)

    # モデルロード
    model = xgb.Booster()
    model.load_model(args.model_path)

    # Optuna で最適化
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, model, X, ret,
                                 args.threshold, args.spread, args.slippage),
        n_trials=args.trials
    )

    best_params = study.best_params.copy()
    best_sharpe = study.best_value

    # 結果保存
    with open(args.output, 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"Best Sharpe : {best_sharpe:.4f}")
    print(f"Best TP/SL  : {best_params}")

if __name__ == '__main__':
    main()
