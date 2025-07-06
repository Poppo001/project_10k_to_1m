#!/usr/bin/env python3
"""
train_model_optuna.py

ステージ1：モデルとエントリー閾値（threshold）の最適化を行うスクリプトです。
Colab無料版のGPU（Tesla T4等）を使った高速化とOptuna Prunerで時短を図ります。
"""
import argparse
import json
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from optuna.pruners import MedianPruner
from backtest import calculate_sharpe

def load_data(csv_path, features_json=None):
    df = pd.read_csv(csv_path)
    if features_json:
        with open(features_json, 'r') as f:
            features = json.load(f)
    else:
        features = [c for c in df.columns if c not in ['time', 'label', 'future_return']]
    return df, features

def objective(trial, df, features, spread, slippage, use_gpu):
    # 探索するハイパーパラ
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }
    if use_gpu:
        params['tree_method'] = 'gpu_hist'

    # エントリー閾値探索
    threshold = trial.suggest_float('threshold', 0.3, 0.7)

    # 学習/検証分割
    df_train, df_val = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    X_train, y_train = df_train[features], df_train['label']
    X_val,   y_val   = df_val[features],   df_val['future_return']

    # XGBoost モデル学習
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        early_stopping_rounds=20,
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )

    # 検証データでシミュレーション
    dval = xgb.DMatrix(X_val)
    preds   = model.predict(dval)
    signals = (preds >= threshold).astype(int)
    pnl_list = [(ret - spread - slippage) if sig == 1 else 0.0
                for sig, ret in zip(signals, y_val)]

    # Sharpe 計算
    return calculate_sharpe(pnl_list)

def main():
    parser = argparse.ArgumentParser(
        description='Stage1: XGBoost + threshold Optuna (Sharpe max)'
    )
    parser.add_argument('--csv',       required=True, help='ラベル付きCSVパス')
    parser.add_argument('--features',  help='特徴量リストJSONパス')
    parser.add_argument('--trials',    type=int, default=30, help='Optuna試行回数')
    parser.add_argument('--spread',    type=float, default=3.4, help='スプレッド(pips)')
    parser.add_argument('--slippage',  type=float, default=10.0, help='スリッページ(pips)')
    parser.add_argument('--use-gpu',   action='store_true', help='GPU を使用')
    parser.add_argument('--output',    required=True, help='結果JSON出力パス')
    args = parser.parse_args()

    df, features = load_data(args.csv, args.features)

    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(
        lambda trial: objective(
            trial, df, features,
            args.spread, args.slippage, args.use_gpu
        ),
        n_trials=args.trials
    )

    best = study.best_params.copy()
    best_threshold = best.pop('threshold')
    best_sharpe    = study.best_value

    # 出力データ作成
    result = best.copy()
    result['threshold'] = best_threshold
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Best Sharpe    : {best_sharpe:.4f}")
    print(f"Best Params    : {result}")

    # 最終モデルを全データ学習で保存
    dtrain = xgb.DMatrix(df[features], label=df['label'])
    final_params = best.copy()
    final_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False
    })
    if args.use_gpu:
        final_params['tree_method'] = 'gpu_hist'
    final_model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=study.best_trial.number * 10,
        verbose_eval=False
    )
    model_path = args.output.replace('.json', '_model.xgb')
    final_model.save_model(model_path)
    print(f"Saved model to {model_path}")

if __name__ == '__main__':
    main()
