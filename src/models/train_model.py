#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train XGBoost model with optional hyperparams and custom CSV')
    parser.add_argument('--symbol', required=True, help='Currency pair symbol')
    parser.add_argument('--timeframe', required=True, help='Timeframe, e.g. M5')
    parser.add_argument('--bars', type=int, required=True, help='Number of bars')
    parser.add_argument('--csv', help='Path to CSV with selected features and label')
    parser.add_argument('--params', help='Path to JSON file with hyperparameters')
    args = parser.parse_args()

    base = '/content/drive/MyDrive/project_10k_to_1m_data'
    models_dir = os.path.join(base, 'processed', 'models')

    # Load data
    if args.csv:
        df = pd.read_csv(args.csv)
        feature_cols = [c for c in df.columns if c not in ['time', 'label', 'future_return']]
        X = df[feature_cols]
        y = df['label']
    else:
        # Fallback to default labeled CSV
        labeled_csv = os.path.join(
            base, 'processed', args.symbol, args.timeframe,
            f'labeled_{args.symbol}_{args.timeframe}_{args.bars}.csv')
        df = pd.read_csv(labeled_csv)
        # Load feature list saved by feature selection
        feat_json = os.path.join(
            models_dir, f'xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_features.json')
        with open(feat_json) as f:
            feature_cols = json.load(f)
        X = df[feature_cols]
        y = df['label']

    # Load hyperparameters or use defaults
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
        params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False
        })
    else:
        # Default parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1
        }

    # Train model
    dtrain = xgb.DMatrix(X, label=y)
    print('[INFO] Training XGBoost model...')
    model = xgb.train(params, dtrain, num_boost_round=100)

    # Save model
    model_path = os.path.join(
        models_dir, f'xgb_model_{args.symbol}_{args.timeframe}_{args.bars}.pkl')
    model.save_model(model_path)
    print(f'[INFO] Model saved: {model_path}')

    # Save feature cols (overwrite or create)
    feat_out = os.path.join(
        models_dir, f'xgb_model_{args.symbol}_{args.timeframe}_{args.bars}_features.json')
    with open(feat_out, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f'[INFO] Feature list saved: {feat_out}')
