#!/usr/bin/env python3
import argparse
import json
import os
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# load labeled data and features JSON

def load_data(csv_path, features_json=None):
    df = pd.read_csv(csv_path)
    if features_json:
        with open(features_json) as f:
            features = json.load(f)
    else:
        features = [c for c in df.columns if c not in ['time','label','future_return']]
    X = df[features]
    y = df['label']
    return X, y, features

# define the objective function for Optuna
def objective(trial, X, y):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, 'validation')],
                      early_stopping_rounds=50, verbose_eval=False)
    preds = model.predict(dval)
    auc = roc_auc_score(y_val, preds)
    return auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGBoost with Optuna hyperparameter tuning')
    parser.add_argument('--csv', required=True, help='Path to labeled CSV')
    parser.add_argument('--features', required=False, help='Path to JSON file listing features')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--output', required=True, help='Path to save best params JSON')
    args = parser.parse_args()

    X, y, features = load_data(args.csv, args.features)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=args.trials)

    print(f"Best AUC: {study.best_value:.4f}")
    print("Best params:", study.best_params)

    # save best params
    with open(args.output, 'w') as f:
        json.dump(study.best_params, f, indent=2)

    # Train final model with best params
    final_params = study.best_params.copy()
    final_params.update({'objective':'binary:logistic', 'eval_metric':'auc', 'use_label_encoder': False})
    dtrain = xgb.DMatrix(X, label=y)
    final_model = xgb.train(final_params, dtrain, num_boost_round=study.best_trial.number*10)
    model_path = os.path.splitext(args.output)[0] + '_model.pkl'
    final_model.save_model(model_path)
    print(f"Final model saved to: {model_path}")
