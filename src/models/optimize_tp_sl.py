#!/usr/bin/env python3
"""
optimize_tp_sl.py

ステージ2：Stage1で学習・閾値確定済みのモデルを用い、TP/SLを最適化するスクリプトです。
Colab無料版でも高速に実行できるよう、Trial数を抑えています。

#!/usr/bin/env python3
\"\"\"optimize_tp_sl.py

ステージ2：Stage1で学習・閾値確定済みのモデルを用い、TP/SLを最適化するスクリプトです。
Colab無料版でも高速に実行できるよう、Trial数を抑えています。
\"\"\"\nimport argparse\nimport json\nimport pandas as pd\nimport numpy as np\nimport optuna\nimport xgboost as xgb\nfrom backtest import calculate_sharpe\n\n\ndef load_data(csv_path, features_json):\n    df = pd.read_csv(csv_path)\n    with open(features_json, 'r') as f:\n        features = json.load(f)\n    X = df[features]\n    ret = df['future_return']\n    return X, ret\n\n\ndef objective(trial, model, X, ret, threshold, spread, slippage):\n    # TP/SL 探索範囲\n    tp = trial.suggest_float('tp', 1.0, 50.0)\n    sl = trial.suggest_float('sl', 1.0, 50.0)\n\n    # 予測 & シミュレーション\n    dmat = xgb.DMatrix(X)\n    preds = model.predict(dmat)\n    signals = (preds >= threshold).astype(int)\n    pnl_list = [\n        (np.clip(r, -sl, tp) - spread - slippage) if sig == 1 else 0.0\n        for sig, r in zip(signals, ret)\n    ]\n\n    return calculate_sharpe(pnl_list)\n\n\ndef main():\n    parser = argparse.ArgumentParser(\n        description='Stage2: TP/SL 最適化 (Sharpe max)'\n    )\n    parser.add_argument('--csv',       required=True, help='ラベル付きCSVパス')\n    parser.add_argument('--features',  required=True, help='特徴量リストJSONパス')\n    parser.add_argument('--model-path', required=True, help='Stage1モデル(.xgb)パス')\n    parser.add_argument('--threshold', type=float, required=True, help='Stage1で最適化された閾値')\n    parser.add_argument('--spread',    type=float, default=3.4, help='スプレッド(pips)')\n    parser.add_argument('--slippage',  type=float, default=10.0, help='スリッページ(pips)')\n    parser.add_argument('--trials',    type=int, default=15, help='Optuna試行回数')\n    parser.add_argument('--output',    required=True, help='結果JSON出力パス')\n    args = parser.parse_args()\n\n    # データ読み込み\n    X, ret = load_data(args.csv, args.features)\n\n    # モデル読み込み\n    model = xgb.Booster()\n    model.load_model(args.model_path)\n\n    # TP/SL 最適化\n    study = optuna.create_study(direction='maximize')\n    study.optimize(\n        lambda trial: objective(\n            trial, model, X, ret,\n            args.threshold, args.spread, args.slippage\n        ),\n        n_trials=args.trials\n    )\n\n    best = study.best_params.copy()\n    best_sharpe = study.best_value\n\n    # 結果保存\n    with open(args.output, 'w') as f:\n        json.dump(best, f, indent=2)\n\n    print(f\"Best Sharpe : {best_sharpe:.4f}\")\n    print(f\"Best TP/SL  : {best}\")\n\nif __name__ == '__main__':\n    main()\n```  

---

これらを順に実行していただくことで、**Colab無料環境×一人開発×最速**を両立しつつ、高品質なパラメータチューニングが可能です。ご活用ください！

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
    # TP/SL探索範囲
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

    return calculate_sharpe(pnl_list)


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
