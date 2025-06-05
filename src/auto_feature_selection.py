# src/auto_feature_selection.py

import argparse
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import shap
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="自動特徴量選択スクリプト")
    parser.add_argument(
        "--csv", required=True,
        help="入力のラベル付きCSVファイルパス（例: data/processed/labeled_USDJPY_H1_100000.csv）"
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="モデルや結果を保存するディレクトリパス（例: data/processed）"
    )
    parser.add_argument(
        "--out", required=True,
        help="選択後特徴量CSVの出力パス（例: data/processed/selfeat_USDJPY_H1_100000_timestamp.csv）"
    )
    parser.add_argument(
        "--window_size", type=int, default=5000,
        help="ローリングウィンドウのサイズ（デフォルト: 5000）"
    )
    parser.add_argument(
        "--step", type=int, default=500,
        help="ローリングウィンドウのステップ幅（デフォルト: 500）"
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="選択する特徴量の上位数（デフォルト: 10）"
    )
    args = parser.parse_args()

    input_path = Path(args.csv)
    output_dir = Path(args.out_dir)
    out_selected_csv = Path(args.out)

    # 入力ファイルの存在確認
    if not input_path.exists():
        print(f"[ERROR] 入力ファイルが見つかりません: {input_path}")
        return

    # 出力先ディレクトリを作成
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    if not out_selected_csv.parent.exists():
        out_selected_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- データ読み込み ---
    df = pd.read_csv(input_path)

    # --- 目的変数および除外列設定 ---
    target_col = "label"
    drop_cols = ["label", "win_loss", "time", "signal", "prob", "equity", "trade_pips"]
    feature_cols_all = [c for c in df.columns if c not in drop_cols]

    window_size = args.window_size
    step = args.step
    top_k = args.top_k

    results = []

    # 最後のウィンドウの end を記録しておき、最後に選んだ特徴量リストで CSV を作成するための変数
    final_selected_features = None

    for start in range(0, len(df) - window_size - step, step):
        end = start + window_size
        df_train = df.iloc[start:end]
        df_test = df.iloc[end:end+step]

        X_train, y_train = df_train[feature_cols_all], df_train[target_col]
        X_test, y_test = df_test[feature_cols_all], df_test[target_col]

        # 初回：全特徴量で学習
        model_full = XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42)
        model_full.fit(X_train, y_train)

        # SHAP値で重要度分析
        explainer = shap.TreeExplainer(model_full)
        shap_values = explainer.shap_values(X_train)

        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(shap_abs_mean)[::-1][:top_k]
        selected_features = [feature_cols_all[i] for i in top_indices]

        final_selected_features = selected_features  # ループ最後に使う

        # 選択された特徴量のみで再学習
        model_selected = XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42)
        model_selected.fit(X_train[selected_features], y_train)

        # テスト期間で予測・評価
        y_pred = model_selected.predict(X_test[selected_features])
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = report["accuracy"]

        results.append({
            "start_index": end,
            "selected_features": selected_features,
            "accuracy": accuracy,
        })

        # モデルと特徴量リストを保存
        model_path = output_dir / f"xgb_model_{end}.pkl"
        joblib.dump(model_selected, model_path)

        feat_json_path = model_path.with_suffix(".json")
        with open(feat_json_path, "w", encoding="utf-8") as f:
            json.dump(selected_features, f, ensure_ascii=False, indent=2)

        print(f"[INFO] インデックス {end}: 精度={accuracy:.4f}, 特徴量={selected_features}")
        print(f"[INFO] モデル保存: {model_path}")
        print(f"[INFO] 特徴量リスト保存: {feat_json_path}")

    # --- 選ばれた特徴量だけでCSVを作成 ---
    if final_selected_features is not None:
        df_selected = df[final_selected_features + [target_col, "time"]]  # ラベル＋timeも残す想定
        df_selected.to_csv(out_selected_csv, index=False)
        print(f"[INFO] 選択後特徴量CSVを保存: {out_selected_csv}")
    else:
        print("[WARNING] 特徴量選択ループが1回も回りませんでした。CSVを作成しません。")

    # --- 結果CSV保存 ---
    results_df = pd.DataFrame(results)
    results_csv_path = output_dir / "auto_feature_selection_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"[INFO] 自動特徴量選択結果をCSV保存完了: {results_csv_path}")

if __name__ == "__main__":
    main()
