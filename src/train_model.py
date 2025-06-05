# src/train_model.py

import argparse
import pandas as pd
import joblib
import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="モデル学習スクリプト")
    parser.add_argument(
        "--file", required=True,
        help="入力の特徴量付きCSVファイルパス（例: data/processed/feat_USDJPY_H1_FULL.csv）"
    )
    parser.add_argument(
        "--model_out", required=True,
        help="出力するモデルファイルパス（例: data/processed/xgb_model_allfeats.pkl）"
    )
    parser.add_argument(
        "--feature_cols_out", required=True,
        help="出力する特徴量リストJSONのパス（例: data/processed/xgb_model_allfeats_feature_cols.json）"
    )
    parser.add_argument(
        "--test_size", type=int, default=2000,
        help="テストデータに使う末尾の行数（デフォルト: 2000）"
    )
    args = parser.parse_args()

    input_path = Path(args.file)
    model_out_path = Path(args.model_out)
    feature_cols_json = Path(args.feature_cols_out)

    # --- 入力ファイル存在チェック ---
    if not input_path.exists():
        print(f"[ERROR] 入力ファイルが見つかりません: {input_path}")
        return

    # --- 出力先ディレクトリ作成 ---
    for p in [model_out_path.parent, feature_cols_json.parent]:
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

    # --- データ読込 ---
    df = pd.read_csv(input_path)

    # --- 説明変数リストを自動抽出（目的変数や不要列は除外）---
    drop_cols = ["label", "win_loss", "time", "signal", "prob", "equity", "trade_pips"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # --- Train/Test 分割（時系列で末尾 test_size 件をテストに）---
    test_size = args.test_size
    if test_size >= len(df):
        print(f"[ERROR] テストサイズがデータ件数 ({len(df)}) を超えています。")
        return

    train = df.iloc[:-test_size]
    test  = df.iloc[-test_size:]

    X_train, y_train = train[feature_cols], train["label"]
    X_test, y_test   = test[feature_cols],  test["label"]

    # --- モデル学習 ---
    model = XGBClassifier(tree_method="hist", use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # --- 精度確認（任意）---
    y_pred = model.predict(X_test)
    print("[INFO] モデル精度レポート")
    print(classification_report(y_test, y_pred))
    print("[INFO] 混同行列")
    print(confusion_matrix(y_test, y_pred))

    # --- モデル保存 ---
    joblib.dump(model, model_out_path)
    print(f"[INFO] モデル保存: {model_out_path}")

    # --- 使用した特徴量リストも保存 ---
    with open(feature_cols_json, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 使用した特徴量リストも保存しました: {feature_cols_json}")

if __name__ == "__main__":
    main()
