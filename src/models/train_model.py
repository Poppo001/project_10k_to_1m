# src/models/train_model.py
r"""
引数でCSV/モデル/レポートの入出力パスを指定
例:
python src/models/train_model.py --file 入力CSV --model_out モデルpkl --report_out レポートjson
python src/models/train_model.py --file data/processed/labeled_USDJPY_H1_FULL.csv --model_out data/processed/xgb_model_allfeats_USDJPY_H1_FULL.pkl --feature_cols_out data/processed/xgb_model_allfeats_USDJPY_H1_FULL_feature_cols.json --report_out data/processed/xgb_model_allfeats_USDJPY_H1_FULL_report.json
"""
import argparse
import pandas as pd
import joblib
import json
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="学習データCSV")
    parser.add_argument("--model_out", type=str, required=True, help="モデル出力パス")
    parser.add_argument("--feature_cols_out", type=str, required=True, help="特徴量リストJSON")
    parser.add_argument("--report_out", type=str, required=False, help="成績レポートJSON出力先")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    feature_cols = [c for c in df.columns if c not in ["label", "win_loss", "time", "signal", "prob", "equity", "trade_pips"]]
    train = df.iloc[:-2000]
    test = df.iloc[-2000:]
    X_train, y_train = train[feature_cols], train["label"]
    X_test, y_test = test[feature_cols], test["label"]

    model = XGBClassifier(tree_method="hist", use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # モデル保存
    joblib.dump(model, args.model_out)
    print(f"[INFO] モデル保存: {args.model_out}")

    # 特徴量リスト保存
    with open(args.feature_cols_out, "w") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 特徴量リスト保存: {args.feature_cols_out}")

    # レポート保存（任意）
    if args.report_out:
        out = {
            "accuracy": acc,
            "classification_report": report
        }
        with open(args.report_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] レポート保存: {args.report_out}")

if __name__ == "__main__":
    main()
