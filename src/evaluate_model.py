# src/evaluate_model.py

import argparse
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="モデル評価スクリプト")
    parser.add_argument(
        "--csv", required=True,
        help="入力の特徴量付きCSVファイルパス（例: data/processed/selfeat_USDJPY_H1_100000.csv）"
    )
    parser.add_argument(
        "--model", required=True,
        help="評価に使用する学習済みモデルファイルパス（.pkl）（例: data/models/xgb_model_….pkl）"
    )
    parser.add_argument(
        "--out", required=True,
        help="出力する評価レポート(JSON)のパス（例: data/reports/eval_report_….json）"
    )
    parser.add_argument(
        "--test_size", type=int, default=2000,
        help="テストデータに使う末尾の行数（デフォルト: 2000）"
    )
    args = parser.parse_args()

    input_path = Path(args.csv)
    model_path = Path(args.model)
    output_path = Path(args.out)

    # 入力ファイルとモデルファイルの存在チェック
    if not input_path.exists():
        print(f"[ERROR] 入力CSVファイルが見つかりません: {input_path}")
        return
    if not model_path.exists():
        print(f"[ERROR] モデルファイルが見つかりません: {model_path}")
        return

    # 出力先ディレクトリを作成
    output_dir = output_path.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # データ読込
    df = pd.read_csv(input_path)

    # 説明変数とラベルの分離
    drop_cols = ["label", "win_loss", "time", "signal", "prob", "equity", "trade_pips"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    if "label" not in df.columns:
        print(f"[ERROR] 'label' 列が入力CSVに存在しません: {input_path}")
        return

    test_size = args.test_size
    if test_size >= len(df):
        print(f"[ERROR] テストサイズ ({test_size}) がデータ行数 ({len(df)}) を超えています。")
        return

    train_df = df.iloc[:-test_size]
    test_df  = df.iloc[-test_size:]

    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    # モデル読み込み
    model = joblib.load(model_path)

    # 予測
    y_pred = model.predict(X_test)

    # 評価指標計算
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # レポート辞書作成
    report_dict = {
        "classification_report": cls_report,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "test_size": test_size,
        "n_samples": int(len(df))
    }

    # JSON 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    print(f"[INFO] モデル評価完了。評価レポートを保存しました: {output_path}")

if __name__ == "__main__":
    main()
