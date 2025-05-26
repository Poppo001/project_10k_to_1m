# src/data/label_gen.py
"""
指定したTP/SLでラベルを自動生成し、指定したパスへ保存するスクリプト
例:
python src/data/label_gen.py --file 入力CSV --tp 15 --sl 10 --out 出力CSV
"""

import argparse
import pandas as pd

def generate_labels(df: pd.DataFrame, tp_pips: int, sl_pips: int) -> pd.DataFrame:
    """
    シンプルなTP/SLラベル生成の例（実務では実際の値動きを見て判定）
    """
    # 必要に応じて、今の価格と将来の高値/安値を比べてラベル付け（簡易例）
    df = df.copy()
    # 仮例: 適当なロジックで勝敗ラベル（本番は正しいTP/SL到達判定関数を実装）
    df["label"] = ((df["close"].shift(-tp_pips) - df["close"]) > 0).astype(int)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="入力CSVファイル")
    parser.add_argument("--tp", type=int, required=True, help="テイクプロフィット(pips)")
    parser.add_argument("--sl", type=int, required=True, help="ストップロス(pips)")
    parser.add_argument("--out", type=str, required=True, help="出力CSVファイル")
    args = parser.parse_args()

    print(f"[INFO] 入力: {args.file}")
    df = pd.read_csv(args.file)
    df_label = generate_labels(df, args.tp, args.sl)
    df_label.to_csv(args.out, index=False)
    print(f"[INFO] 出力: {args.out}")
    print(f"[INFO] 完了: ラベル付与（TP={args.tp}, SL={args.sl}）")

if __name__ == "__main__":
    main()
