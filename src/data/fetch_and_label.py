# src/data/fetch_and_label.py
import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--bars", type=int, default=10000)
    parser.add_argument("--tp", type=int, default=30)
    parser.add_argument("--sl", type=int, default=30)
    args = parser.parse_args()

    # 例: MT5やAPIでOHLCVデータ取得
    # df = fetch_ohlcv(args.symbol, args.bars)
    # 今回はダミーデータ
    df = pd.DataFrame({"open":[1,2,3], "high":[2,3,4], "low":[0,1,2], "close":[1,2,3], "symbol":[args.symbol]*3})

    # 特徴量・ラベル作成
    df["row_number"] = range(1, len(df)+1)
    df["label"] = [1,0,1]  # 仮ラベル

    # 出力先を統一
    out_dir = Path("/content/drive/MyDrive/project_10k_to_1m_data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = f"{args.symbol}_{args.bars}bars"
    out_path = out_dir / f"fetched_and_labeled_{basename}.csv"
    print(f"[INFO] 出力: {out_path}")
    df.to_csv(out_path, index=False)
    print("[INFO] 完了: fetch_and_labelのCSVを保存しました。")

if __name__ == "__main__":
    main()
