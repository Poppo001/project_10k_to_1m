"""
src/data/feature_gen.py
-----------------------
raw CSV → テクニカル指標／価格派生を付与して
data/processed/ に保存。
"""
# src/data/feature_gen.py

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="入力CSVファイルの絶対パス")
    args = parser.parse_args()

    infile_path = Path(args.file)
    if not infile_path.exists():
        raise FileNotFoundError(f"[ERROR] 入力ファイルが見つかりません: {infile_path}")

    print(f"[INFO] 入力: {infile_path}")
    df = pd.read_csv(infile_path)

    # 仮の特徴量生成（例: 行番号列）
    df_feat = df.copy()
    df_feat["row_number"] = range(1, len(df) + 1)

    # 出力先フォルダを「大容量データ専用」に変更
    basename = infile_path.stem
    out_dir = Path("/content/drive/MyDrive/project_10k_to_1m_data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"feat_{basename}.csv"
    print(f"[INFO] 出力: {out_path}")
    df_feat.to_csv(out_path, index=False)
    print("[INFO] 完了: 特徴量CSVを保存しました。")


if __name__ == "__main__":
    main()