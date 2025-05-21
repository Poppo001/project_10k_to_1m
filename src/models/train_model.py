# src/models/train_model.py
import argparse
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="ラベル付きCSVファイル")
    args = parser.parse_args()

    # ラベル付きCSVを大容量データ保管場所から読み込む
    in_path = Path(args.file)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    df = pd.read_csv(in_path)
    print(f"[INFO] 学習データ読込: {in_path}")

    # 仮の特徴量・ラベル指定
    features = [c for c in df.columns if c not in ["label", "win_loss"]]
    X = df[features]
    y = df["label"]

    # XGBoostモデルで学習
    model = XGBClassifier()
    model.fit(X, y)

    # モデル出力先も統一（必要ならpickle等で保存）
    out_dir = Path("/content/drive/MyDrive/project_10k_to_1m_data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xgb_model_{in_path.stem}.pkl"
    import joblib
    joblib.dump(model, out_path)
    print(f"[INFO] モデル保存: {out_path}")

if __name__ == "__main__":
    main()
