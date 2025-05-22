import argparse
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="ラベル付きCSVファイル")
    args = parser.parse_args()

    in_path = Path(args.file)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    df = pd.read_csv(in_path)
    print(f"[INFO] 学習データ読込: {in_path}")

    # 特徴量の選択： label, win_loss, time は除外
    features = [c for c in df.columns if c not in ["label", "win_loss", "time"]]
    X = df[features].copy()

    # object型（文字列）カラムはfloat化。それが無理なら除外
    for col in X.columns:
        if X[col].dtype == "O":
            try:
                X[col] = X[col].astype(float)
            except:
                print(f"カラム {col} はfloat化できず除外します")
                X = X.drop(columns=[col])
    y = df["label"]

    # 学習・検証データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # モデル学習
    model = XGBClassifier(tree_method="hist", use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    print("[INFO] モデル学習完了")

    # 推論・評価
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] 精度: {acc:.3f}")
    print(classification_report(y_test, y_pred))
    print("混同行列:\n", confusion_matrix(y_test, y_pred))

    # モデル保存
    out_dir = Path("/content/drive/MyDrive/project_10k_to_1m_data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xgb_model_{in_path.stem}.pkl"
    joblib.dump(model, out_path)
    print(f"[INFO] モデル保存: {out_path}")

if __name__ == "__main__":
    main()