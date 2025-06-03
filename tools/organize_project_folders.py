import os
import shutil
from pathlib import Path

# プロジェクトルートを自動判定（このスクリプトの1つ上の階層を想定）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 整理用ディレクトリ
FOLDERS = {
    "raw": ["_raw", "rawdata", "raw"],  # 生データ
    "processed": ["labeled_", "feat_", "processed"],  # 前処理済み
    "model": [".pkl"],  # モデルファイル
    "report": [".json"],  # 学習レポート等
    "excel": [".xlsx", ".xls"],
    # 必要に応じて追加
}

def categorize_file(filename):
    """ファイル名や拡張子で種別判定"""
    name = filename.lower()
    if name.endswith((".pkl", ".joblib")):
        return "model"
    if name.endswith(".json"):
        return "report"
    if name.endswith((".xlsx", ".xls")):
        return "excel"
    if any(name.startswith(prefix) for prefix in FOLDERS["raw"]):
        return "raw"
    if any(name.startswith(prefix) for prefix in FOLDERS["processed"]):
        return "processed"
    # デフォルトはprocessedへ
    return "processed"

def organize_data_folder():
    # data配下の全ファイルを走査
    for file in DATA_DIR.glob("*"):
        if file.is_file():
            category = categorize_file(file.name)
            dest_dir = DATA_DIR / category
            dest_dir.mkdir(exist_ok=True)
            dest_file = dest_dir / file.name
            # すでに同名ファイルがあればスキップ（必要なら上書きオプション追加可）
            if dest_file.exists():
                print(f"[SKIP] {dest_file} 既に存在")
                continue
            shutil.move(str(file), str(dest_file))
            print(f"[MOVE] {file.name} → {dest_dir}/")

if __name__ == "__main__":
    print(f"[INFO] data/ フォルダを整理中...")
    organize_data_folder()
    print("[INFO] 完了！")
