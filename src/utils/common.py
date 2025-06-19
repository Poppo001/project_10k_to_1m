# src/utils/common.py

from pathlib import Path
import sys
import yaml

def is_colab():
    """Colab 上かどうかを判定"""
    return 'google.colab' in sys.modules

def load_config():
    """プロジェクト直下の config.yaml を読み込む"""
    root = Path(__file__).resolve().parents[2]  # project root (C:/Projects/project_10k_to_1m)
    cfg_path = root / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def resolve_data_root(cfg):
    """Colab／ローカルそれぞれのデータルートパスを返す"""
    if is_colab():
        return Path("/content/drive/MyDrive") / cfg["data_root_colab"]
    else:
        return Path(cfg["data_root_local"]).resolve()

def get_latest_file(directory: Path, prefix: str, suffix: str):
    """
    指定ディレクトリ内で、prefix + 任意文字列 + suffix という名前のファイルを探し、
    最終更新日時が最新のものを返す。該当ファイルがなければ None を返す。
    """
    # 例: directory.glob("selfeat_USDJPY_M5_100000_* .csv")
    pattern = f"{prefix}*{suffix}"
    files = list(directory.glob(pattern))
    if not files:
        return None

    # 最終更新日時で最新のファイルを選択
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest
