# src/utils/common.py

import sys
import os
from pathlib import Path
import yaml

def is_colab() -> bool:
    """
    Colab 上かどうかを判定。
    """
    return os.getcwd().startswith("/content/drive/MyDrive")

def load_config() -> dict:
    """
    プロジェクト直下の config.yaml を読み込んで辞書で返す
    """
    # ── ここを修正 ──
    # utils/common.py      -> parent=utils
    # parent.parent=src    -> parent.parent.parent=project_root
    project_root = Path(__file__).resolve().parents[2]
    cfg_path = project_root / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def resolve_path(path_template: str, cfg: dict) -> Path:
    """
    config.yaml 中のパス指定文字列（"${…}" を含む）を展開して絶対パス化するユーティリティ。
    run_pipeline.py 等で使われます。
    """
    s = path_template
    for key, val in cfg.items():
        placeholder = "${" + key + "}"
        if placeholder in s:
            s = s.replace(placeholder, str(val))
    if is_colab() and not s.startswith("/content/drive"):
        s = "/content/drive/MyDrive/" + s.lstrip("/\\")
    return Path(s).resolve()

def resolve_data_root(cfg: dict) -> Path:
    """
    FXデータのルートディレクトリを返す。
    config.yaml に data_base_local／data_base_colab を定義しておく前提。
    """
    if is_colab():
        key = "data_base_colab"
    else:
        key = "data_base_local"
    base = cfg.get(key)
    if base is None:
        raise KeyError(f"[common.resolve_data_root] `{key}` not set in config.yaml")
    return Path(base).resolve()
