# src/utils/common.py

import sys
import os
from pathlib import Path
import yaml

def is_colab() -> bool:
    return os.getcwd().startswith("/content/drive/MyDrive")

def load_config() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    cfg_path = project_root / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def resolve_data_root(cfg: dict) -> Path:
    """
    Colabなら data_base_colab、ローカルなら data_base_local を返す。
    """
    key = "data_base_colab" if is_colab() else "data_base_local"
    base = cfg.get(key)
    if base is None:
        raise KeyError(f"`{key}` not set in config.yaml")
    return Path(base).resolve()

def resolve_path(path_template: str, cfg: dict) -> Path:
    """
    "${...}" プレースホルダ展開＋Colabパス補完。
    run_pipeline.py 内のパス解決で使います。
    """
    s = path_template
    for k, v in cfg.items():
        placeholder = "${" + k + "}"
        if placeholder in s:
            s = s.replace(placeholder, str(v))
    if is_colab() and not s.startswith("/content/drive"):
        s = "/content/drive/MyDrive/" + s.lstrip("/\\")
    return Path(s).resolve()

def get_latest_file(dir_path: Path, patterns):
    """
    globパターンのリストからマッチファイルを集め、ソートして最新を返す。
    patterns: 単一文字列 or 文字列リスト
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    files = []
    for pat in patterns:
        files.extend(dir_path.glob(pat))
    files = sorted(files)
    return files[-1] if files else None
