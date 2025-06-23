#!/usr/bin/env python3
# src/utils/common.py

import yaml
import sys
from pathlib import Path

def load_config() -> dict:
    """
    プロジェクトルートの config.yaml を読み込んで辞書で返す
    """
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def resolve_path(template: str, cfg: dict) -> Path:
    """
    config.yaml 内のパス文字列（"${data_base}/raw" など）を解決して Path で返す
    """
    path_str = template
    for key, val in cfg.items():
        if isinstance(val, str):
            path_str = path_str.replace(f"${{{key}}}", val)
    return Path(path_str).resolve()
