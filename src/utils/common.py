#!/usr/bin/env python3
# src/utils/common.py

import sys
import yaml
from pathlib import Path

def is_colab_env() -> bool:
    """
    Google Colab 上で動いているかどうかを判定する。
    """
    return 'google.colab' in sys.modules

def load_config() -> dict:
    """
    config.yaml を読み込んで辞書で返す。
    Colab 実行時は data_base のみ Drive のパスに差し替える。
    """ 
    # 1) config.yaml の読み込み
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 2) Colab 環境なら data_base を Drive 上に書き換え
    if is_colab_env():
        cfg['data_base'] = '/content/drive/MyDrive/project_10k_to_1m_data'

    return cfg

def resolve_path(template: str, cfg: dict) -> Path:
    """
    config.yaml 中の "${key}" プレースホルダを
    cfg[key] の文字列で置き換えて Path オブジェクトで返す。
    例:
        template = "${data_base}/raw"
        cfg["data_base"] = "/content/drive/MyDrive/.../project_10k_to_1m_data"
        → Path("/content/drive/MyDrive/.../project_10k_to_1m_data/raw")
    """
    path_str = template
    for key, val in cfg.items():
        if isinstance(val, str):
            path_str = path_str.replace(f"${{{key}}}", val)
    return Path(path_str).resolve()
