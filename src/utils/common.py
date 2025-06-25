#!/usr/bin/env python3
# src/utils/common.py

import sys
import yaml
from pathlib import Path

def is_colab_env() -> bool:
    """
    実行環境が Google Colab 上かどうかを判定。
    """
    return 'google.colab' in sys.modules

def load_config() -> dict:
    """
    プロジェクトルートの config.yaml を読み込み、
    Colab 実行時は data_base を Drive 上のパスに上書きした設定辞書を返す。
    """
    # config.yaml の場所を特定
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Colab で走らせる場合のみ data_base を Drive の実パスに書き換え
    if is_colab_env():
        cfg['data_base'] = '/content/drive/MyDrive/project_10k_to_1m_data'

    return cfg

def resolve_path(template: str, cfg: dict) -> Path:
    """
    config.yaml 中の "${key}" プレースホルダを
    設定値に展開して実際の Path を返す。
    例: template="${data_base}/raw" → "/content/drive/.../raw"
    """
    path_str = template
    for key, val in cfg.items():
        if isinstance(val, str):
            path_str = path_str.replace(f"${{{key}}}", val)
    return Path(path_str).resolve()
