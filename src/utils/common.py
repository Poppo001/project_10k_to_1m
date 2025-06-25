#!/usr/bin/env python3
# src/utils/common.py

import sys
import yaml
from pathlib import Path

def is_colab_env() -> bool:
    """
    Google Colab 上で動作しているか判定する。
    """
    return 'google.colab' in sys.modules

def load_config() -> dict:
    """
    config.yaml を読み込んで辞書を返す。
    Colab 実行時には data_base に data_base_colab を、
    ローカル実行時には data_base_local をセットしてから返す。
    """
    # 1) config.yaml 読み込み
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 2) data_base を環境に応じて上書き
    if is_colab_env():
        # Colab 上では data_base_colab を使う
        cfg['data_base'] = cfg.get('data_base_colab')
    else:
        # ローカルでは data_base_local を使う
        cfg['data_base'] = cfg.get('data_base_local')

    return cfg

def resolve_path(template: str, cfg: dict) -> Path:
    """
    config.yaml のテンプレート ("${data_base}/raw" など) を
    load_config() でセットされた cfg['data_base'] を含む cfg で展開し、
    実際のパス(Path)を返す。
    """
    path_str = template
    for key, val in cfg.items():
        if isinstance(val, str):
            path_str = path_str.replace(f"${{{key}}}", val)
    return Path(path_str).resolve()
