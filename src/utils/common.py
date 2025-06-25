#!/usr/bin/env python3
# src/utils/common.py

import sys
import yaml
from pathlib import Path

def is_colab_env() -> bool:
    """
    Google Colab 上で走っているかどうか判定
    """
    return 'google.colab' in sys.modules

def load_config() -> dict:
    """
    config.yaml を読み込んで辞書を返す。
    Colab 実行時には data_base と関連ディレクトリを Drive 上のパスに上書き。
    """
    # 1) config.yaml の読み込み
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 2) Colab 環境なら data_base を Drive のマウント先に置き換え
    if is_colab_env():
        # Google Drive 上の project_10k_to_1m_data を指定
        drive_data = '/content/drive/MyDrive/project_10k_to_1m_data'
        cfg['data_base'] = drive_data

    # 3) data_base を基に各ディレクトリの絶対パス文字列を再構築
    #    （config.yaml 内に "${data_base}/raw" などテンプレートで書かれている前提）
    cfg['mt5_data_dir']   = f"{cfg['data_base']}/raw"
    cfg['processed_dir']  = f"{cfg['data_base']}/processed"
    cfg['model_dir']      = f"{cfg['processed_dir']}/models"
    cfg['report_dir']     = f"{cfg['processed_dir']}/reports"

    return cfg

def resolve_path(template: str, cfg: dict) -> Path:
    """
    config.yaml 内の "${key}" プレースホルダを設定値で展開し、
    実際のディレクトリ Path を返す。
    """
    path_str = template
    for key, val in cfg.items():
        if isinstance(val, str):
            path_str = path_str.replace(f"${{{key}}}", val)
    return Path(path_str).resolve()
