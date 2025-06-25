#!/usr/bin/env python3
# src/utils/common.py

import sys
import yaml
from pathlib import Path

def is_colab_env() -> bool:
    """Google Colab 上かどうかを判定する"""
    return 'google.colab' in sys.modules

def load_config() -> dict:
    """
    config.yaml を読み込んで辞書を返す。
    Colab 実行時には mt5_data_dir, processed_dir, model_dir, report_dir を
    Drive 上のパスに上書きする。
    """
    # 1) config.yaml の読み込み
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 2) Colab ならディレクトリを Google Drive 上のパスに置き換え
    if is_colab_env():
        # Drive 上のプロジェクトコードルート
        code_drive = '/content/drive/MyDrive/project_10k_to_1m'
        # Drive 上のデータルート
        data_drive = '/content/drive/MyDrive/project_10k_to_1m_data'

        # 上書き
        cfg['mt5_data_dir']   = f"{data_drive}/raw"
        cfg['processed_dir']  = f"{data_drive}/processed"
        cfg['model_dir']      = f"{data_drive}/processed/models"
        cfg['report_dir']     = f"{data_drive}/processed/reports"
        # （必要なら project_base も置き換え）
        cfg['project_base']   = code_drive

    return cfg

def resolve_path(template: str, cfg: dict) -> Path:
    """
    テンプレート文字列 "${data_base}/…" または
    そのままの絶対パスを展開して Path を返す。
    """
    path_str = template
    # プレースホルダ置換
    for key, val in cfg.items():
        if isinstance(val, str):
            path_str = path_str.replace(f"${{{key}}}", val)
    return Path(path_str).resolve()
