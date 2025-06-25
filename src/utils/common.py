#!/usr/bin/env python3
# src/utils/common.py

import sys
import yaml
from pathlib import Path

def is_colab_env() -> bool:
    """
    Google Colab 上かどうかを判定する。
    """
    return 'google.colab' in sys.modules

def load_config() -> dict:
    """
    プロジェクトルートの config.yaml を読み込み、
    ローカル／Colab 両環境で共通に使えるよう data_base を動的に切り替えた設定辞書を返す。
    
    ※ config.yaml には以下のキーを追加してください：
       data_base_local: "<ローカルのパス>/project_10k_to_1m_data"
       data_base_colab: "/content/drive/MyDrive/project_10k_to_1m_data"
       
    その他のディレクトリ設定は既存のまま、
    mt5_data_dir: "${data_base}/raw"
    processed_dir: "${data_base}/processed"
    model_dir: "${processed_dir}/models"
    report_dir: "${processed_dir}/reports"
    としておき、resolve_path() で展開します。
    """
    # 1) config.yaml 読み込み
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 2) data_base をローカル or Colab 用に上書き
    if is_colab_env():
        # Colab 実行時には data_base_colab を使う
        if 'data_base_colab' not in cfg:
            raise KeyError("config.yaml に 'data_base_colab' が定義されていません")
        cfg['data_base'] = cfg['data_base_colab']
    else:
        # ローカル実行時には data_base_local を使う
        if 'data_base_local' not in cfg:
            raise KeyError("config.yaml に 'data_base_local' が定義されていません")
        cfg['data_base'] = cfg['data_base_local']

    return cfg

def resolve_path(template: str, cfg: dict) -> Path:
    """
    config.yaml のテンプレート文字列 ("${data_base}/raw" など) を
    load_config() で設定された cfg['data_base'] を用いて展開し、
    実際のパス(Path)を返す。
    """
    path_str = template
    for key, val in cfg.items():
        if isinstance(val, str):
            path_str = path_str.replace(f"${{{key}}}", val)
    return Path(path_str).resolve()
