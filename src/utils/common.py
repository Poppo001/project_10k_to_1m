# src/utils/common.py

import sys
import os
from pathlib import Path
import yaml

def is_colab() -> bool:
    """
    Colab 上かどうかを判定。
    """
    # Drive マウント先のパスをキーに判定
    return os.getcwd().startswith("/content/drive/MyDrive")

def load_config() -> dict:
    """
    プロジェクト直下の config.yaml を読み込んで辞書で返す
    """
    here = Path(__file__).resolve().parent.parent  # project_root/src/utils → project_root
    cfg_path = here / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def resolve_path(path_template: str, cfg: dict) -> Path:
    """
    config.yaml 中のパス指定文字列（"${…}" を含む）を展開して絶対パス化するユーティリティ。
    run_pipeline.py 等で使われます。
    例:
      path_template = "${data_base}/raw"
      cfg["data_base"] = "C:/Projects/project_10k_to_1m_data"
    """
    # ${…} の簡易展開
    s = path_template
    for key, val in cfg.items():
        placeholder = "${" + key + "}"
        if placeholder in s:
            s = s.replace(placeholder, str(val))
    # Colab 判定でドライブルートを適宜付与
    if is_colab():
        # もし相対パスなら MyDrive ルートを付与
        if not s.startswith("/content/drive"):
            s = "/content/drive/MyDrive/" + s.lstrip("/\\")
    return Path(s).resolve()

def resolve_data_root(cfg: dict) -> Path:
    """
    FXデータのルートディレクトリを返す（MT5データや processed データの親フォルダ）。
    train_model.py や backtest.py から呼び出されます。
    config.yaml に「data_base_local」「data_base_colab」を定義しておく前提です。
    """
    # Colab であれば data_base_colab、ローカルなら data_base_local を利用
    if is_colab():
        key = "data_base_colab"
    else:
        key = "data_base_local"
    base = cfg.get(key)
    if base is None:
        raise KeyError(f"[common.resolve_data_root] `{key}` is not set in config.yaml")
    return Path(base).resolve()

# 互換性のためのエイリアス
# 以前は resolve_data_root(cfg) しかなかった箇所で
# resolve_path も使いたいスクリプトがある場合に両方維持します。
