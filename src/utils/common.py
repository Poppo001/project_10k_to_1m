# src/utils/common.py

from pathlib import Path
import sys
import yaml

def is_colab():
    return 'google.colab' in sys.modules

def load_config():
    root = Path(__file__).resolve().parents[2]  # project root
    cfg_path = root / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def resolve_data_root(cfg):
    if is_colab():
        return Path("/content/drive/MyDrive") / cfg["data_root_colab"]
    else:
        return Path(cfg["data_root_local"]).resolve()
