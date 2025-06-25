#!/usr/bin/env python3
# src/utils/common.py

import yaml
from pathlib import Path

def load_config() -> dict:
    root = Path(__file__).resolve().parents[2]
    return yaml.safe_load((root/"config.yaml").read_text(encoding="utf-8"))

def resolve_path(template: str, cfg: dict) -> Path:
    s = template
    for k, v in cfg.items():
        if isinstance(v, str):
            s = s.replace(f"${{{k}}}", v)
    return Path(s).resolve()
