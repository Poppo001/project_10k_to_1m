# src/data/fetch_and_label.py

import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import yaml
import MetaTrader5 as mt5
import sys

def load_config():
    root = Path(__file__).resolve().parents[2]  # project root
    cfg_path = root / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def is_colab():
    return "google.colab" in sys.modules

def resolve_data_root(cfg):
    if is_colab():
        return Path("/content/drive/MyDrive") / cfg["data_root_colab"]
    else:
        return Path(cfg["data_root_local"]).resolve()

def fetch_mt5_data(symbol, timeframe, bars):
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--bars", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config()
    data_root = resolve_data_root(cfg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = data_root / "raw" / args.symbol / args.timeframe
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_name = f"{args.symbol}_{args.timeframe}_{args.bars}_{ts}.csv"
    save_path = save_dir / csv_name

    mt5.initialize()
    df = fetch_mt5_data(args.symbol, args.timeframe, args.bars)
    mt5.shutdown()

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✔ Saved: {save_path}")

if __name__ == "__main__":
    main()
