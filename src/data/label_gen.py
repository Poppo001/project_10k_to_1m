# src/data/label_gen.py
import argparse
import pandas as pd
from pathlib import Path

def generate_labels(df: pd.DataFrame, tp_pips: int, sl_pips: int) -> pd.DataFrame:
    """TP/SL 到達順からラベルを付与"""
    pip_factor = 0.01 if "JPY" in df["symbol"].iloc[0] else 0.0001
    tp = tp_pips * pip_factor
    sl = sl_pips * pip_factor

    win_loss = []
    for open_price, high, low in zip(df["open"], df["high"], df["low"]):
        # ロング仮定: 先に TPorSL どっち？
        hit_tp = high - open_price >= tp
        hit_sl = open_price - low >= sl
        if hit_tp and hit_sl:
            win_loss.append(1 if high - open_price >= open_price - low else 0)
        elif hit_tp:
            win_loss.append(1)
        elif hit_sl:
            win_loss.append(0)
        else:
            win_loss.append(None)          # 到達せず → 使わない or 後で除外
    df["win_loss"] = win_loss
    df = df.dropna(subset=["win_loss"])
    df["label"] = df["win_loss"].astype(int)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="特徴量CSV (feat_*.csv)")
    parser.add_argument("--tp", type=int, default=30, help="TP 幅 (pips)")
    parser.add_argument("--sl", type=int, default=30, help="SL 幅 (pips)")
    args = parser.parse_args()

    in_path = Path(args.file)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    print(f"[INFO]  読込: {in_path}")
    df = pd.read_csv(in_path)

    df_labeled = generate_labels(df, args.tp, args.sl)
    print(f"[INFO] ラベル付与完了: {len(df_labeled)} rows")

    # 出力パス生成
    basename = in_path.stem.replace("feat_", "")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"labeled_{basename}.csv"
    df_labeled.to_csv(out_path, index=False)
    print(f"[INFO] 保存: {out_path}")

if __name__ == "__main__":
    main()
