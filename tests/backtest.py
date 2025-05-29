import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    model = joblib.load(args.model)
    with open(args.model.replace(".pkl", "_feature_cols.json")) as f:
        feature_cols = json.load(f)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    df["prob"] = model.predict_proba(df[feature_cols])[:, 1]
    df["signal"] = (df["prob"] > 0.5).astype(int)
    # シンプル資産曲線例
    # トレード損益例: Buyで+1, Sellで-1（実運用はTP/SL到達判定で実損益を計算）
    df["trade_pips"] = np.where(df["signal"] == 1, 1, -1)  # 仮
    df["equity"] = df["trade_pips"].cumsum()
    plt.plot(df["equity"])
    plt.title("資産曲線（仮）")
    plt.xlabel("Trade")
    plt.ylabel("Pips")
    plt.tight_layout()
    plt.show()
    print(f"[INFO] 累積獲得pips: {df['equity'].iloc[-1]:.2f}")
    print(f"[INFO] 最大ドローダウン: {abs(df['equity'] - df['equity'].cummax()).max():.2f}")

if __name__ == "__main__":
    main()
