#!/usr/bin/env python3
# src/data/backtest.py

import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.family'] = 'MS Gothic'

def load_config():
    """
    プロジェクト直下の config.yaml を読み込んで辞書で返す
    """
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser(description="Backtest using config.yaml defaults")
    parser.add_argument("--csv",       required=True, help="selfeat_*.csv path")
    parser.add_argument("--model",     required=True, help="trained model .pkl")
    parser.add_argument("--report",    required=True, help="output JSON path")
    parser.add_argument("--curve_out", required=True, help="output PNG path")
    args = parser.parse_args()

    # config.yaml からデフォルト値を読み込む
    cfg        = load_config()
    tp_pips    = float(cfg.get("tp",        30.0))
    sl_pips    = float(cfg.get("sl",        30.0))
    spread     = float(cfg.get("spread",     0.2))
    commission = float(cfg.get("commission", 0.1))
    slippage   = float(cfg.get("slippage",   0.5))

    sel_path    = Path(args.csv)
    model_path  = Path(args.model)
    report_path = Path(args.report)
    curve_path  = Path(args.curve_out)

    # 入力ファイル存在チェック
    for p in (sel_path, model_path):
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            return

    # 出力ディレクトリを作成
    report_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    feat_df = pd.read_csv(sel_path)
    model   = joblib.load(model_path)

    # 特徴量リスト読み込み
    feat_json = model_path.with_name(model_path.stem + "_feature_cols.json")
    features  = json.loads(feat_json.read_text(encoding="utf-8"))

    # signal列がなければ予測で追加
    if "signal" not in feat_df.columns:
        feat_df["signal"] = model.predict(feat_df[features])

    # raw CSV を推定パスから読み込み
    parts   = sel_path.stem.split("_")
    # selfeat_SYMBOL_TIMEFRAME_BARS[_timestamp]
    symbol, timeframe, bars = parts[1], parts[2], parts[3]
    raw_csv = sel_path.parents[2] / "raw" / f"{symbol}_{timeframe}_{bars}.csv"
    raw_df  = pd.read_csv(raw_csv)

    # time でマージして trade-level DataFrame を作成
    df = pd.merge(raw_df, feat_df[["time","signal"]], on="time", how="inner")

    # Equity 曲線をシミュレーション
    equity = []
    cum    = 0.0
    n      = len(df)
    for i in range(n - 1):
        entry = df.at[i, "open"]
        sig   = int(df.at[i, "signal"])

        # TP/SL 到達価格を計算
        pip_unit = 0.0001  # USDJPY なら 1 pip = 0.0001
        if sig == 1:
            tp_price = entry + tp_pips * pip_unit
            sl_price = entry - sl_pips * pip_unit
        else:
            tp_price = entry - tp_pips * pip_unit
            sl_price = entry + sl_pips * pip_unit

        hit = None
        # TP/SL 到達判定
        for j in range(i + 1, n):
            high = df.at[j, "high"]
            low  = df.at[j, "low"]
            if sig == 1:
                if high >= tp_price:
                    hit =  tp_pips
                    break
                if low  <= sl_price:
                    hit = -sl_pips
                    break
            else:
                if low  <= tp_price:
                    hit =  tp_pips
                    break
                if high >= sl_price:
                    hit = -sl_pips
                    break

        # いずれも hit しなかったら終値で評価
        if hit is None:
            exit_price = df.at[n - 1, "close"]
            raw_pips   = (exit_price - entry) / pip_unit
            hit = raw_pips if sig == 1 else -raw_pips

        # ネット損益
        net = hit - spread - commission - slippage
        cum += net
        equity.append(cum)

    # プロット
    plt.figure(figsize=(8,4))
    plt.plot(equity, linewidth=1)
    plt.title("Equity Curve (Config Defaults)")
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative Pips")
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()

    # 指標計算
    cumulative_pips = cum
    running_max     = np.maximum.accumulate(equity)
    drawdowns       = running_max - equity
    max_dd          = float(np.max(drawdowns))

    print(f"[INFO] 累積獲得pips: {cumulative_pips:.2f}")
    print(f"[INFO] 最大ドローダウン: {max_dd:.2f}")

    # レポート作成・保存
    report = {
        "cumulative_pips": cumulative_pips,
        "max_drawdown":    max_dd,
        "n_trades":        len(equity),
        "tp_pips":         tp_pips,
        "sl_pips":         sl_pips,
        "spread":          spread,
        "commission":      commission,
        "slippage":        slippage
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] バックテストレポートを保存: {report_path}")

if __name__ == "__main__":
    main()
