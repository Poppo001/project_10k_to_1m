#!/usr/bin/env python3
# src/models/backtest.py

import argparse
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# ── 日本語フォント設定 ─────────────────────────────────
plt.rcParams['font.family'] = 'MS Gothic'

# ── プロジェクトルートを sys.path に追加 ────────────────────
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]   # src/models → src → プロジェクトルート
sys.path.insert(0, str(project_root))

# ── 共通ユーティリティ読み込み ───────────────────────────────
from src.utils.common import load_config, resolve_data_root, get_latest_file

def calc_equity_curve(df: pd.DataFrame, model) -> pd.DataFrame:
    """予測ラベルでエクイティカーブを計算"""
    X = df.drop(columns=["time", "label", "future_return"])
    # ラベル
    try:
        proba = model.predict_proba(X)[:, 1]
        pred  = (proba > 0.5).astype(int)
    except AttributeError:
        pred  = model.predict(X).astype(int)

    df_result = df.copy()
    df_result["pred"]    = pred
    df_result["correct"] = (pred == df["label"]).astype(int)
    df_result["ret"]     = np.where(pred == 1, df_result["future_return"], 0)
    df_result["equity"]  = df_result["ret"].cumsum()
    return df_result

def plot_equity_curve(df_result: pd.DataFrame, out_path: Path):
    """エクイティカーブを PNG 保存"""
    plt.figure(figsize=(10, 5))
    plt.plot(df_result["time"], df_result["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Pips")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    # ── 引数（symbol/timeframe/bars）────────────────────────────
    parser = argparse.ArgumentParser(
        description="Auto-path backtest: symbol/timeframe/bars だけで最新ファイルを自動選択"
    )
    parser.add_argument("--symbol",    required=True, help="通貨ペア (例: USDJPY)")
    parser.add_argument("--timeframe", required=True, help="時間軸  (例: M5, H1)")
    parser.add_argument("--bars",      type=int,  required=True, help="バー数      (例: 100000)")
    args = parser.parse_args()

    symbol    = args.symbol
    timeframe = args.timeframe
    bars      = args.bars
    print(f"\n▶ symbol={symbol}, timeframe={timeframe}, bars={bars}")

    # ── config と data_root を取得 ─────────────────────────────
    cfg       = load_config()
    data_root = resolve_data_root(cfg)

    # ── 処理対象ディレクトリ ───────────────────────────────────
    feat_dir = data_root / "processed" / symbol / timeframe

    # ── 最新の特徴量 CSV を自動取得 ─────────────────────────────
    feat_path = get_latest_file(
        feat_dir,
        prefix=f"selfeat_{symbol}_{timeframe}_{bars}_",
        suffix=".csv"
    )
    if feat_path is None:
        print(f"❌ 特徴量CSVが見つかりません: {feat_dir}")
        return

    # ── 最新モデル & 特徴量リストを自動取得 ────────────────────
    model_path = get_latest_file(
        feat_dir,
        prefix=f"xgb_model_{symbol}_{timeframe}_{bars}_",
        suffix=".pkl"
    )
    feats_path = get_latest_file(
        feat_dir,
        prefix=f"xgb_model_{symbol}_{timeframe}_{bars}_",
        suffix="_features.json"
    )
    if model_path is None or feats_path is None:
        print("❌ モデル(.pkl)または特徴量リスト(.json)が見つかりません")
        return

    print(f"✔ Using CSV   : {feat_path.name}")
    print(f"✔ Using Model : {model_path.name}")
    print(f"✔ Using Feats : {feats_path.name}")

    # ── DataFrame 読み込み & カラム準備 ─────────────────────────
    df = pd.read_csv(feat_path, parse_dates=["time"])
    # future_return 列をコピー
    if "future_return" not in df.columns and "return" in df.columns:
        df["future_return"] = df["return"]
    # label 列を自動生成
    if "label" not in df.columns:
        df["label"] = (df["future_return"] > 0).astype(int)

    # ── モデル & 特徴量ロード ─────────────────────────────────
    model         = joblib.load(model_path)
    used_features = json.loads(feats_path.read_text(encoding="utf-8"))

    # ── 必要な列だけ抽出 ─────────────────────────────────────
    df = df[["time", "label"] + used_features + ["future_return"]]

    # ── エクイティカーブ計算 & 指標算出 ─────────────────────────
    df_result    = calc_equity_curve(df, model)
    final_eq     = df_result["equity"].iloc[-1]
    win_rate     = df_result["correct"].mean()
    max_dd       = (df_result["equity"].cummax() - df_result["equity"]).max()
    sharpe       = (
        df_result["ret"].mean() / df_result["ret"].std()
        if df_result["ret"].std() != 0 else 0
    )

    print(f"✔ Final Equity : {final_eq:.2f} pips")
    print(f"✔ Win Rate     : {win_rate:.2%}")
    print(f"✔ Max Drawdown : {max_dd:.2f} pips")
    print(f"✔ Sharpe Ratio : {sharpe:.2f}")

    # ── 出力ファイル名を自動生成 ───────────────────────────────
    timestamp = feat_path.stem.split("_")[-1]
    png_out   = feat_dir / f"equity_{symbol}_{timeframe}_{bars}_{timestamp}.png"
    json_out  = feat_dir / f"report_{symbol}_{timeframe}_{bars}_{timestamp}.json"

    # ── グラフ & レポート保存 ─────────────────────────────────
    plot_equity_curve(df_result, png_out)
    json_out.write_text(
        json.dumps({
            "final_equity": final_eq,
            "win_rate":      win_rate,
            "max_drawdown":  max_dd,
            "sharpe_ratio":  sharpe
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✔ Saved curve  : {png_out.name}")
    print(f"✔ Saved report : {json_out.name}\n")

if __name__ == "__main__":
    main()
