# src/backtest.py

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="バックテストスクリプト")
    parser.add_argument(
        "--csv", required=True,
        help="入力のラベル付き特徴量CSVファイルパス（例: data/processed/selfeat_USDJPY_H1_100000.csv）"
    )
    parser.add_argument(
        "--model", required=True,
        help="使用する学習済みモデルファイルパス（.pkl）（例: data/models/xgb_model_5000.pkl）"
    )
    parser.add_argument(
        "--report", required=True,
        help="出力するバックテストレポートJSONのパス（例: data/reports/backtest_report_5000.json）"
    )
    parser.add_argument(
        "--curve_out", required=True,
        help="出力する損益曲線PNGのパス（例: data/reports/backtest_curve_5000.png）"
    )
    args = parser.parse_args()

    input_path = Path(args.csv)
    model_path = Path(args.model)
    report_path = Path(args.report)
    curve_path = Path(args.curve_out)

    # 入力ファイルとモデルファイルの存在チェック
    if not input_path.exists():
        print(f"[ERROR] 入力CSVファイルが見つかりません: {input_path}")
        return
    if not model_path.exists():
        print(f"[ERROR] モデルファイルが見つかりません: {model_path}")
        return

    # 出力先ディレクトリを作成
    for p in [report_path.parent, curve_path.parent]:
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    df = pd.read_csv(input_path)

    # 特徴量リスト読み込み
    feat_json_path = model_path.with_suffix(".json")
    if not feat_json_path.exists():
        print(f"[ERROR] 特徴量リストJSONが見つかりません: {feat_json_path}")
        return
    with open(feat_json_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # 欠損値を持つ行を削除
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # モデル読み込み
    model = joblib.load(model_path)

    # 予測確率と売買シグナル計算
    df["prob"] = model.predict_proba(df[feature_cols])[:, 1]
    df["signal"] = (df["prob"] > 0.5).astype(int)

    # シンプル資産曲線例
    # 仮: Buy (signal=1) で +1 pips、Sell (signal=0) で -1 pips
    df["trade_pips"] = np.where(df["signal"] == 1, 1, -1)

    # 累積損益（equity）
    df["equity"] = df["trade_pips"].cumsum()

    # 損益曲線を描画して保存
    plt.figure(figsize=(8, 4))
    plt.plot(df["equity"], linewidth=1)
    plt.title("資産曲線（仮）")
    plt.xlabel("トレード番号")
    plt.ylabel("累積獲得pips")
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()

    # 最終の累積獲得pips
    cumulative_pips = float(df["equity"].iloc[-1])

    # 最大ドローダウン計算
    running_max = df["equity"].cummax()
    drawdowns = running_max - df["equity"]
    max_drawdown = float(drawdowns.max())

    print(f"[INFO] 累積獲得pips: {cumulative_pips:.2f}")
    print(f"[INFO] 最大ドローダウン: {max_drawdown:.2f}")

    # レポートをJSONで保存
    report = {
        "cumulative_pips": cumulative_pips,
        "max_drawdown": max_drawdown,
        "n_trades": int(len(df)),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] バックテストレポートを保存しました: {report_path}")

if __name__ == "__main__":
    main()
