import json, glob
from pathlib import Path
import pandas as pd
BASE = Path("project_10k_to_1m_data") 
REP  = BASE/"reports"
PROC = BASE/"processed"

# ■ モデル評価集約
eval_recs = []
for p in REP.glob("eval_report_*.json"):
    d = json.load(open(p, encoding="utf-8"))
    key = p.stem.replace("eval_report_","")
    symbol, timeframe, bars, ts = key.split("_",3)
    rec = {
        "symbol":symbol, "timeframe":timeframe, "bars":int(bars), "ts":ts,
        "accuracy": d["classification_report"]["accuracy"],
        "tn": d["confusion_matrix"]["true_negative"],
        "fp": d["confusion_matrix"]["false_positive"],
        "fn": d["confusion_matrix"]["false_negative"],
        "tp": d["confusion_matrix"]["true_positive"]
    }
    eval_recs.append(rec)
df_eval = pd.DataFrame(eval_recs)

# ■ バックテスト集約
bt_recs = []
for p in REP.glob("backtest_*.json"):
    d = json.load(open(p, encoding="utf-8"))
    key = p.stem.replace("backtest_","")
    symbol, timeframe, bars, ts = key.split("_",3)
    rec = {
        "symbol":symbol, "timeframe":timeframe, "bars":int(bars), "ts":ts,
        **d
    }
    bt_recs.append(rec)
df_bt = pd.DataFrame(bt_recs)

# ■ 特徴量選択結果
df_featsel = pd.read_csv(PROC/"auto_feature_selection_results.csv")

# マージして表示
df_summary = df_eval.merge(df_bt, on=["symbol","timeframe","bars","ts"])
from ace_tools import display_dataframe_to_user
display_dataframe_to_user("Summary", df_summary)
