# project_10k_to_1m/paper_trade/mt5watch.py

import os
import yaml
import logging
from logging.handlers import RotatingFileHandler
import MetaTrader5 as mt5
import pandas as pd
import time
import datetime
import pytz
import pickle
import requests

from src.data.feature_gen import generate_features

# ─── 設定読み込み ───────────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, encoding='utf-8-sig') as f:
    cfg = yaml.safe_load(f)

# ─── ログ設定 ─────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(cfg['output']['log_file']), exist_ok=True)
logger = logging.getLogger('mt5watch')
logger.setLevel(getattr(logging, cfg['logging']['level']))
handler = RotatingFileHandler(
    cfg['output']['log_file'],
    maxBytes=cfg['logging']['max_bytes'],
    backupCount=cfg['logging']['backup_count'],
    encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())

# ─── モデルロード ─────────────────────────────────────────────────────
with open(cfg['model']['path'], 'rb') as f:
    model = pickle.load(f)
threshold_long = cfg['model']['threshold_long']
threshold_short = cfg['model']['threshold_short']

# ─── MT5 初期化 ───────────────────────────────────────────────────────
init_args = {
    "path": cfg['mt5']['path'],
    "login": cfg['mt5']['login'],
    "password": cfg['mt5']['password'],
    "server": cfg['mt5']['server']
}
if not mt5.initialize(**init_args):
    logger.error(f"MT5 initialize failed: {mt5.last_error()}")
    raise SystemExit("MT5 の初期化に失敗しました。")
logger.info("MT5 initialized successfully.")

# ─── CSV 出力準備 ─────────────────────────────────────────────────────
signal_csv = cfg['output']['signal_csv']
os.makedirs(os.path.dirname(signal_csv), exist_ok=True)
if not os.path.exists(signal_csv):
    pd.DataFrame(columns=[
        'time','signal','price','spread','slippage','exec_price'
    ]).to_csv(signal_csv, index=False)

# ─── Webhook（Slackのみ）────────────────────────────────────────────────
'''
def send_webhook(message: str):
    url     = cfg['webhook']['slack_url']
    timeout = cfg['webhook']['timeout_seconds']
    for attempt in range(cfg['webhook']['retry_count']):
        try:
            resp = requests.post(
                url,
                json={'text': message},
                timeout=timeout
            )
            if resp.status_code == 200:
                return
            else:
                logger.warning(f"Webhook responded {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Webhook send failed attempt {attempt+1}: {e}")
            time.sleep(1)
'''
# ─── 最新バー取得関数 ─────────────────────────────────────────────────
def fetch_latest_bar():
    symbol   = cfg['mt5']['symbol']
    tf_const = getattr(mt5, f"TIMEFRAME_{cfg['mt5']['timeframe']}")
    rates    = mt5.copy_rates_from_pos(symbol, tf_const, 1, 1)
    bar      = rates[0]
    tz       = pytz.timezone(cfg['mt5']['timezone'])
    t        = datetime.datetime.fromtimestamp(bar['time'], tz)
    return pd.Series({
        'time':  t,
        'open':  bar['open'],
        'high':  bar['high'],
        'low':   bar['low'],
        'close': bar['close']
    })

# ─── シグナル重複防止用変数 ───────────────────────────────────────────
last_signal_time = None

# ─── シグナル判定／推論関数 ─────────────────────────────────────────
def decide_signal(bar: pd.Series):
    global last_signal_time
    if last_signal_time == bar['time']:
        return None
    try:
        feats     = generate_features(bar, **cfg['features'])
        X         = feats.values.reshape(1, -1)
        proba_long = model.predict_proba(X)[0][1]
    except Exception as e:
        logger.error(f"推論エラー: {e}")
        send_webhook(f"推論エラー: {e}")
        return None

    signal = None
    if proba_long >= threshold_long:
        signal = 'LONG'
    elif proba_long <= threshold_short:
        signal = 'SHORT'

    if signal:
        last_signal_time = bar['time']
    return signal

# ─── シグナル書き込み関数 ───────────────────────────────────────────
def append_signal(rec: dict):
    try:
        pd.DataFrame([rec]).to_csv(signal_csv, mode='a', header=False, index=False)
    except Exception as e:
        logger.error(f"CSV書き込みエラー: {e}")
        send_webhook(f"CSV書き込みエラー: {e}")

# ─── メインループ ───────────────────────────────────────────────────
try:
    logger.info("mt5watch started.")
    while True:
        bar = fetch_latest_bar()
        sig = decide_signal(bar)
        if sig:
            spread     = cfg['trade']['spread_pips']
            slip       = cfg['trade']['slippage_pips']
            exec_price = (
                bar['close'] + ((spread + slip)/10000) if sig == 'LONG'
                else bar['close'] - ((spread + slip)/10000)
            )
            record = {
                'time':     bar['time'],
                'signal':   sig,
                'price':    bar['close'],
                'spread':   spread,
                'slippage': slip,
                'exec_price': exec_price
            }
            append_signal(record)

            msg = f"{bar['time']} {sig} @ {bar['close']} → exec {exec_price}"
            logger.info(msg)
            send_webhook(msg)

        # 1分ちょうどに動かす
        now        = datetime.datetime.now(pytz.timezone(cfg['mt5']['timezone']))
        sleep_sec  = 60 - now.second
        time.sleep(sleep_sec)

except KeyboardInterrupt:
    logger.info("mt5watch stopped by user.")

except Exception as e:
    logger.exception(f"予期せぬ例外: {e}")
    send_webhook(f"mt5watch fatal error: {e}")

finally:
    mt5.shutdown()
    logger.info("MT5 shutdown complete.")
