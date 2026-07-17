import asyncio
import redis.asyncio as redis
import json
import collections
import torch
import numpy as np
import pandas as pd
import pandas_ta as ta
import math
import time
import logging
import os

# =====================================================================
# 🧠 SUPER QUANT ENGINE v3 — Institutional 1-Minute Bar Pipeline
# =====================================================================
# 🔴 [CRITICAL FIX] Train/Live Time-Scale Alignment:
#   BEFORE: raw sub-second ticker ticks → feature_window.append() per tick
#           500 ticks = minutes of noise. Model trained on 500 1-min bars = 8.3hrs.
#           COMPLETE TIME-SCALE MISMATCH → garbage live predictions.
#   AFTER: ticks aggregated into 1-minute OHLCV bars. Bar closes on minute boundary.
#          feature_window.append() per 1-minute bar close ONLY.
#          500 bars = 500 minutes = 8.3hrs → perfect alignment with training.
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("QuantEngine")

# --- [Config] ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
CHANNEL = 'market_data'
SEQUENCE_LEN = 500          # 500 × 1-minute bars = 8.3 hours (matches train SEQUENCE_LEN)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SCALER_PATH = 'weights/feature_scaler.npz'

QUANT_FEATURES_20 = [
    'log_return', 'volume', 'spread_pct', 'cvd', 'cvd_velocity',
    'obi', 'trade_count', 'ema_9_dist', 'ema_50_dist', 'vwap_zscore',
    'vwap_slope', 'bb_pct_b', 'bb_width', 'rsi_14', 'stoch_rsi_k',
    'roc_10', 'atr_ratio', 'volatility_regime', 'time_sine', 'time_cosine'
]

# =====================================================================
# Feature Scaler (loaded from train_models.py Base-Train split)
# =====================================================================
_feat_min = None
_feat_max = None
_feat_range = None

def _load_scaler():
    global _feat_min, _feat_max, _feat_range
    if not os.path.exists(SCALER_PATH):
        log.warning(f"⚠️  [{SCALER_PATH}] හමු නොවීය! train_models.py ක්‍රියාත්මක කරන්න. Local fallback mode.")
        return False
    data = np.load(SCALER_PATH)
    _feat_min   = data['min']
    _feat_max   = data['max']
    _feat_range = _feat_max - _feat_min + 1e-7
    log.info(f"✅ Feature Scaler loaded from [{SCALER_PATH}] — {len(_feat_min)} features.")
    return True


# =====================================================================
# ✅ [FIX #1] 1-Minute Bar Aggregator
# Sub-second ticks → OHLCV + orderflow bars aligned to minute boundary.
# =====================================================================
class MinuteBarAggregator:
    """
    Raw trade/ticker/orderbook ticks → closed 1-minute OHLCV bars.
    Bar closes when UTC minute changes (matches Binance 1m kline boundary).
    Returns completed bar dict on close, None otherwise.

    Bar fields (matches training columns):
        open, high, low, close  : price (float)
        volume                  : total traded volume (float)
        taker_buy_volume        : buy-side volume for OBI proxy (float)
        spread_pct              : (best_ask - best_bid) / close at bar close (float)
        obi                     : taker_buy_volume / volume (float, 0–1)
        trade_count             : number of trades in bar (int)
        cvd_delta               : net buy volume this bar (float) — accumulated daily
        minute_ts               : bar open UTC minute (int, for time features)
    """

    def __init__(self):
        self.reset()
        self.current_minute = -1    # UTC minute marker (0-1439)
        self.best_bid = 0.0
        self.best_ask = 0.0

    def reset(self):
        self.bar_open   = None
        self.bar_high   = -np.inf
        self.bar_low    = np.inf
        self.bar_close  = None
        self.bar_vol    = 0.0
        self.buy_vol    = 0.0
        self.trade_cnt  = 0

    def update_orderbook(self, bids, asks):
        if bids and asks:
            try:
                self.best_bid = float(bids[0][0])
                self.best_ask = float(asks[0][0])
            except (IndexError, ValueError):
                pass

    def on_trade(self, price: float, amount: float, is_buy: bool):
        """Incorporate a trade tick into the current open bar."""
        if self.bar_open is None:
            self.bar_open = price
        self.bar_high  = max(self.bar_high, price)
        self.bar_low   = min(self.bar_low, price)
        self.bar_close = price
        self.bar_vol  += amount
        self.trade_cnt += 1
        if is_buy:
            self.buy_vol += amount

    def on_ticker(self, price: float) -> dict | None:
        """
        Called on each ticker message. Returns a closed bar dict when the minute
        boundary crosses; returns None otherwise.
        """
        t = time.gmtime()
        minute_of_day = t.tm_hour * 60 + t.tm_min

        if self.bar_open is None:
            # First tick ever — start fresh bar
            self.bar_open  = price
            self.bar_high  = price
            self.bar_low   = price
            self.bar_close = price
            self.current_minute = minute_of_day
            return None

        # Update last price in current bar
        self.bar_high  = max(self.bar_high, price)
        self.bar_low   = min(self.bar_low, price)
        self.bar_close = price

        if minute_of_day == self.current_minute:
            return None  # Same minute — keep accumulating

        # ── Minute boundary crossed → close bar ──────────────────────────
        vol = max(self.bar_vol, 1e-9)  # guard division-by-zero

        # ✅ [FIX #2] spread_pct / obi from real live data (not constants).
        spread_pct = 0.0
        if self.best_ask > self.best_bid > 0 and self.bar_close > 0:
            spread_pct = (self.best_ask - self.best_bid) / self.bar_close
        elif self.bar_high > self.bar_low and self.bar_close > 0:
            # fallback: use bar's high-low range as synthetic spread proxy
            spread_pct = (self.bar_high - self.bar_low) / self.bar_close

        obi = float(np.clip(self.buy_vol / vol, 0.0, 1.0))

        completed_bar = {
            'open'            : self.bar_open,
            'high'            : self.bar_high,
            'low'             : self.bar_low,
            'close'           : self.bar_close,
            'volume'          : vol,
            'taker_buy_volume': self.buy_vol,
            'spread_pct'      : spread_pct,
            'obi'             : obi,
            'trade_count'     : self.trade_cnt,
            'cvd_delta'       : self.buy_vol - (vol - self.buy_vol),  # net buy volume this bar
            'minute_ts'       : self.current_minute,                  # previous minute (closing bar)
        }

        # Start new bar at current tick price
        self.reset()
        self.bar_open   = price
        self.bar_high   = price
        self.bar_low    = price
        self.bar_close  = price
        self.current_minute = minute_of_day
        return completed_bar


# =====================================================================
# Quant Feature Computation (on the 1-minute bar window)
# =====================================================================
def compute_features(bar_window: collections.deque) -> pd.DataFrame:
    """
    Deque of 1-minute bar dicts → Super-20 quant features DataFrame.
    Input columns: open, high, low, close, volume, spread_pct, obi,
                   trade_count, cvd (daily cumsum), minute_ts
    Must exactly mirror train_models.py build_quant_features().
    """
    df = pd.DataFrame(list(bar_window))

    # ── Basic price features ──────────────────────────────────────────
    df['log_return']    = ta.log_return(df['close'])
    df['cvd_velocity']  = df['cvd'].diff(periods=5).fillna(0)

    # ── Spread & OBI (already computed per-bar) ──────────────────────
    # spread_pct and obi come directly from MinuteBarAggregator → no constants.

    # ── Trend (EMA distances) ─────────────────────────────────────────
    df['ema_9_dist']  = (df['close'] - ta.ema(df['close'], length=9))  / (df['close'] + 1e-9)
    df['ema_50_dist'] = (df['close'] - ta.ema(df['close'], length=50)) / (df['close'] + 1e-9)

    # ── Session VWAP (rolling 480-bar ≈ 8 hours, mirrors training daily VWAP) ───
    VWAP_WINDOW = min(480, len(df))
    df['_pv']   = df['close'] * df['volume']
    df['vwap']  = (df['_pv'].rolling(VWAP_WINDOW, min_periods=1).sum() /
                   (df['volume'].rolling(VWAP_WINDOW, min_periods=1).sum() + 1e-7))
    df.drop(columns=['_pv'], inplace=True)
    df['vwap_zscore'] = (df['close'] - df['vwap']) / (df['close'].rolling(20).std() + 1e-7)
    df['vwap_slope']  = (df['vwap'] - df['vwap'].shift(5)).fillna(0) / (df['close'] + 1e-9)

    # ── Volatility (Bollinger + ATR) ──────────────────────────────────
    bb = ta.bbands(df['close'], length=20)
    if bb is not None:
        bbp = [c for c in bb.columns if c.startswith('BBP_')]
        bbb = [c for c in bb.columns if c.startswith('BBB_')]
        df['bb_pct_b'] = bb[bbp[0]] if bbp else 0.5
        df['bb_width'] = bb[bbb[0]] if bbb else 0.0
    else:
        df['bb_pct_b'] = 0.5
        df['bb_width'] = 0.0

    # ATR computation using high/low/close (correct; matches training)
    atr_s = ta.atr(df['high'], df['low'], df['close'], length=7)
    atr_l = ta.atr(df['high'], df['low'], df['close'], length=21)
    df['atr_ratio']        = atr_s / (df['close'] + 1e-9)
    df['volatility_regime'] = atr_s / (atr_l + 1e-7)

    # ── Momentum ──────────────────────────────────────────────────────
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    stoch = ta.stochrsi(df['close'], length=14)
    try:
        k_col = [c for c in stoch.columns if 'STOCHRSIk' in c][0]
        df['stoch_rsi_k'] = stoch[k_col]
    except (IndexError, AttributeError, TypeError):
        df['stoch_rsi_k'] = 0.5
    df['roc_10'] = ta.roc(df['close'], length=10)

    # ✅ [FIX #4] time_sine/time_cosine — minute precision (matches fixed train_models.py)
    df['time_sine']   = np.sin(2 * np.pi * df['minute_ts'] / 1440.0)
    df['time_cosine'] = np.cos(2 * np.pi * df['minute_ts'] / 1440.0)

    final_df = df[QUANT_FEATURES_20].bfill().fillna(0).replace([np.inf, -np.inf], 0.0)
    return final_df


def to_model_tensor(final_df: pd.DataFrame) -> torch.Tensor:
    """
    DataFrame [500, 20] → normalized [1, 500, 20] GPU tensor.
    Uses training-time scaler (feature_scaler.npz) for identical normalization.
    """
    raw = final_df.values  # [500, 20] numpy

    if _feat_min is not None:
        normalized = (raw - _feat_min) / _feat_range
    else:
        feat_min   = raw.min(axis=0)
        feat_range = raw.max(axis=0) - feat_min + 1e-7
        normalized = (raw - feat_min) / feat_range
        log.warning("⚠️  Scaler not loaded — using local window normalization (inaccurate).")

    normalized = np.clip(normalized, 0.0, 1.0)
    tensor_data = torch.tensor(normalized, dtype=torch.float32).to(DEVICE)
    return tensor_data.unsqueeze(0)


# =====================================================================
# Main Async Loop
# =====================================================================
async def main():
    _load_scaler()

    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(CHANNEL)

    log.info("🧠 [SUPER QUANT ENGINE v3] AI Consumer සක්‍රීයයි!")
    log.info(f"⚡ 1-Minute Bar Aggregation Active | Device: {DEVICE}")
    log.info(f"📊 {SEQUENCE_LEN} × 1-min bars = {SEQUENCE_LEN/60:.1f}hrs history window")

    # ── Session-level state ──────────────────────────────────────────
    aggregator   = MinuteBarAggregator()
    session_cvd  = 0.0                      # daily-reset CVD accumulator
    last_day     = time.gmtime().tm_yday    # UTC day tracker for CVD reset

    # ✅ Each element in bar_window is a completed 1-minute bar dict.
    bar_window = collections.deque(maxlen=SEQUENCE_LEN)

    bars_received = 0

    while True:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            log.warning(f"Redis connection issue: {e}. Retrying in 3s...")
            await asyncio.sleep(3)
            continue

        if message is None:
            await asyncio.sleep(0.0005)
            continue

        try:
            data_dict = json.loads(message['data'])
        except (json.JSONDecodeError, TypeError):
            continue

        msg_type = data_dict.get('type')

        # ── A. Live Trades → feed aggregator ────────────────────────
        if msg_type == 'trade':
            try:
                price  = float(data_dict['last'])
                amount = float(data_dict['amount'])
                is_buy = data_dict.get('side') == 'buy'
                aggregator.on_trade(price, amount, is_buy)
            except (ValueError, KeyError, TypeError):
                continue

        # ── B. Order Book → update aggregator's best bid/ask ────────
        elif msg_type == 'orderbook':
            try:
                bids = data_dict.get('bids', [])
                asks = data_dict.get('asks', [])
                aggregator.update_orderbook(bids, asks)
            except (ValueError, IndexError, KeyError, TypeError):
                continue

        # ── C. Ticker → attempt bar close ────────────────────────────
        elif msg_type == 'ticker':
            try:
                last_price = float(data_dict['last'])
            except (ValueError, KeyError, TypeError):
                continue

            # ── UTC Midnight CVD reset (daily session) ────────────────
            t = time.gmtime()
            if t.tm_yday != last_day:
                session_cvd = 0.0
                last_day    = t.tm_yday
                log.info("🔄 [SESSION RESET] UTC Midnight — CVD daily reset!")

            # ── Try to close a 1-minute bar ───────────────────────────
            completed_bar = aggregator.on_ticker(last_price)
            if completed_bar is None:
                continue  # Still within current minute

            # ── Bar closed! Accumulate session CVD ───────────────────
            session_cvd += completed_bar['cvd_delta']
            bars_received += 1

            # ── Build the bar row for the feature window ──────────────
            bar_row = {
                'open'         : completed_bar['open'],
                'high'         : completed_bar['high'],
                'low'          : completed_bar['low'],
                'close'        : completed_bar['close'],
                'volume'       : completed_bar['volume'],
                'spread_pct'   : completed_bar['spread_pct'],
                'obi'          : completed_bar['obi'],
                'trade_count'  : completed_bar['trade_count'],
                'cvd'          : session_cvd,          # daily-scoped cumsum ✅
                'minute_ts'    : completed_bar['minute_ts'],
            }
            bar_window.append(bar_row)

            log.info(
                f"📊 Bar #{bars_received} closed | Close: {completed_bar['close']:.2f} | "
                f"Vol: {completed_bar['volume']:.2f} | OBI: {completed_bar['obi']:.3f} | "
                f"CVD: {session_cvd:.2f} | Window: {len(bar_window)}/{SEQUENCE_LEN}"
            )

            # ── Inference trigger (window full) ──────────────────────
            if len(bar_window) < SEQUENCE_LEN:
                continue

            try:
                final_df          = compute_features(bar_window)
                final_model_input = to_model_tensor(final_df)
            except Exception as e:
                log.error(f"⚠️ Feature computation failed: {e}")
                continue

            log.info(
                f"🚀 Tensor {tuple(final_model_input.shape)} ready for inference | "
                f"RSI: {final_model_input[0, -1, 13]:.4f} | "
                f"VolRegime: {final_model_input[0, -1, 17]:.4f} | "
                f"VWAPSlope: {final_model_input[0, -1, 10]:.4f}"
            )

            # Pass final_model_input to MasterJudge here:
            # e.g.: decision, confidence = await master_judge.execute_trade_decision(
            #           final_model_input, redis_client)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("👋 Engine stopped manually.")
        
        
        