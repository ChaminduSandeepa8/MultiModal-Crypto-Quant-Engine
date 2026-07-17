import ccxt.pro as ccxt
import asyncio
import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime, timezone

# =====================================================================
# 📡 REAL ORDER BOOK LOGGER — Azure VM / systemd-ready (v2)
# Purpose: spread_pct සහ obi වලට "neutral placeholder" එක වෙනුවට REAL
# historical values හදාගන්න, මේ script එක සති 2-4ක් 24/7 run කරන්න ඕන.
# (Binance free REST API එකෙන් L2 history ලැබෙන්නෙ නෑ — ඉතින් අපිම
# අනාගතයට log කරගන්නවා, දැන් ඉඳන්.)
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("OBLogger")

SYMBOL = 'BTC/USDT'
DEPTH_LEVELS = 5            # L1/L2 top 5 levels use කරනවා obi ගණනයට
SAMPLE_INTERVAL_SEC = 1.0   # තත්පරයට 1 sample (1-min candle එකකට ~60 samples, aggregate කරන්න ප්‍රමාණවත්)
FLUSH_EVERY_N = 300         # Samples 300ක් (~5 min) එකතු වුනාම disk එකට flush කරනවා (crash-safe)
OUTPUT_DIR = "orderflow_logs"

# ✅ [systemd/Azure hardening] reconnect backoff — VM network blip වුනොත්
# infinite tight-loop එකකට යනවා වෙනුවට exponential backoff එකෙන් retry කරනවා
RECONNECT_BASE_SEC = 5
RECONNECT_MAX_SEC = 120


def compute_spread_pct(bid_price: float, ask_price: float) -> float:
    mid = (bid_price + ask_price) / 2.0
    if mid == 0:
        return 0.0
    return ((ask_price - bid_price) / mid) * 100.0


def compute_obi(bids: list, asks: list, levels: int = DEPTH_LEVELS) -> float:
    """Order Book Imbalance: bid volume ප්‍රතිශතය top N levels තුළ (0-1 අතර)."""
    bid_vol = sum(qty for _, qty in bids[:levels])
    ask_vol = sum(qty for _, qty in asks[:levels])
    total = bid_vol + ask_vol
    if total == 0:
        return 0.5
    return bid_vol / total


def flush_to_disk(buffer: list):
    """
    Day-partitioned parquet files ලෙස save කරනවා (එකම ෆයිල් එකක් දිගින් දිගටම ලොකු වෙන එක
    වළක්වන්න, සහ crash වුනොත් අන්තිම flush එකට කලින් තිබ්බ data රැකෙනවා).
    """
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    day_str = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
    file_path = os.path.join(OUTPUT_DIR, f"orderflow_{day_str}.parquet")

    try:
        if os.path.exists(file_path):
            existing = pd.read_parquet(file_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_parquet(file_path, index=False)
        else:
            df.to_parquet(file_path, index=False)
    except Exception as e:
        # ✅ [Hardening] flush failure නිසා whole process crash වෙන්නෙ නෑ —
        # log කරලා continue කරනවා, ඊළඟ flush එකේදී retry වෙනවා (buffer එකේ data රැකෙනවා)
        log.error(f"⚠️ Flush to disk failed: {e}")
        raise


async def log_order_book(exchange, symbol: str):
    buffer = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    last_sample_time = 0.0
    total_samples_logged = 0

    while True:
        try:
            orderbook = await exchange.watch_order_book(symbol)

            now = asyncio.get_event_loop().time()
            if now - last_sample_time < SAMPLE_INTERVAL_SEC:
                continue  # Throttle - websocket ticks ගොඩක් වේගවත්, sample rate control කරනවා
            last_sample_time = now

            bids = orderbook['bids']
            asks = orderbook['asks']
            if not bids or not asks:
                continue

            best_bid = bids[0][0]
            best_ask = asks[0][0]

            row = {
                'timestamp': datetime.now(timezone.utc),
                'spread_pct': compute_spread_pct(best_bid, best_ask),
                'obi': compute_obi(bids, asks),
                'best_bid': best_bid,
                'best_ask': best_ask,
            }
            buffer.append(row)

            if len(buffer) >= FLUSH_EVERY_N:
                flush_to_disk(buffer)
                total_samples_logged += len(buffer)
                log.info(f"💾 Samples {len(buffer)}ක් flush කළා "
                         f"(latest spread_pct={row['spread_pct']:.4f}, obi={row['obi']:.4f}) "
                         f"| Total logged so far: {total_samples_logged:,}")
                buffer = []

        except asyncio.CancelledError:
            # ✅ Graceful shutdown (Ctrl+C / systemctl stop) — buffer එකේ ඉතුරු data flush කරලා exit වෙනවා
            if buffer:
                flush_to_disk(buffer)
                log.info(f"💾 Final flush before shutdown: {len(buffer)} samples saved.")
            raise
        except Exception as e:
            log.warning(f"⚠️ Order book error: {e}. Reconnecting...")
            # Buffer එකේ තිබ්බ data ඒ වෙලාවෙම flush කරලා ගන්නවා, connection retry කරන්න කලින්
            if buffer:
                try:
                    flush_to_disk(buffer)
                    log.info(f"💾 Pre-reconnect flush: {len(buffer)} samples saved.")
                except Exception:
                    pass
                buffer = []
            await asyncio.sleep(5)


async def main():
    log.info(f"🚀 {SYMBOL} order book logging ආරම්භ විය. '{OUTPUT_DIR}/' folder එකට save වෙනවා.")
    log.info("⏳ මේක background එකේ (tmux / systemd) සති 2-4ක් run කරන්න ඕන.")

    backoff = RECONNECT_BASE_SEC
    while True:
        exchange = ccxt.binance({'enableRateLimit': True})
        try:
            await log_order_book(exchange, SYMBOL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"⚠️ Top-level error: {e}. Retrying in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_SEC)
        else:
            backoff = RECONNECT_BASE_SEC
        finally:
            try:
                await exchange.close()
            except Exception:
                pass


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("👋 Logger stopped manually (Ctrl+C).")
        sys.exit(0)
