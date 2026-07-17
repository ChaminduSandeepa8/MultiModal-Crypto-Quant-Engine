import ccxt.pro as ccxt
import asyncio
import redis.asyncio as redis
import json
import logging

# =====================================================================
# 🚀 LIVE DATA PRODUCER - Binance -> Redis (v2 - Fixed & Hardened)
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("Producer")

SYMBOL = 'BTC/USDT'
CHANNEL = 'market_data'
MAX_BACKOFF = 60  # seconds


async def _run_with_backoff(coro_fn, label, *args):
    """Exponential backoff wrapper - watch_* loops permanently down වුනොත් Binance ban නොවී ලස්සනට retry කරයි."""
    backoff = 5
    while True:
        try:
            await coro_fn(*args)
            backoff = 5  # success එකකින් පස්සේ backoff reset කරන්න
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"[{label}] error: {e}. Retrying in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)


async def watch_ticker(exchange, symbol, redis_client):
    ticker = await exchange.watch_ticker(symbol)
    data = {
        'type': 'ticker',
        'last': ticker['last'],
        'volume': ticker['baseVolume']
    }
    try:
        await redis_client.publish(CHANNEL, json.dumps(data))
    except redis.RedisError as e:
        log.error(f"[Redis publish/ticker] {e}")


async def watch_trades(exchange, symbol, redis_client):
    trades = await exchange.watch_trades(symbol)
    for trade in trades:
        data = {
            'type': 'trade',
            'price': trade['price'],
            'amount': trade['amount'],
            'side': trade['side'],
        }
        try:
            await redis_client.publish(CHANNEL, json.dumps(data))
        except redis.RedisError as e:
            log.error(f"[Redis publish/trade] {e}")


async def watch_order_book(exchange, symbol, redis_client):
    orderbook = await exchange.watch_order_book(symbol)
    data = {
        'type': 'orderbook',
        'bids': orderbook['bids'][:5],
        'asks': orderbook['asks'][:5]
    }
    try:
        await redis_client.publish(CHANNEL, json.dumps(data))
    except redis.RedisError as e:
        log.error(f"[Redis publish/orderbook] {e}")


async def loop_ticker(exchange, symbol, redis_client):
    while True:
        await watch_ticker(exchange, symbol, redis_client)


async def loop_trades(exchange, symbol, redis_client):
    while True:
        await watch_trades(exchange, symbol, redis_client)


async def loop_order_book(exchange, symbol, redis_client):
    while True:
        await watch_order_book(exchange, symbol, redis_client)


async def main():
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    exchange = ccxt.binance({
        'enableRateLimit': True,  # binance block වීම වළක්වයි
    })

    tasks = []
    try:
        # Markets මුලින්ම load කරගැනීම - race condition වළක්වයි
        await exchange.load_markets()
        log.info(f"🚀 {SYMBOL} - Ticker, Trades, OrderBook streaming ආරම්භයි")

        tasks = [
            asyncio.create_task(_run_with_backoff(loop_ticker, "Ticker", exchange, SYMBOL, redis_client)),
            asyncio.create_task(_run_with_backoff(loop_trades, "Trades", exchange, SYMBOL, redis_client)),
            asyncio.create_task(_run_with_backoff(loop_order_book, "OrderBook", exchange, SYMBOL, redis_client)),
        ]
        await asyncio.gather(*tasks)

    except asyncio.CancelledError:
        pass
    finally:
        log.info("🛑 Shutting down gracefully...")
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await exchange.close()
        await redis_client.aclose()
        log.info("✅ Exchange & Redis connections closed cleanly.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("👋 Producer stopped manually (Ctrl+C).")