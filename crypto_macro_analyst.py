import asyncio
import redis.asyncio as redis
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import feedparser
import logging
import time
import json
import collections
import hashlib

# =====================================================================
# 🧠 CRYPTO MACRO ANALYST - v3 (Final: EMA Global Score + Hardened)
# Hardware target: RTX 3050 (FP16 Half-Precision ~250MB VRAM)
#
# Design decision: Mamba/TCN tick-level features (500-tick window) වලට
# news එකතු කරන්නෙ නෑ - frequency mismatch (news = minutes, ticks = seconds)
# noise බවට පත් වෙනවා. ඒ වෙනුවට මේ engine එක Redis key එකක් (JSON blob)
# විදිහට Global Macro Sentiment එක background process එකක් විදිහට
# maintain කරනවා, XGBoost meta-learner එක decision ගන්න වෙලාවට කෙලින්ම
# read කරගන්නවා. Clean decoupling - pipeline එකේ frequency ගැටලුවක් නෑ.
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("MacroAnalyst")

# --- [Config] ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_KEY_MACRO = 'macro_sentiment'   # single JSON blob - atomic read for XGBoost
POLL_INTERVAL_SEC = 60
MAX_BACKOFF = 300
VOLUME_WINDOW_SEC = 300                # news_volume_5min window
SEEN_CACHE_SIZE = 300
EMA_ALPHA = 0.3                        # අලුත් headlines වලට දෙන බර

MODEL_NAME = "ElKulako/cryptobert"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NEWS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptoslate.com/feed/",
    "https://decrypt.co/feed",
]


class CryptoMacroAnalyst:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        log.info(f"⚡ [INIT] Crypto-RoBERTa Model Load වීම ආරම්භයි... Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # FP16 - VRAM footprint ~250MB, 3050 ට ඉතා සැහැල්ලුයි
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        ).to(self.device)
        self.model.eval()

        # [FIX] deque + set combo - "අන්තිම N" කියන order guarantee කරන්න
        # (raw set slicing එකෙන් random titles delete වෙන bug එක fix කළා)
        self.seen_hashes = collections.deque(maxlen=SEEN_CACHE_SIZE)
        self.seen_set = set()

        vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if self.device.type == 'cuda' else 0
        log.info(f"✅ [SUCCESS] Crypto-RoBERTa සූදානම්! VRAM භාවිතය: {vram_mb:.1f} MB පමණයි.")

    def _remember(self, uid: str) -> bool:
        """True නම් අලුත් article එකක්, False නම් දැනටමත් processed."""
        if uid in self.seen_set:
            return False
        if len(self.seen_hashes) == self.seen_hashes.maxlen:
            oldest = self.seen_hashes[0]  # deque overflow වුනාම auto-evict වෙන id එක
            self.seen_set.discard(oldest)
        self.seen_hashes.append(uid)
        self.seen_set.add(uid)
        return True

    def analyze_headline(self, text: str):
        """-1.0 (Severe Bearish) සිට +1.0 (Strong Bullish) දක්වා continuous score."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)

        with torch.no_grad():
            # [FIX] torch.cuda.amp.autocast deprecated -> torch.amp.autocast
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(self.device.type == 'cuda')):
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)[0]

        # ElKulako/cryptobert labels -> 0: Bearish, 1: Neutral, 2: Bullish
        p_bearish, p_neutral, p_bullish = probs[0].item(), probs[1].item(), probs[2].item()
        sentiment_score = p_bullish - p_bearish
        confidence = max(p_bearish, p_neutral, p_bullish)
        return sentiment_score, confidence

    def fetch_latest_news(self) -> list:
        """RSS Feeds වලින් අලුත්ම (duplicate නොවන) headlines ලබාගැනීම."""
        new_headlines = []
        for url in NEWS_FEEDS:
            try:
                feed = feedparser.parse(url)
                if feed.bozo and not feed.entries:
                    log.warning(f"⚠️ Feed unreachable: {url}")
                    continue
                for entry in feed.entries[:5]:
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "")
                    if not title:
                        continue
                    uid = hashlib.md5(f"{title}|{link}".encode()).hexdigest()
                    if self._remember(uid):
                        new_headlines.append(title)
            except Exception as e:
                log.warning(f"⚠️ RSS Feed error ({url}): {e}")
        return new_headlines


async def main_loop():
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    # Model load කිරීම heavy/blocking - executor thread එකකින්
    analyst = await asyncio.to_thread(CryptoMacroAnalyst)

    current_global_score = 0.0
    last_confidence = 0.0
    news_timestamps = collections.deque()  # rolling news_volume_5min tracker

    initial_state = {
        "score": 0.0, "confidence": 0.0, "news_volume_5min": 0,
        "updated_at": time.time(), "status": "warming_up"
    }
    await redis_client.set(REDIS_KEY_MACRO, json.dumps(initial_state))
    log.info("📡 [RUNNING] Live Crypto News Scanning සහ Sentiment Engine ආරම්භ විය...")

    backoff = POLL_INTERVAL_SEC

    while True:
        try:
            # [FIX] blocking RSS fetch එක executor thread එකකින් - event loop free
            headlines = await asyncio.to_thread(analyst.fetch_latest_news)

            if headlines:
                log.info(f"📰 නව පුවත් {len(headlines)} ක් හමු විය! විශ්ලේෂණය කරයි...")
                batch_scores = []
                now = time.time()

                for title in headlines:
                    try:
                        score, confidence = await asyncio.to_thread(analyst.analyze_headline, title)
                        batch_scores.append(score)
                        last_confidence = confidence
                        news_timestamps.append(now)

                        badge = "🟢 BULLISH" if score > 0.2 else ("🔴 BEARISH" if score < -0.2 else "🟡 NEUTRAL")
                        log.info(f"   {badge} ({score:+.3f}, conf={confidence:.2f}) -> {title[:60]}...")
                    except Exception as e:
                        log.error(f"⚠️ Headline analysis failed: {e}")
                        continue

                if batch_scores:
                    avg_new_score = sum(batch_scores) / len(batch_scores)
                    # EMA smoothing - ඔයාගේ original idea, sentiment "decay" එකම
                    current_global_score = (EMA_ALPHA * avg_new_score) + ((1 - EMA_ALPHA) * current_global_score)

            # [NEW] news_volume_5min - rolling window (event/risk spike detector)
            now = time.time()
            while news_timestamps and now - news_timestamps[0] > VOLUME_WINDOW_SEC:
                news_timestamps.popleft()
            volume_5min = len(news_timestamps)

            # [ENHANCEMENT] එකම JSON blob එකක් විදිහට atomic ව save කිරීම -
            # XGBoost meta-learner එකට එකම round-trip එකකින් සියල්ල ලැබෙනවා
            state = {
                "score": round(current_global_score, 4),
                "confidence": round(last_confidence, 4),
                "news_volume_5min": volume_5min,
                "updated_at": now,
                "status": "live"
            }
            await redis_client.set(REDIS_KEY_MACRO, json.dumps(state))

            log.info("-" * 55)
            log.info(f"👑 [MACRO LOCKED] Score: {current_global_score:+.4f} | "
                     f"Volume(5m): {volume_5min} | Conf: {last_confidence:.2f}")
            log.info("-" * 55)

            backoff = POLL_INTERVAL_SEC  # success -> reset
            await asyncio.sleep(POLL_INTERVAL_SEC)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"⚠️ Macro Loop Error: {e}. Retrying in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv or "--demo" in sys.argv:
        print("🎯 Crypto-RoBERTa Macro Engine පරීක්ෂා කිරීම ආරම්භ වේ...")
        tester = CryptoMacroAnalyst()
        test_headlines = [
            "Spot Bitcoin ETF sees record $1.5 Billion daily inflow from institutional investors as BTC surges",
            "SEC files urgent lawsuit to freeze assets of major crypto exchange over unregistered securities",
            "Bitcoin network difficulty adjusts by 0.2% in normal routine liquidity cycle"
        ]
        print("\n" + "=" * 65)
        print("📊 SAMPLE HEADLINE SENTIMENT ANALYSIS (FP16 Half-Precision):")
        print("=" * 65)
        for headline in test_headlines:
            score, confidence = tester.analyze_headline(headline)
            status = "🟢 BULLISH" if score > 0.2 else ("🔴 BEARISH" if score < -0.2 else "🟡 NEUTRAL")
            print(f"\n📰 Headline: \"{headline}\"")
            print(f"   🎯 Prediction: {status} | Score: {score:+.4f} | Confidence: {confidence:.2f}")
        print("\n" + "=" * 65)
        print("🏆 Crypto-RoBERTa Sentiment Engine එක සාර්ථකයි! (VRAM භාවිතය අවමයි)")
    else:
        log.info("🚀 Crypto-RoBERTa Macro Engine Live Loop ආරම්භ වේ...")
        asyncio.run(main_loop())