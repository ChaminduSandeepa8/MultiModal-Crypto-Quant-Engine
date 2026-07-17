import asyncio
import redis.asyncio as redis
import torch
import numpy as np
import json
import logging
import time
import os

import lightgbm as lgb
import xgboost as xgb

from mamba_model import MambaQuantModel
from tcn_model import TCNQuantModel

# =====================================================================
# 👑 MASTER JUDGE - Institutional Stacking Execution Hub (v3 - Fixed)
# Hardware target: RTX 3050 (Ubuntu, CUDA 12.1, PyTorch 2.4.0)
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("MasterJudge")

# --- [Config] ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_KEY_MACRO = 'macro_sentiment'          # [FIX] macro_sentiment_engine.py v3 එකේම key එක
MACRO_STALE_SEC = 300                        # මේ වේලාවට වඩා පරණ macro data නම් - ignore කරන්න
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Veto thresholds - symmetric (bearish blocks buy, bullish blocks sell)
VETO_BEARISH_THRESHOLD = -0.40
VETO_BULLISH_THRESHOLD = 0.40
VETO_MIN_CONFIDENCE = 0.55                    # මේකට අඩු confidence එකක් තියෙන sentiment veto trigger කරන්නෙ නෑ

# Meta-feature column layout - [FIX] documented explicitly, column-order bugs වළක්වයි
META_FEATURE_NAMES = [
    "mamba_sell", "mamba_hold", "mamba_buy",
    "tcn_sell", "tcn_hold", "tcn_buy",
    "lgb_sell", "lgb_hold", "lgb_buy",
    "macro_score", "macro_confidence", "macro_volume_norm",
]
N_META_FEATURES = len(META_FEATURE_NAMES)     # 12

ACTIONS = ["🔴 SELL (Short Trade Order)", "🟡 HOLD (No Trade / Safe Risk)", "🟢 BUY (Long Trade Order)"]


class InstitutionalMasterJudge:
    def __init__(self, device=DEVICE,
                 mamba_weights_path=None, tcn_weights_path=None,
                 lgb_model_path=None, xgb_model_path=None):
        self.device = torch.device(device)
        log.info(f"👑 [INIT] Master Stacking Engine ආරම්භ වේ... Device: {self.device}")

        # 1. Base Deep Learning Models
        self.mamba_net = MambaQuantModel(input_dim=20, d_model=128, n_layers=3, n_classes=3).to(self.device)
        self.tcn_net = TCNQuantModel(input_dim=20, num_channels=[64, 128, 128], n_classes=3).to(self.device)

        self.lgb_expert = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, verbose=-1)
        self.meta_judge = xgb.XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=4,
                                             objective='multi:softprob', num_class=3)

        # [FIX #1] Explicit trained-weight loading with fail-safe production guard.
        # Weights load වුනේ නැත්නම් `self.production_ready = False` -> execute_trade_decision()
        # live trades reject කරනවා, silent random-prediction risk එක වළක්වනවා.
        self.production_ready = True
        self._load_or_warn(mamba_weights_path, tcn_weights_path, lgb_model_path, xgb_model_path)

        if not self.production_ready:
            log.warning("🛑 " + "=" * 56)
            log.warning("🛑 [WARM-UP / TESTING MODE] Trained weights load වුනේ නෑ!")
            log.warning("🛑 මොඩල predictions මේ mode එකේදී RANDOM/MEANINGLESS.")
            log.warning("🛑 Live trading වලට කලින් weight paths සපයන්න.")
            log.warning("🛑 " + "=" * 56)

        log.info(f"{'✅ [PRODUCTION READY]' if self.production_ready else '⚠️ [WARM-UP ONLY]'} Master Judge initialized.")

    def _load_or_warn(self, mamba_path, tcn_path, lgb_path, xgb_path):
        """Trained weights තියෙනවා නම් load කරනවා. නැත්නම් dummy warm-up එකකින්
        structural crash වළක්වගෙන, production_ready=False ලෙස flag කරනවා."""

        if mamba_path and os.path.exists(mamba_path):
            self.mamba_net.load_state_dict(torch.load(mamba_path, map_location=self.device, weights_only=True))
            log.info(f"✅ Mamba weights loaded: {mamba_path}")
        else:
            log.warning(f"⚠️ Mamba weights not found ({mamba_path}) - using random init.")
            self.production_ready = False

        if tcn_path and os.path.exists(tcn_path):
            self.tcn_net.load_state_dict(torch.load(tcn_path, map_location=self.device, weights_only=True))
            log.info(f"✅ TCN weights loaded: {tcn_path}")
        else:
            log.warning(f"⚠️ TCN weights not found ({tcn_path}) - using random init.")
            self.production_ready = False

        if lgb_path and os.path.exists(lgb_path):
            self.lgb_expert = lgb.Booster(model_file=lgb_path)
            log.info(f"✅ LightGBM model loaded: {lgb_path}")
        else:
            log.warning(f"⚠️ LightGBM model not found ({lgb_path}) - using dummy warm-up fit.")
            self._warmup_lgb()
            self.production_ready = False

        if xgb_path and os.path.exists(xgb_path):
            self.meta_judge.load_model(xgb_path)
            log.info(f"✅ XGBoost meta-judge loaded: {xgb_path}")
        else:
            log.warning(f"⚠️ XGBoost meta-judge not found ({xgb_path}) - using dummy warm-up fit.")
            self._warmup_xgb()
            self.production_ready = False

    def _warmup_lgb(self):
        rng = np.random.default_rng(42)
        X_dummy = rng.standard_normal((200, 80))
        y_dummy = rng.integers(0, 3, size=200)
        self.lgb_expert.fit(X_dummy, y_dummy)

    def _warmup_xgb(self):
        rng = np.random.default_rng(42)
        X_meta_dummy = rng.random((200, N_META_FEATURES))
        y_dummy = rng.integers(0, 3, size=200)
        self.meta_judge.fit(X_meta_dummy, y_dummy)

    def _extract_tabular_features(self, tensor_3d: torch.Tensor) -> np.ndarray:
        """[1, 500, 20] GPU Tensor එකක් LightGBM එකට තේරෙන 80-feature 2D Vector එකක් කිරීම."""
        data_np = tensor_3d.cpu().numpy()
        seq_len = data_np.shape[1]
        lookback = min(50, seq_len)  # [FIX] match exact lookback range

        last_tick = data_np[:, -1, :]
        mean_vals = np.mean(data_np, axis=1)
        std_vals = np.std(data_np, axis=1)
        # ✅ [FIX] index mismatch (-1 - lookback = -51 vs train -50). Using -lookback (= -50) matches train_models.py exactly!
        momentum = last_tick - data_np[:, -lookback, :]

        return np.hstack([last_tick, mean_vals, std_vals, momentum])


    async def _read_macro_state(self, redis_client: redis.Redis) -> dict:
        """[FIX #3] macro_sentiment_engine.py v3 එකේම JSON blob schema එකට align කළා.
        Stale/missing/malformed data නම් - neutral (safe) fallback."""
        neutral = {"score": 0.0, "confidence": 0.0, "news_volume_5min": 0, "stale": True}
        try:
            raw = await redis_client.get(REDIS_KEY_MACRO)
            if not raw:
                log.warning("⚠️ Macro sentiment key empty - engine started ද කියලා check කරන්න.")
                return neutral

            state = json.loads(raw)
            age_sec = time.time() - state.get("updated_at", 0)
            is_stale = age_sec > MACRO_STALE_SEC

            if is_stale:
                log.warning(f"⚠️ Macro sentiment STALE ({age_sec:.0f}s old) - neutral ලෙස treat කරයි.")
                return neutral

            return {
                "score": float(state.get("score", 0.0)),
                "confidence": float(state.get("confidence", 0.0)),
                "news_volume_5min": int(state.get("news_volume_5min", 0)),
                "stale": False,
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log.error(f"⚠️ Macro sentiment parse error: {e}. Neutral fallback.")
            return neutral
        except redis.RedisError as e:
            log.error(f"⚠️ Redis read error: {e}. Neutral fallback.")
            return neutral

    async def execute_trade_decision(self, live_tensor_3d: torch.Tensor, redis_client: redis.Redis):
        """
        Live [1, 500, 20] Tensor එක + Redis macro sentiment blob එක ගලපා
        අවසාන trade තීන්දුව ගැනීම. ඕනම component fail උනත් fail-safe HOLD.
        """
        if not self.production_ready:
            log.warning("🛑 [BLOCKED] Model weights untrained - trade decision execute කරන්නෙ නෑ.")
            return 1, 0.0  # Force HOLD, zero confidence

        try:
            live_tensor_3d = live_tensor_3d.to(self.device)
            self.mamba_net.eval()
            self.tcn_net.eval()

            with torch.no_grad():
                mamba_probs = torch.softmax(self.mamba_net(live_tensor_3d), dim=1).cpu().numpy()
                tcn_probs = torch.softmax(self.tcn_net(live_tensor_3d), dim=1).cpu().numpy()

            tab_features = self._extract_tabular_features(live_tensor_3d)
            # ✅ [CRITICAL BUG FIX] lgb.Booster object එකට .predict_proba() නැත.
            # Booster එකෙහි .predict() මඟින් multiclass සඳහා කෙළින්ම [N, num_classes] probability matrix එක return වේ.
            if hasattr(self.lgb_expert, "predict_proba"):
                lgb_probs = self.lgb_expert.predict_proba(tab_features)
            else:
                l_raw = self.lgb_expert.predict(tab_features)
                if l_raw.ndim == 1 and l_raw.size == 3:
                    lgb_probs = l_raw.reshape(1, -1)
                elif l_raw.ndim == 1 or (l_raw.ndim == 2 and l_raw.shape[1] == 1):
                    flat_p = l_raw.ravel()
                    if hasattr(self.meta_judge, "n_features_in_") and self.meta_judge.n_features_in_ == 11:
                        lgb_probs = np.zeros((tab_features.shape[0], 2))
                        lgb_probs[:, 0] = 1.0 - flat_p
                        lgb_probs[:, 1] = flat_p
                    else:
                        lgb_probs = np.zeros((tab_features.shape[0], 3))
                        lgb_probs[:, 0] = 1.0 - flat_p
                        lgb_probs[:, 2] = flat_p
                else:
                    lgb_probs = l_raw

            macro = await self._read_macro_state(redis_client)
            # [ENHANCEMENT] macro block එකේ score + confidence + volume (log-normalized, capped 0-1)
            volume_norm = min(macro["news_volume_5min"] / 10.0, 1.0)
            macro_vector = np.array([[macro["score"], macro["confidence"], volume_norm]])

            meta_input = np.hstack([mamba_probs, tcn_probs, lgb_probs, macro_vector])
            assert meta_input.shape[1] == N_META_FEATURES, \
                f"Meta-feature shape mismatch: got {meta_input.shape[1]}, expected {N_META_FEATURES}"

            final_probs = self.meta_judge.predict_proba(meta_input)[0]
            decision_class = int(np.argmax(final_probs))
            confidence = float(final_probs[decision_class] * 100)

        except Exception as e:
            log.error(f"⚠️ [PIPELINE ERROR] {e} - fail-safe HOLD.")
            return 1, 0.0

        # --- [Institutional Veto Logic - Symmetric + Confidence-Gated] ---
        veto_triggered = False
        macro_score, macro_conf = macro["score"], macro["confidence"]

        if not macro["stale"] and macro_conf >= VETO_MIN_CONFIDENCE:
            if decision_class == 2 and macro_score < VETO_BEARISH_THRESHOLD:
                log.warning("🛡️ [VETO] BUY signal REJECTED - severe bearish news (high confidence).")
                decision_class = 1
                veto_triggered = True
            elif decision_class == 0 and macro_score > VETO_BULLISH_THRESHOLD:
                log.warning("🛡️ [VETO] SELL signal REJECTED - severe bullish news (high confidence).")
                decision_class = 1
                veto_triggered = True

        # --- [Report] ---
        print("\n" + "⚡" * 60)
        log.info("📊 විශේෂඥ මඩුල්ලේ වාර්තාව (Base Experts Output):")
        print(f"   🧠 Mamba:      [Sell:{mamba_probs[0,0]*100:5.1f}% Hold:{mamba_probs[0,1]*100:5.1f}% Buy:{mamba_probs[0,2]*100:5.1f}%]")
        print(f"   🎯 TCN:        [Sell:{tcn_probs[0,0]*100:5.1f}% Hold:{tcn_probs[0,1]*100:5.1f}% Buy:{tcn_probs[0,2]*100:5.1f}%]")
        print(f"   ⚡ LightGBM:   [Sell:{lgb_probs[0,0]*100:5.1f}% Hold:{lgb_probs[0,1]*100:5.1f}% Buy:{lgb_probs[0,2]*100:5.1f}%]")
        print(f"   📰 Macro:      score={macro_score:+.3f} conf={macro_conf:.2f} vol5m={macro['news_volume_5min']} "
              f"{'(STALE)' if macro['stale'] else ''}")
        print("-" * 60)
        log.info("👑 MASTER XGBOOST JUDGE ගේ අවසාන තීරණය:")
        print(f"   🚀 ACTION:     {ACTIONS[decision_class]} {'🛡️ (VETO OVERRIDE)' if veto_triggered else ''}")
        print(f"   🔥 CONFIDENCE: {confidence:.2f}%")
        print("⚡" * 60 + "\n")

        return decision_class, confidence


# --- [මහා පරීක්ෂණය (Master Stacking Simulation)] ---
async def main():
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    # [NOTE] Path මෙතන දෙන්න - නැත්නම් warm-up/testing mode එකේ run වෙනවා
    judge = InstitutionalMasterJudge(
        mamba_weights_path="weights/mamba_best.pt",
        tcn_weights_path="weights/tcn_best.pt",
        lgb_model_path="weights/lgb_expert.txt",
        xgb_model_path="weights/xgb_meta.json",
    )

    # Test: macro_sentiment_engine.py v3 එකේම JSON schema එකෙන් sample bearish state එකක්
    test_state = {
        "score": -0.65, "confidence": 0.72, "news_volume_5min": 4,
        "updated_at": time.time(), "status": "live"
    }
    await redis_client.set(REDIS_KEY_MACRO, json.dumps(test_state))

    sample_live_tensor = torch.randn(1, 500, 20, dtype=torch.float32).to(judge.device)

    log.info("📡 Live Simulation Tensor එකක් Master Judge වෙත යවයි...")
    await judge.execute_trade_decision(sample_live_tensor, redis_client)

    await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())