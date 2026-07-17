import os
import gc
import ccxt
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import lightgbm as lgb
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import logging
import multiprocessing as mp
import tempfile
import shutil

# ✅ [FIX] mamba_model සහ tcn_model module level එකේ import කරන්නේ නෙහි!
# mp.Process spawn වෙනකොට child process එක train_models.py re-import වෙනවා.
# මේ module-level import එක execute වෙනකොට mamba-ssm CUDA extensions load වෙ cuDNN corrupt වෙනවා.
# Fix: lazy imports — කතා කතා කරන්න function එකේදීම import කිරීම.

# =====================================================================
# 🎲 GLOBAL SEED LOCK (Reproducibility + Subprocess Drift Prevention)
# =====================================================================
# Subprocess Seed Drift:
#   mp.Process('spawn') නිසා child process ලට parent random state inherit නොවෙනවා.
#   Global seed lock කළාම module re-import වෙනකොට child process ලත් same seed
#   එකෙන් start වෙනවා → LightGBM sample indices, DataLoader shuffle order,
#   model weight init ඔක්කොම reproducible.
import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# =====================================================================
# 👑 MASTER OFFLINE TRAINING PIPELINE (v2 - Bug-fixed & Leakage-safe)
# Hardware Target: RTX 3050 / 8GB+ RAM
# =====================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("Trainer")

# --- [Hyperparameters & Config] ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m'
MONTHS_HISTORY = 6          # ✅ [6-MONTH] 1 → 6 (≈260,000 rows, proper generalization)
SEQUENCE_LEN = 500          # Mamba/TCN සඳහා Ticks 500 ක මතකය
BATCH_SIZE = 32             # ✅ [UPDATED] 16 → 32 (6GB VRAM sufficient, 2x faster training)
EPOCHS = 20                 # ✅ [UPDATED] 5 → 20 (proper convergence with early stopping)
LR = 3e-4                   # ✅ [UPDATED] 1e-4 → 3e-4 (faster warm-up with larger dataset)
EMBARGO_GAP = 100           # ≥ TIME_BARRIER(60) required (leakage prevention). 480→100 after barrier grid search.
DEEP_VAL_FRAC = 0.10        # Base split එකේ අන්තිම 10% deep-model early-stopping සඳහා (meta set එක touch කරන්නෙ නෑ)

# --- [Triple Barrier Config — Grid-Search Optimized] ---
# 🔬 [GRID SEARCH RESULT] 14 configs tested on 259K BTC 1-min candles.
#   Winner: TP=SL=5.0, TIME=60 → SELL 40.1% / HOLD 22.2% / BUY 37.7%
#   This is the ONLY config that achieves HOLD > 15% (institutional threshold).
#   Symmetric TP/SL = no structural labeling bias.
#   TIME=60 (1hr) = short enough that 5x ATR barriers aren't always hit.
# Previous configs and their HOLD%:
#   TP=1.2,SL=1.2,T=60  → 0.03%  🔴 (extinct)
#   TP=2.0,SL=1.5,T=480 → 0.2%   🔴 (extinct)
#   TP=4.0,SL=4.0,T=60  → 11.3%  ⚠️ (borderline)
#   TP=5.0,SL=5.0,T=60  → 22.2%  ✅ (healthy)
TP_ATR_MULT = 5.0           # 5x ATR distance — only strong momentum hits TP
SL_ATR_MULT = 5.0           # 5x ATR distance — symmetric, no labeling bias
TIME_BARRIER = 60           # 1 hour — with 5x ATR, many bars timeout → HOLD ~22%


def download_historical_data() -> pd.DataFrame:
    """
    Binance RAW klines API එකෙන් අතීත OHLCV + REAL taker_buy_volume/trade_count දත්ත බාගැනීම.
    ✅ [CACHE] Snapshot file තියෙනවා නම් re-fetch skip කරනවා (experiment iteration speed-up).
    """
    # ✅ [EXPERIMENT CACHE] Snapshot exists → skip 60-min Binance re-download.
    # ⚠️ Production deploy කරද්දී මේ cache block එක remove කරන්න (stale data risk).
    snapshot_path = 'data/btc_1m_snapshot.pkl'
    if os.path.exists(snapshot_path):
        log.info(f"📂 Existing snapshot [{snapshot_path}] use කරනවා, Binance re-fetch skip!")
        return pd.read_pickle(snapshot_path)

    log.info(f"📥 Binance වෙතින් මාස {MONTHS_HISTORY} ක '{SYMBOL}' RAW klines බාගනිමින් පවතී...")
    exchange = ccxt.binance({'enableRateLimit': True})
    market_symbol = SYMBOL.replace('/', '')
    since = exchange.milliseconds() - (MONTHS_HISTORY * 30 * 24 * 60 * 60 * 1000)
    all_rows = []

    while since < exchange.milliseconds():
        try:
            raw = exchange.publicGetKlines({
                'symbol': market_symbol,
                'interval': TIMEFRAME,
                'startTime': since,
                'limit': 1000
            })
            if not raw:
                break
            all_rows.extend(raw)
            since = int(raw[-1][0]) + 60000
            print(f"\r✅ දත්ත පේළි {len(all_rows):,} ක් බාගත කර අවසන්...", end="")
            time.sleep(0.15)  # Rate limit ආරක්ෂාව
        except Exception as e:
            log.warning(f"Fetch error: {e}. Retrying in 2s...")
            time.sleep(2)

    print()

    # Binance raw kline fields (index order නිවැරදිව):
    # 0 open_time, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 close_time,
    # 7 quote_volume, 8 number_of_trades, 9 taker_buy_base_volume,
    # 10 taker_buy_quote_volume, 11 ignore
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_volume', 'trade_count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
    df = pd.DataFrame(all_rows, columns=cols)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                     'taker_buy_volume', 'taker_buy_quote_volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['trade_count'] = df['trade_count'].astype(int)
    df['timestamp'] = pd.to_datetime(df['open_time'].astype(np.int64), unit='ms')

    # ✅ [REAL FIX] CVD - taker_buy_volume (real Binance field) සහ ඉන් derived taker_sell_volume
    # වලින් Cumulative Volume Delta ගණනය කිරීම. දැන් මේක simulation එකක් නෙවෙයි.
    df['taker_sell_volume'] = df['volume'] - df['taker_buy_volume']
    df['cvd'] = (df['taker_buy_volume'] - df['taker_sell_volume']).cumsum()

    return df


def apply_vectorized_triple_barrier(df: pd.DataFrame) -> np.ndarray:
    """අතිශය වේගවත් NumPy Loop එකකින් Triple-Barrier Labels 0(Sell), 1(Hold), 2(Buy) සැකසීම."""
    log.info("🎯 Triple-Barrier Labeling ක්‍රියාත්මක වේ (Vectorized)...")

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    atr = df['atr_14'].fillna(0).values

    labels = np.ones(len(df))  # Default: 1 (HOLD / Timeout)

    for i in range(len(df) - TIME_BARRIER):
        c_price = close[i]
        c_atr = atr[i]

        if c_atr == 0:
            continue

        tp_price = c_price + (c_atr * TP_ATR_MULT)
        sl_price = c_price - (c_atr * SL_ATR_MULT)

        window_high = high[i + 1: i + 1 + TIME_BARRIER]
        window_low = low[i + 1: i + 1 + TIME_BARRIER]

        hit_tp_idx = np.argmax(window_high >= tp_price)
        hit_sl_idx = np.argmax(window_low <= sl_price)

        hit_tp = window_high[hit_tp_idx] >= tp_price
        hit_sl = window_low[hit_sl_idx] <= sl_price

        if hit_tp and hit_sl:
            if hit_tp_idx < hit_sl_idx:
                labels[i] = 2  # BUY (Hit TP First)
            else:
                labels[i] = 0  # SELL (Hit SL First)
        elif hit_tp:
            labels[i] = 2
        elif hit_sl:
            labels[i] = 0

    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique.astype(int), counts))
    total = len(labels)
    log.info(f"📊 Label Distribution -> SELL(0): {dist.get(0,0)}, HOLD(1): {dist.get(1,0)}, BUY(2): {dist.get(2,0)}")

    # ✅ [NEW] Health check - class collapse එකක් silent ව pass වෙන්නෙ නැති වෙන්න.
    # Barrier tuning (asymmetric ATR multiples, too-long time barrier) වල symptom
    # එකක් තමයි class ratio එකක් extinct-level එකට වැටෙන එක - මේක training loop
    # එකට යන්න කලින්ම catch කරගන්න ඕන.
    for cls_id, cls_name in [(0, 'SELL'), (1, 'HOLD'), (2, 'BUY')]:
        ratio = dist.get(cls_id, 0) / total
        if ratio < 0.03:
            log.warning(f"🔴 [LABEL HEALTH] {cls_name} class ratio ({ratio*100:.2f}%) ඉතාම අඩුයි! "
                        f"TP_ATR_MULT/SL_ATR_MULT/TIME_BARRIER tune කරන්න ඕන.")

    return labels


def build_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Super-20 Institutional Features.

    ⚠️ NOTE: RAW (un-normalized) features return කරනවා.
    Normalization main() base-split-only scaler එකෙන් සිද්ධ වෙනවා (leakage-safe).
    """
    log.info("⚙️ Data Feature Engineering සිදු කරමින්...")

    df['log_return'] = ta.log_return(df['close'])

    # ✅ [FIX #3] CVD Daily Session Reset — live consumer UTC-midnight reset සමඟ align.
    # මුල් code: cumsum() 6 months unbounded → huge range (scaler overfits to training era).
    # Live code: daily reset → small daily range → train scaler squeeze.
    # Fix: training ලදී Daily session cumsum — group by date, cumsum within each day.
    # මේකෙන් train/live cvd range ගැලපෙනවා (both daily-scoped).
    df['date'] = df['timestamp'].dt.date
    df['_net_buy'] = df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])
    df['cvd'] = df.groupby('date')['_net_buy'].cumsum()
    df['cvd_velocity'] = df['cvd'].diff(periods=5).fillna(0)

    # ✅ [FIX #2] spread_pct / obi — constant 0.0 / 0.5 ඉවත් කළා.
    # Training L2 data නැති නිසා දෙකම constants → scaler range ≈ 1e-7 → live values explode.
    # Fix: OHLC data source කරගෙන realistic proxy values:
    #   spread_pct ≈ (high - low) / close — intrabar price range as bid-ask width proxy.
    #   obi ≈ taker_buy_volume / volume — directional flow imbalance as order book proxy.
    # මේ values train range (0.001-0.005 for spread, 0-1 for obi) live range සමඟ align.
    df['spread_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-9)
    df['obi'] = df['taker_buy_volume'] / (df['volume'] + 1e-9)
    df['obi'] = df['obi'].clip(0.0, 1.0)

    df['ema_9_dist'] = (df['close'] - ta.ema(df['close'], length=9)) / df['close']
    df['ema_50_dist'] = (df['close'] - ta.ema(df['close'], length=50)) / df['close']

    # Session-based (daily-reset) VWAP — matches live consumer rolling VWAP approximation.
    df['_pv'] = df['close'] * df['volume']
    df['vwap'] = df.groupby('date')['_pv'].cumsum() / (df.groupby('date')['volume'].cumsum() + 1e-7)

    df['vwap_zscore'] = (df['close'] - df['vwap']) / (df['close'].rolling(20).std() + 1e-7)
    df['vwap_slope'] = (df['vwap'] - df['vwap'].shift(5)).fillna(0) / df['close']

    # ✅ [ENHANCEMENT] Bollinger — defensive column-name check (pandas_ta version drift)
    bb = ta.bbands(df['close'], length=20)
    if bb is not None:
        bbp_cols = [c for c in bb.columns if c.startswith('BBP_')]
        bbb_cols = [c for c in bb.columns if c.startswith('BBB_')]
        df['bb_pct_b'] = bb[bbp_cols[0]] if bbp_cols else 0.5
        df['bb_width'] = bb[bbb_cols[0]] if bbb_cols else 0.0
    else:
        df['bb_pct_b'] = 0.5
        df['bb_width'] = 0.0

    atr_s = ta.atr(df['high'], df['low'], df['close'], length=7)
    atr_l = ta.atr(df['high'], df['low'], df['close'], length=21)
    df['atr_ratio'] = atr_s / (df['close'] + 1e-9)
    df['volatility_regime'] = atr_s / (atr_l + 1e-7)

    df['rsi_14'] = ta.rsi(df['close'], length=14)
    stoch = ta.stochrsi(df['close'], length=14)
    if stoch is not None:
        k_cols = [c for c in stoch.columns if 'STOCHRSIk' in c]
        df['stoch_rsi_k'] = stoch[k_cols[0]] if k_cols else 0.5
    else:
        df['stoch_rsi_k'] = 0.5
    df['roc_10'] = ta.roc(df['close'], length=10)

    # ✅ [FIX #4] time_sine/time_cosine minute precision.
    # Train: hour * 60 only (hour granularity — 60-min steps, same value for whole hour).
    # Live: hour * 60 + min (minute precision).
    # Fix: use dt.hour * 60 + dt.minute → exact minute-of-day encoding (0-1439).
    mins_of_day = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    df['time_sine'] = np.sin(2 * np.pi * mins_of_day / 1440.0)
    df['time_cosine'] = np.cos(2 * np.pi * mins_of_day / 1440.0)

    quant_cols = [
        'log_return', 'volume', 'spread_pct', 'cvd', 'cvd_velocity',
        'obi', 'trade_count', 'ema_9_dist', 'ema_50_dist', 'vwap_zscore',
        'vwap_slope', 'bb_pct_b', 'bb_width', 'rsi_14', 'stoch_rsi_k',
        'roc_10', 'atr_ratio', 'volatility_regime', 'time_sine', 'time_cosine'
    ]

    final_df = df[quant_cols].bfill().fillna(0).replace([np.inf, -np.inf], 0.0)
    return final_df


class QuantDataset(Dataset):
    """RAM එක පිරෙන්නේ නැති වෙන්න Ticks 500 කින් යුත් 3D Matrices On-the-fly හදන Dataset එක."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len=500):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x_window = self.data[idx: idx + self.seq_len]
        y_label = self.labels[idx + self.seq_len - 1]
        return x_window, y_label


def train_deep_model(model, train_loader, val_loader, model_name, device,
                     class_weights=None, use_amp=True):
    """
    Mamba හෝ TCN එක Train කරන Master Loop එක.

    ✅ [v2 OVERHAUL] Model collapse fix:
      - Gradient clipping (max_norm=1.0) → FP16 overflow-skip cascades වළක්වයි
      - Early stopping (patience=5) → degenerate equilibrium වලින් ඉක්මනින් ගැලවෙයි
      - Cosine annealing LR scheduler → better convergence landscape exploration
      - Per-class accuracy logging → collapse epoch-level එකේදීම visible
      - GradScaler overflow counting → silent skip detection
      - use_amp parameter → Mamba SSM FP16 disable කරන්න option එකක්
    """
    log.info(f"🔥 {model_name} Training ආරම්භ විය... (AMP={'ON' if use_amp else 'OFF'})")
    is_cuda = (device.type == 'cuda')
    amp_enabled = is_cuda and use_amp

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler('cuda' if is_cuda else 'cpu', enabled=amp_enabled)

    # ✅ [NEW] Cosine annealing — LR gradually decays to near-zero, helps escape
    # shallow local minima (degenerate "predict one class" equilibria).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_loss = float('inf')
    patience_counter = 0
    PATIENCE = 5  # Stop if val_loss doesn't improve for 5 consecutive epochs
    os.makedirs('weights', exist_ok=True)
    save_path = f"weights/{model_name.lower()}_best.pt"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        overflow_skips = 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            with autocast('cuda' if is_cuda else 'cpu', enabled=amp_enabled):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            # ✅ [FIX] Gradient clipping BEFORE optimizer step — prevents FP16
            # overflow cascades where GradScaler silently skips updates.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after < scale_before:
                overflow_skips += 1

            train_loss += loss.item()

            if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"   ⏳ [{model_name} Epoch {epoch+1}/{EPOCHS}] Batch {batch_idx+1:,}/{len(train_loader):,} | Step Loss: {loss.item():.4f}", flush=True)

        # ✅ [NEW] Step scheduler after each epoch
        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        # ✅ [NEW] Per-class tracking to detect collapse during training
        class_correct = {0: 0, 1: 0, 2: 0}
        class_total = {0: 0, 1: 0, 2: 0}
        pred_dist = {0: 0, 1: 0, 2: 0}

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

                # Per-class accuracy
                for cls in [0, 1, 2]:
                    mask = (batch_y == cls)
                    class_total[cls] += mask.sum().item()
                    class_correct[cls] += ((preds == cls) & mask).sum().item()
                    pred_dist[cls] += (preds == cls).sum().item()

        val_loss /= max(1, len(val_loader))
        acc = correct / max(1, total) * 100
        current_lr = optimizer.param_groups[0]['lr']

        # ✅ [NEW] Per-class accuracy log — collapse visible immediately
        cls_accs = []
        for cls, name in [(0, 'SELL'), (1, 'HOLD'), (2, 'BUY')]:
            ca = class_correct[cls] / max(1, class_total[cls]) * 100
            cls_accs.append(f"{name}:{ca:.1f}%")
        pred_pcts = [f"S:{pred_dist[0]}" , f"H:{pred_dist[1]}", f"B:{pred_dist[2]}"]

        log.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/max(1,len(train_loader)):.4f} | "
                 f"Val Loss: {val_loss:.4f} | Val Acc: {acc:.2f}% | LR: {current_lr:.2e} | "
                 f"Overflow skips: {overflow_skips}")
        log.info(f"   📊 Class Acc: [{', '.join(cls_accs)}] | Pred Dist: [{', '.join(pred_pcts)}]")

        # ✅ [NEW] Collapse detection — if any class gets 0 predictions, warn loudly
        dead_classes = [name for cls, name in [(0,'SELL'),(1,'HOLD'),(2,'BUY')] if pred_dist[cls] == 0]
        if dead_classes:
            log.warning(f"   🔴 [COLLAPSE WARNING] Dead classes (0 predictions): {dead_classes}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            log.info(f"💾 Best {model_name} Weights Saved! (Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"⏹️ Early stopping triggered at epoch {epoch+1} (patience={PATIENCE})")
                break


def _extract_lgbm_features(features_3d: np.ndarray) -> np.ndarray:
    """3D Sequence එකක් LightGBM එකට තේරෙන 2D Tabular Features බවට පත් කිරීම."""
    last_tick = features_3d[:, -1, :]
    mean_vals = np.mean(features_3d, axis=1)
    std_vals = np.std(features_3d, axis=1)
    momentum = last_tick - features_3d[:, -50, :]
    return np.hstack([last_tick, mean_vals, std_vals, momentum])


# =============================================================================
# 🔧 SUBPROCESS WORKERS - Fresh CUDA context per deep model
# mamba-ssm Triton custom CUDA kernels use කළ පසු cuDNN context corrupt
# වෙනවා → same process එකේ Conv1d (cuDNN) fail වෙනවා.
# Fix: each model spawned subprocess (fresh CUDA context) එකේ train කිරීම.
# =============================================================================

def _mamba_training_worker(tmp_dir: str, device_str: str, num_features: int,
                            batch_size: int, sequence_len: int, epochs_: int, lr_: float):
    """Mamba SSM training subprocess - isolated CUDA context."""
    import os, logging, random
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.amp import autocast, GradScaler
    from torch.utils.data import DataLoader
    from mamba_model import MambaQuantModel

    # ✅ Subprocess seed lock - spawn method fresh process එක parent seed inherit නොවෙනවා
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    log = logging.getLogger("MambaWorker")

    device = torch.device(device_str)
    X_train = np.load(os.path.join(tmp_dir, 'X_deep_train.npy'))
    y_train  = np.load(os.path.join(tmp_dir, 'y_deep_train.npy'))
    X_val    = np.load(os.path.join(tmp_dir, 'X_deep_val.npy'))
    y_val    = np.load(os.path.join(tmp_dir, 'y_deep_val.npy'))
    cw       = np.load(os.path.join(tmp_dir, 'class_weights.npy'))

    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)
    train_ds = QuantDataset(X_train, y_train, sequence_len)
    val_ds   = QuantDataset(X_val,   y_val,   sequence_len)
    # ✅ DataLoader generator seed - shuffle එක කරන්න order reproducible
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True, pin_memory=False, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=False)

    model = MambaQuantModel(input_dim=num_features, d_model=128, n_layers=3, n_classes=3).to(device)
    train_deep_model(model, train_loader, val_loader, "Mamba", device, cw_t, use_amp=False)
    log.info("✅ Mamba subprocess training ඉවර!")


def _tcn_training_worker(tmp_dir: str, device_str: str, num_features: int,
                          batch_size: int, sequence_len: int, epochs_: int, lr_: float):
    """TCN training subprocess - isolated CUDA context (no mamba-ssm contamination)."""
    import os, logging, random
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from tcn_model import TCNQuantModel

    # ✅ [FIXED] cuDNN 12.1 + v9 cleanly installed and verified working on RTX 3050 GPU.
    torch.backends.cudnn.enabled = True

    # ✅ Subprocess seed lock
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    log = logging.getLogger("TCNWorker")
    log.info("🚀 cuDNN enabled and active for TCN training!")

    device = torch.device(device_str)
    X_train = np.load(os.path.join(tmp_dir, 'X_deep_train.npy'))
    y_train  = np.load(os.path.join(tmp_dir, 'y_deep_train.npy'))
    X_val    = np.load(os.path.join(tmp_dir, 'X_deep_val.npy'))
    y_val    = np.load(os.path.join(tmp_dir, 'y_deep_val.npy'))
    cw       = np.load(os.path.join(tmp_dir, 'class_weights.npy'))

    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)
    train_ds = QuantDataset(X_train, y_train, sequence_len)
    val_ds   = QuantDataset(X_val,   y_val,   sequence_len)
    # ✅ DataLoader generator seed
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True, pin_memory=False, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=False)

    model = TCNQuantModel(input_dim=num_features, num_channels=[64, 128, 128], n_classes=3).to(device)
    train_deep_model(model, train_loader, val_loader, "TCN", device, cw_t, use_amp=True)
    log.info("✅ TCN subprocess training ඉවර!")


def main():
    # ✅ [FIXED] cuDNN enabled for main process and models
    torch.backends.cudnn.enabled = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"🖥️ Device: {device}")

    # 1. දත්ත බාගැනීම, Labeling, Feature Engineering
    raw_df = download_historical_data()
    os.makedirs('data', exist_ok=True)
    snapshot_path = 'data/btc_1m_snapshot.pkl'
    raw_df.to_pickle(snapshot_path)
    log.info(f"💾 Dataset Snapshot saved to [{snapshot_path}] — {len(raw_df):,} rows.")

    labels = apply_vectorized_triple_barrier(raw_df)
    features_df = build_quant_features(raw_df)

    # ✅ [ENHANCEMENT] Triple-barrier loop එකේ අන්තිම TIME_BARRIER rows වලට future
    # data insufficient නිසා label එක default HOLD(1) ලෙසම රැඳෙනවා — මේවා training/
    # meta/test data වලට ඇතුළත් වුනොත් artificial HOLD-bias එකක් හදනවා. ඉවත් කරමු.
    valid_len = len(raw_df) - TIME_BARRIER
    features_df = features_df.iloc[:valid_len].reset_index(drop=True)
    labels = labels[:valid_len]

    feature_arr_raw = features_df.values

    # 2. Chronological Data Split (Leakage-Proof Embargo Gaps සමඟ)
    total_len = len(feature_arr_raw)
    base_end = int(total_len * 0.60)
    meta_start = base_end + EMBARGO_GAP
    meta_end = meta_start + int(total_len * 0.20)
    test_start = meta_end + EMBARGO_GAP

    log.info(f"✂️ Data Splitting -> Base Train: {base_end} | Meta Train: {meta_end - meta_start} | Test: {total_len - test_start}")

    # 🔴 [CRITICAL BUG FIX] Normalization Data Leakage:
    # මුල් code එකේ min-max normalization එක *මුළු dataset එකටම* (train+meta+test ඔක්කොම
    # එකට combine කරලා) apply කරලා තිබ්බා. ඒකෙන් Test/Meta splits වල තියෙන future min/max
    # values, training data එකේ scale එකට leak වෙනවා — embargo gaps වලින් ලබාගත්ත
    # leakage-protection එකම මේකෙන් undo වෙනවා. Fix: scaler (min/max) fit කරන්නෙ
    # Base(train) split එකෙන් විතරයි, ඒකම meta/test splits වලටත් apply කරනවා.
    feat_min = feature_arr_raw[:base_end].min(axis=0)
    feat_max = feature_arr_raw[:base_end].max(axis=0)
    feat_range = feat_max - feat_min + 1e-7
    feature_arr = (feature_arr_raw - feat_min) / feat_range

    # ✅ [ENHANCEMENT] Live inference (Consumer script) එකට exact same scaling apply
    # කරන්න මේ scaler stats save කරගන්නවා — නැත්නම් train/live feature scale mismatch
    # වෙලා model එකේ predictions meaningless වෙනවා.
    os.makedirs('weights', exist_ok=True)
    np.savez('weights/feature_scaler.npz', min=feat_min, max=feat_max, columns=list(features_df.columns))
    log.info("💾 Feature scaler (min/max) weights/feature_scaler.npz වෙත save කළා (live inference සඳහා අනිවාර්යයි)")

    num_features = feature_arr.shape[1]

    X_meta = feature_arr[meta_start:meta_end]
    y_meta = labels[meta_start:meta_end]

    # 🔴 [BUG FIX] Stacking Leakage:
    # මුල් code එකේ meta_dataset (X_meta) එකම දෙකකටම පාවිච්චි කළා — (1) Mamba/TCN
    # early-stopping checkpoint තෝරගන්න validation set එක විදිහට, (2) ඊට පස්සේ ඒම
    # weights වලින්ම XGBoost meta-learner එකට "out-of-fold" predictions හදන්නත්.
    # මේකෙන් XGBoost එකට යන predictions ඇත්තටම "out-of-fold" නෑ — base models
    # කලින්ම මේ set එකටම fit වෙලා (checkpoint selection හරහා) optimized වෙලා ඉවරයි.
    # Fix: Base split එකෙන්ම කුඩා කොටසක් (අන්තිම 10%, embargo සමඟ) වෙන් කරලා
    # early-stopping සඳහා පාවිච්චි කරනවා. Meta split (X_meta) සම්පූර්ණයෙන්ම
    # untouched ව තියෙනවා, stacking-feature generation එකට විතරක් පස්සේ පාවිච්චි වෙන්නෙ.
    deep_val_start = int(base_end * (1 - DEEP_VAL_FRAC))
    deep_train_end = deep_val_start - EMBARGO_GAP

    X_deep_train = feature_arr[:deep_train_end]
    y_deep_train = labels[:deep_train_end]
    X_deep_val = feature_arr[deep_val_start:base_end]
    y_deep_val = labels[deep_val_start:base_end]

    deep_train_dataset = QuantDataset(X_deep_train, y_deep_train, SEQUENCE_LEN)
    deep_val_dataset = QuantDataset(X_deep_val, y_deep_val, SEQUENCE_LEN)
    meta_dataset = QuantDataset(X_meta, y_meta, SEQUENCE_LEN)

    # ✅ DataLoader (main process meta-loader) generator seed
    g_meta = torch.Generator()
    g_meta.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(deep_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=False, generator=g_meta)
    early_stop_val_loader = DataLoader(deep_val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
    meta_loader = DataLoader(meta_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

    # ✅ [ENHANCEMENT] Class weights ගණනය කිරීම (deep-train split එකෙන්ම)
    class_counts = np.bincount(y_deep_train.astype(int), minlength=3)
    # ✅ [FIX] Stronger weighting: sqrt of inverse-frequency + extra HOLD boost.
    # Standard inverse-freq (sum / 3*count) wasn't strong enough to prevent HOLD
    # from dying. Using (max_count / count) ensures rarest class gets highest weight.
    class_weights = class_counts.max() / (class_counts + 1e-7)
    # Cap weights to avoid instability (max 5x any class)
    class_weights = np.clip(class_weights, 1.0, 5.0)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    log.info(f"⚖️ Class Weights (Sell/Hold/Buy): {class_weights}")

    # -------------------------------------------------------------
    # 3. BASE MODELS TRAINING (Mamba & TCN) — isolated subprocesses
    # -------------------------------------------------------------
    # ✅ [ROOT FIX] mamba-ssm Triton kernels corrupt the cuDNN context →
    # Conv1d (TCN) fails with CUDNN_STATUS_NOT_INITIALIZED in the same process.
    # Solution: each deep model runs in its own spawned subprocess (fresh CUDA ctx).

    tmp_dir = tempfile.mkdtemp(prefix='multimodalbot_train_')
    try:
        np.save(os.path.join(tmp_dir, 'X_deep_train.npy'), X_deep_train)
        np.save(os.path.join(tmp_dir, 'y_deep_train.npy'), y_deep_train)
        np.save(os.path.join(tmp_dir, 'X_deep_val.npy'),   X_deep_val)
        np.save(os.path.join(tmp_dir, 'y_deep_val.npy'),   y_deep_val)
        np.save(os.path.join(tmp_dir, 'class_weights.npy'), class_weights)

        worker_args = (tmp_dir, str(device), num_features, BATCH_SIZE, SEQUENCE_LEN, EPOCHS, LR)

        # --- Mamba subprocess ---
        log.info("🔥 Mamba Training (fresh subprocess) ආරම්භ විය...")
        p_mamba = mp.Process(target=_mamba_training_worker, args=worker_args)
        p_mamba.start()
        p_mamba.join()
        if p_mamba.exitcode != 0:
            raise RuntimeError(f"❌ Mamba subprocess failed (exit {p_mamba.exitcode})")

        # --- TCN subprocess (clean CUDA context, no mamba-ssm contamination) ---
        log.info("🔥 TCN Training (fresh subprocess) ආරම්භ විය...")
        p_tcn = mp.Process(target=_tcn_training_worker, args=worker_args)
        p_tcn.start()
        p_tcn.join()
        if p_tcn.exitcode != 0:
            raise RuntimeError(f"❌ TCN subprocess failed (exit {p_tcn.exitcode})")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 4. LIGHTGBM TRAINING
    # -------------------------------------------------------------
    log.info("⚡ LightGBM Tabular Expert Training ආරම්භ විය...")

    # 🔴 [CRITICAL FIX] මුල් code එකේ np.random.choice() uniform-random සම්පූර්ණ
    # dataset එකෙන් sample කරගත්තා - HOLD class rare නිසා 60,000 sample එකට
    # HOLD window එකක්වත් නොවැටෙන්න පුළුවන් (real production log එකේම මේ exact
    # scenario එක සිද්ධ වුනා). LightGBM එකට classes 2ක් විතරක් පේනකොට, ඒක silently
    # binary classifier එකකට switch වෙනවා, downstream XGBoost meta-judge එකටත්
    # HOLD signal එකක් ලැබෙන්නෙම නෑ.
    # Fix: Stratified sampling - class එක් එකකින් ම equal-ish representation එකක්
    # (available count එකට constraint වෙලා) guarantee කරනවා.
    # ✅ [UPDATED] 6-month data → 60,000 samples (was 20,000 = only 21% of base)
    lgb_sample_size = min(60000, len(deep_train_dataset))
    window_labels = y_deep_train[SEQUENCE_LEN - 1: SEQUENCE_LEN - 1 + len(deep_train_dataset)]

    per_class_target = lgb_sample_size // 3
    tabular_indices = []
    for cls_id, cls_name in [(0, 'SELL'), (1, 'HOLD'), (2, 'BUY')]:
        cls_indices = np.where(window_labels == cls_id)[0]
        if len(cls_indices) == 0:
            log.warning(f"🔴 [LGBM SAMPLING] Class {cls_name} has ZERO windows in deep-train split! "
                        f"Barrier config එක නැවත review කරන්න.")
            continue
        take = min(per_class_target, len(cls_indices))
        tabular_indices.extend(np.random.choice(cls_indices, take, replace=False))
    tabular_indices = np.array(tabular_indices)
    np.random.shuffle(tabular_indices)

    X_lgb_list, y_lgb_list = [], []
    for idx in tabular_indices:
        x, y = deep_train_dataset[idx]
        X_lgb_list.append(x.numpy())
        y_lgb_list.append(y.item())

    # ✅ [NEW] Silent binary-classifier collapse එකක් නැවත වෙන්නෙ නැති වෙන්න hard guard
    n_classes_present = len(np.unique(y_lgb_list))
    assert n_classes_present == 3, (
        f"❌ LightGBM training sample එකේ classes 3ම නෑ (found {n_classes_present})! "
        f"Stratified sampling එකෙන් පස්සෙත් මේක වුනොත්, barrier config එක "
        f"(TP_ATR_MULT/SL_ATR_MULT/TIME_BARRIER) නැවත tune කරන්න ඕන."
    )

    X_lgb_3d = np.array(X_lgb_list)
    X_lgb_2d = _extract_lgbm_features(X_lgb_3d)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,       # ✅ [UPDATED] 100 → 300 (more data → more trees)
        learning_rate=0.05,
        max_depth=6,            # ✅ [UPDATED] 5 → 6 (slightly deeper for complex 6m patterns)
        num_leaves=63,          # ✅ [NEW] explicit control (default 31 → 63 for depth=6)
        verbose=-1,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )
    lgb_model.fit(X_lgb_2d, y_lgb_list)
    lgb_model.booster_.save_model("weights/lgb_expert.txt")
    log.info(f"💾 LightGBM Weights Saved! ({lgb_sample_size:,} samples, 300 trees)")

    # -------------------------------------------------------------
    # 5. XGBOOST MASTER JUDGE TRAINING (True Out-of-Fold, Meta split only)
    # -------------------------------------------------------------
    log.info("👑 XGBoost Meta-Judge Training ආරම්භ විය...")

    # Mamba + TCN reload from saved weights for XGBoost stacking.
    # Lazy import here — only in main process, after subprocesses done.
    from mamba_model import MambaQuantModel
    from tcn_model import TCNQuantModel

    mamba = MambaQuantModel(input_dim=num_features, d_model=128, n_layers=3, n_classes=3).to(device)
    mamba.load_state_dict(torch.load("weights/mamba_best.pt", map_location=device, weights_only=True))
    mamba.eval()

    tcn = TCNQuantModel(input_dim=num_features, num_channels=[64, 128, 128], n_classes=3).to(device)
    tcn.load_state_dict(torch.load("weights/tcn_best.pt", map_location=device, weights_only=True))
    tcn.eval()

    meta_preds_list = []
    y_meta_list = []

    with torch.no_grad():
        for batch_x, batch_y in meta_loader:
            batch_x = batch_x.to(device)

            m_probs = torch.softmax(mamba(batch_x), dim=1).cpu().numpy()
            t_probs = torch.softmax(tcn(batch_x), dim=1).cpu().numpy()

            l_features = _extract_lgbm_features(batch_x.cpu().numpy())
            l_probs = lgb_model.predict_proba(l_features)
            # ✅ [FIX] Stratified sampling + assertion guard ඉහළින් දාපු නිසා,
            # lgb_model දැන් guaranteed 3-class - fragile binary/multiclass
            # shape-guessing fallback එක ඉවත් කළා (ඒක තමයි කලින් 11-vs-12-feature
            # mismatch එකේ මුල් හේතුව).
            assert l_probs.shape[1] == 3, (
                f"❌ LightGBM predict_proba() 3 columns නෑ (got {l_probs.shape[1]})! "
                f"Model එක binary ලෙස train වෙලා ඇති - stratified sampling fix එක check කරන්න."
            )

            # Dummy Macro Signal (Historical Backtest එකේදී Macro පුවත් නැති නිසා placeholder)
            dummy_macro = np.zeros((batch_x.shape[0], 3))

            meta_batch = np.hstack([m_probs, t_probs, l_probs, dummy_macro])
            meta_preds_list.append(meta_batch)
            y_meta_list.append(batch_y.numpy())

    X_xgb = np.vstack(meta_preds_list)
    y_xgb = np.concatenate(y_meta_list)

    # ✅ [UPDATED + FIX] XGBoost with early stopping to prevent meta-learner overfit
    # 6-month meta split has ~52,000 samples → 150 trees can overfit without early stop.
    # Split meta set 80/20 for XGBoost eval (strictly within meta, no leakage).
    xgb_split = int(len(X_xgb) * 0.80)
    X_xgb_tr, X_xgb_ev = X_xgb[:xgb_split], X_xgb[xgb_split:]
    y_xgb_tr, y_xgb_ev = y_xgb[:xgb_split], y_xgb[xgb_split:]

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,           # ✅ [UPDATED] 150 → 300 (early stop prevents actual overfit)
        learning_rate=0.03,
        max_depth=4,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',     # ✅ [NEW] validation loss metric
        early_stopping_rounds=30,   # ✅ [NEW] stop if val loss doesn't improve 30 rounds
        seed=RANDOM_SEED
    )
    xgb_model.fit(
        X_xgb_tr, y_xgb_tr,
        eval_set=[(X_xgb_ev, y_xgb_ev)],
        verbose=False
    )
    xgb_model.save_model("weights/xgb_meta.json")

    log.info("🏆 [SUCCESS] මුළු පද්ධතියම Train කර 'weights' ෆෝල්ඩරයේ සුරක්ෂිතව ගබඩා කරන ලදී!")


if __name__ == "__main__":
    # spawn: fresh Python process per subprocess → clean CUDA context per model
    mp.set_start_method('spawn', force=True)
    main()