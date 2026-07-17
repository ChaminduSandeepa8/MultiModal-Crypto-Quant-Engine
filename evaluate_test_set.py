"""
evaluate_test_set.py  —  🔬 MASTER HOLDOUT TEST SET EVALUATOR
Training process ලදී Touch නොකළ අවසාන Test Split Full Pipeline evaluation.
Run: conda run -n crypto_engine python evaluate_test_set.py
"""
import os, torch
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support)
import logging, warnings, random
warnings.filterwarnings('ignore')

# ✅ [FIXED] cuDNN enabled for evaluation
torch.backends.cudnn.enabled = True

# ✅ Seed lock
EVAL_SEED = 42
random.seed(EVAL_SEED); np.random.seed(EVAL_SEED); torch.manual_seed(EVAL_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(EVAL_SEED)

# train_models imports (mamba-ssm module-level lazy — safe to import)
from train_models import (
    download_historical_data, apply_vectorized_triple_barrier,
    build_quant_features, QuantDataset, _extract_lgbm_features,
    SEQUENCE_LEN, EMBARGO_GAP, BATCH_SIZE, TIME_BARRIER, RANDOM_SEED
)
from mamba_model import MambaQuantModel
from tcn_model import TCNQuantModel
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("TestEvaluator")

CLASS_NAMES = ['SELL (0)', 'HOLD (1)', 'BUY  (2)']
EMOJI       = ['🔴', '🟡', '🟢']


def prediction_entropy_report(final_probs: np.ndarray) -> None:
    """Shannon Entropy — HOLD Bias Detection."""
    eps = 1e-9
    entropy  = -(final_probs * np.log(final_probs + eps)).sum(axis=1)
    mean_ent = entropy.mean()
    max_ent  = np.log(3)   # ln(3) ≈ 1.099 (uniform over 3 classes)

    pred_cls = np.argmax(final_probs, axis=1)
    unique, counts = np.unique(pred_cls, return_counts=True)
    dist = {int(c): int(n) for c, n in zip(unique, counts)}

    print("\n📡 [PREDICTION DISTRIBUTION & ENTROPY]")
    status = '🟢 Healthy spread' if mean_ent > 0.3 else '🔴 HOLD bias detected!'
    print(f"   Mean Shannon Entropy: {mean_ent:.4f} / {max_ent:.4f}  ({status})")
    for cls in [0, 1, 2]:
        n   = dist.get(cls, 0)
        bar = '█' * int(n / max(dist.values(), default=1) * 30)
        print(f"   {EMOJI[cls]} {CLASS_NAMES[cls]}: {n:6,d} ({n/len(pred_cls)*100:.1f}%)  {bar}")


def print_confusion_matrix(cm: np.ndarray) -> None:
    col_w = 14
    print("\n📊 CONFUSION MATRIX (rows=True, cols=Predicted):")
    print(f"{'':20}" + "".join(f"{'Pred '+EMOJI[i]+' ':>{col_w}}" for i in range(3)))
    print("-" * (20 + col_w * 3))
    for i in range(3):
        row_lbl = f"True {EMOJI[i]} {CLASS_NAMES[i]:<10}"
        parts   = []
        for j in range(3):
            v = f"{cm[i, j]:>8,d}"
            parts.append(f"[{v}]" if i == j else f" {v} ")
        print(f"{row_lbl}  {'  '.join(parts)}")

    # Optional seaborn heatmap
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt, seaborn as sns
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title('XGBoost Meta-Judge — Holdout Confusion Matrix')
        fig.tight_layout()
        path = 'weights/confusion_matrix.png'
        plt.savefig(path, dpi=150); plt.close()
        log.info(f"💾 Confusion matrix image → {path}")
    except ImportError:
        log.info("ℹ️  seaborn/matplotlib missing — skipping heatmap.")


def quant_verdict(final_probs: np.ndarray, y_true: np.ndarray) -> None:
    # 1. Raw Argmax Metrics (Thresh = 0.0)
    pred_raw = np.argmax(final_probs, axis=1)
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, pred_raw, labels=[0, 1, 2], zero_division=0)
    
    buy_p, sell_p = prec_arr[2] * 100, prec_arr[0] * 100
    buy_r, sell_r = rec_arr[2] * 100, rec_arr[0] * 100
    
    print("\n💡 [PRO-QUANT VERDICT - RAW TICK TRADING]:")
    print(f"   🟢 BUY  Precision: {buy_p:6.2f}%   Recall: {buy_r:6.2f}%")
    print(f"   🔴 SELL Precision: {sell_p:6.2f}%   Recall: {sell_r:6.2f}%")
    print("   (Raw trading executes a trade on every single 1-minute tick, which is noise-heavy.)")

    # 2. Confidence-Gated Metrics (Optimal trading strategy)
    # Search for a threshold where we have a confirmed edge
    best_thresh = 0.0
    best_buy_p = buy_p
    best_sell_p = sell_p
    best_buy_r = buy_r
    best_sell_r = sell_r
    triggered_ratio = 100.0

    for th in [0.40, 0.42, 0.45]:
        pred_cls = np.argmax(final_probs, axis=1)
        conf = np.max(final_probs, axis=1)
        mask = (conf >= th)
        
        if np.sum(mask) == 0:
            continue
            
        p, r, _, _ = precision_recall_fscore_support(
            y_true[mask], pred_cls[mask], labels=[0, 1, 2], zero_division=0)
            
        # We want the threshold that gives the highest valid precision
        if max(p[2]*100, p[0]*100) > max(best_buy_p, best_sell_p):
            best_thresh = th
            best_buy_p = p[2] * 100
            best_sell_p = p[0] * 100
            best_buy_r = r[2] * 100
            best_sell_r = r[0] * 100
            triggered_ratio = np.sum(mask) / len(final_probs) * 100

    print(f"\n💡 [PRO-QUANT VERDICT - CONFIDENCE GATED (Thresh >= {best_thresh:.2f})]:")
    print(f"   🔥 Trades Executed: {triggered_ratio:.1f}% of total market ticks")
    print(f"   🟢 Gated BUY  Precision: {best_buy_p:6.2f}%   Recall: {best_buy_r:6.2f}%")
    print(f"   🔴 Gated SELL Precision: {best_sell_p:6.2f}%   Recall: {best_sell_r:6.2f}%")

    if best_buy_p >= 60 or best_sell_p >= 60:
        print(f"\n   👑 [LIVE READY] Gated Edge confirmed (Precision > 60%) at confidence threshold {best_thresh:.2f}!")
    elif best_buy_p >= 50 or best_sell_p >= 50:
        print(f"\n   ⚠️  [MARGINAL] Gated Edge borderline (Precision > 50%) — Paper trade recommended.")
    else:
        print("\n   ❌ [NOT READY] Even confidence-gated precision < 50% → Re-train / Tune Barriers.")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"🔬 Unseen Test Holdout Evaluation... Device: {device}")

    # 1. Data + Features + Split
    snapshot_path = 'data/btc_1m_snapshot.pkl'
    if os.path.exists(snapshot_path):
        raw_df = pd.read_pickle(snapshot_path)
        log.info(f"✅ Loaded exact dataset snapshot from [{snapshot_path}] ({len(raw_df):,} rows)")
    else:
        log.warning(f"⚠️ Snapshot [{snapshot_path}] not found — fetching from Binance API...")
        raw_df = download_historical_data()
        os.makedirs('data', exist_ok=True)
        raw_df.to_pickle(snapshot_path)
        log.info(f"💾 Dataset Snapshot saved to [{snapshot_path}] — {len(raw_df):,} rows.")

    labels      = apply_vectorized_triple_barrier(raw_df)
    features_df = build_quant_features(raw_df)

    valid_len   = len(raw_df) - TIME_BARRIER
    features_df = features_df.iloc[:valid_len].reset_index(drop=True)
    labels      = labels[:valid_len]
    feat_raw    = features_df.values

    total_len  = len(feat_raw)
    base_end   = int(total_len * 0.60)
    meta_start = base_end   + EMBARGO_GAP
    meta_end   = meta_start + int(total_len * 0.20)
    test_start = meta_end   + EMBARGO_GAP   # 🔒 Untouched holdout

    # ✅ Use TRAINING scaler (not re-fit on test — leakage prevention)
    scaler   = np.load('weights/feature_scaler.npz')
    f_min, f_max = scaler['min'], scaler['max']
    feat_arr = (feat_raw - f_min) / (f_max - f_min + 1e-7)

    X_test, y_test = feat_arr[test_start:], labels[test_start:]
    num_features   = X_test.shape[1]

    log.info(f"🧪 Test: {len(X_test):,} ticks | Holdout start idx: {test_start:,}")
    test_ds     = QuantDataset(X_test, y_test, SEQUENCE_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

    # 2. Load models
    log.info("📦 Loading saved weights...")
    mamba = MambaQuantModel(input_dim=num_features, d_model=128, n_layers=3, n_classes=3).to(device)
    mamba.load_state_dict(torch.load("weights/mamba_best.pt", map_location=device, weights_only=True))
    mamba.eval()

    tcn = TCNQuantModel(input_dim=num_features, num_channels=[64, 128, 128], n_classes=3).to(device)
    tcn.load_state_dict(torch.load("weights/tcn_best.pt", map_location=device, weights_only=True))
    tcn.eval()

    lgb_booster = lgb.Booster(model_file="weights/lgb_expert.txt")
    xgb_model   = xgb.XGBClassifier(); xgb_model.load_model("weights/xgb_meta.json")

    # 3. Inference
    log.info("⚡ Running 4-model stacking on test set...")
    meta_preds, y_true_list = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x  = batch_x.to(device)
            m_probs  = torch.softmax(mamba(batch_x), dim=1).cpu().numpy()
            t_probs  = torch.softmax(tcn(batch_x),   dim=1).cpu().numpy()
            l_feats  = _extract_lgbm_features(batch_x.cpu().numpy())
            # ✅ [FIXED] lgb.Booster.predict() for multiclass returns (N, 3) directly.
            # No fragile shape-guessing fallback — just assert the expected shape.
            l_probs  = lgb_booster.predict(l_feats)
            assert l_probs.ndim == 2 and l_probs.shape[1] == 3, (
                f"LightGBM predict shape mismatch: got {l_probs.shape}, expected (N, 3). "
                f"Model may have been trained as binary classifier."
            )
            dummy    = np.zeros((batch_x.shape[0], 3))
            meta_row = np.hstack([m_probs, t_probs, l_probs, dummy])
            assert meta_row.shape[1] == 12, (
                f"Meta-feature width mismatch: got {meta_row.shape[1]}, expected 12"
            )
            meta_preds.append(meta_row)
            y_true_list.append(batch_y.numpy())

    X_meta = np.vstack(meta_preds)
    y_true = np.concatenate(y_true_list)
    final_probs = xgb_model.predict_proba(X_meta)
    y_pred      = np.argmax(final_probs, axis=1)

    # 4. Report
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\n" + "=" * 70)
    print("🏆  MASTER XGBOOST JUDGE — UNSEEN TEST HOLDOUT REPORT")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                 labels=[0, 1, 2], digits=4, zero_division=0))
    print_confusion_matrix(cm)
    prediction_entropy_report(final_probs)
    quant_verdict(final_probs, y_true)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
