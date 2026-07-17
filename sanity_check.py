"""
sanity_check.py — 🩺 MODEL COLLAPSE DIAGNOSTIC (matches raw_df pickle snapshot pipeline)
Purpose: confirm whether Mamba/TCN are actually reacting to input features,
or whether they've collapsed to a learned bias that ignores the input
(explains the 99.6% always-SELL prediction pattern in evaluate_test_set.py).

Run: python3 sanity_check.py
"""
import os
import numpy as np
import torch
import pandas as pd
import xgboost as xgb
import logging

from train_models import (
    apply_vectorized_triple_barrier, build_quant_features,
    SEQUENCE_LEN, TIME_BARRIER, EMBARGO_GAP
)
from mamba_model import MambaQuantModel
from tcn_model import TCNQuantModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("SanityCheck")

# ✅ matches your train_models.py: snapshot_path = 'data/btc_1m_snapshot.pkl' (raw_df, pre-features)
SNAPSHOT_PATH = 'data/btc_1m_snapshot.pkl'

CLASS_NAMES = ['SELL', 'HOLD', 'BUY']


def load_data():
    """Rebuild the exact same features/labels evaluate_test_set.py uses —
    raw_df snapshot -> triple-barrier labels -> build_quant_features (deterministic,
    identical output every run since raw_df itself is frozen)."""
    if not os.path.exists(SNAPSHOT_PATH):
        raise FileNotFoundError(f"{SNAPSHOT_PATH} හමු නොවීය. train_models.py run කරලා තියෙනවද?")

    raw_df = pd.read_pickle(SNAPSHOT_PATH)
    log.info(f"📂 Loaded raw_df snapshot: {len(raw_df):,} rows")

    labels = apply_vectorized_triple_barrier(raw_df.copy())
    features_df = build_quant_features(raw_df.copy())

    valid_len = len(raw_df) - TIME_BARRIER
    features_df = features_df.iloc[:valid_len].reset_index(drop=True)
    labels = labels[:valid_len]

    return features_df, labels


def test_1_input_sensitivity(mamba, tcn, device, X_test, num_features):
    print("\n" + "=" * 70)
    print("🔬 TEST 1: INPUT SENSITIVITY (Random noise vs Real data)")
    print("=" * 70)

    n_samples = 8
    real_windows = []
    step = (len(X_test) - SEQUENCE_LEN) // n_samples
    for i in range(n_samples):
        start = i * step
        real_windows.append(X_test[start:start + SEQUENCE_LEN])
    real_batch = torch.tensor(np.stack(real_windows), dtype=torch.float32).to(device)

    random_batch = torch.rand((n_samples, SEQUENCE_LEN, num_features), dtype=torch.float32).to(device)

    mamba.eval(); tcn.eval()
    with torch.no_grad():
        real_mamba_probs = torch.softmax(mamba(real_batch), dim=1).cpu().numpy()
        rand_mamba_probs = torch.softmax(mamba(random_batch), dim=1).cpu().numpy()
        real_tcn_probs = torch.softmax(tcn(real_batch), dim=1).cpu().numpy()
        rand_tcn_probs = torch.softmax(tcn(random_batch), dim=1).cpu().numpy()

    def summarize(name, real_p, rand_p):
        real_std = real_p.std(axis=0)
        mean_diff = np.abs(real_p.mean(axis=0) - rand_p.mean(axis=0)).mean()
        print(f"\n   [{name}]")
        print(f"   Real-data windows -> mean probs: SELL={real_p[:,0].mean():.4f} HOLD={real_p[:,1].mean():.4f} BUY={real_p[:,2].mean():.4f}")
        print(f"   Random noise      -> mean probs: SELL={rand_p[:,0].mean():.4f} HOLD={rand_p[:,1].mean():.4f} BUY={rand_p[:,2].mean():.4f}")
        print(f"   Std-dev ACROSS 8 different real windows (higher = more input-reactive): {real_std}")
        print(f"   |mean(real) - mean(random)| avg diff: {mean_diff:.5f}")
        if real_std.max() < 0.01:
            print(f"   🔴 VERDICT: {name} output barely changes across 8 different real market windows.")
            print(f"      -> Model has likely collapsed onto a learned bias, ignoring input features.")
        elif mean_diff < 0.02:
            print(f"   🔴 VERDICT: {name} can't tell real market data from random noise.")
            print(f"      -> Model is not using the input meaningfully.")
        else:
            print(f"   🟢 VERDICT: {name} reacts differently to real vs random input — input IS being used.")

    summarize("MAMBA", real_mamba_probs, rand_mamba_probs)
    summarize("TCN", real_tcn_probs, rand_tcn_probs)


def test_2_xgb_feature_importance():
    print("\n" + "=" * 70)
    print("🔬 TEST 2: XGBOOST META-JUDGE FEATURE IMPORTANCE")
    print("=" * 70)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("weights/xgb_meta.json")

    names = ["mamba_sell", "mamba_hold", "mamba_buy",
             "tcn_sell", "tcn_hold", "tcn_buy",
             "lgb_sell", "lgb_hold", "lgb_buy",
             "macro_score", "macro_confidence", "macro_volume_norm"]

    importances = xgb_model.feature_importances_
    order = np.argsort(importances)[::-1]

    print(f"\n   {'Feature':<20} {'Importance':>12}")
    print(f"   {'-'*20} {'-'*12}")
    for i in order:
        name = names[i] if i < len(names) else f"feat_{i}"
        bar = '█' * int(importances[i] * 80)
        print(f"   {name:<20} {importances[i]:>10.4f}  {bar}")

    mamba_tcn_total = importances[0:6].sum()
    lgb_total = importances[6:9].sum() if len(importances) > 8 else 0.0
    macro_total = importances[9:12].sum() if len(importances) > 9 else 0.0

    print(f"\n   Mamba+TCN combined importance: {mamba_tcn_total:.4f}")
    print(f"   LightGBM combined importance:  {lgb_total:.4f}")
    print(f"   Macro (dummy=0 in training) importance: {macro_total:.4f}")

    if mamba_tcn_total < 0.15:
        print("\n   🔴 VERDICT: XGBoost barely uses Mamba/TCN outputs. It's leaning on")
        print("      LightGBM (or noise) to make the final call — consistent with Mamba/TCN collapse.")
    else:
        print("\n   🟢 VERDICT: XGBoost is meaningfully weighting Mamba/TCN outputs.")


def test_3_split_label_balance(labels):
    print("\n" + "=" * 70)
    print("🔬 TEST 3: LABEL DISTRIBUTION PER SPLIT (regime-shift check)")
    print("=" * 70)

    total_len = len(labels)
    base_end = int(total_len * 0.60)
    meta_start = base_end + EMBARGO_GAP
    meta_end = meta_start + int(total_len * 0.20)
    test_start = meta_end + EMBARGO_GAP

    splits = {
        "Base-Train (0 - 60%)": labels[:base_end],
        "Meta (60% - 80%)": labels[meta_start:meta_end],
        "Test / Holdout (80% - 100%)": labels[test_start:],
    }

    for name, seg in splits.items():
        u, c = np.unique(seg, return_counts=True)
        dist = dict(zip(u.astype(int), c))
        total = len(seg)
        sell_pct = dist.get(0, 0) / total * 100
        hold_pct = dist.get(1, 0) / total * 100
        buy_pct = dist.get(2, 0) / total * 100
        print(f"\n   {name} (n={total:,}):")
        print(f"     SELL: {sell_pct:5.1f}%   HOLD: {hold_pct:5.1f}%   BUY: {buy_pct:5.1f}%")

    base_sell_pct = np.mean(labels[:base_end] == 0) * 100
    test_sell_pct = np.mean(labels[test_start:] == 0) * 100
    diff = abs(base_sell_pct - test_sell_pct)
    print(f"\n   Base-Train SELL% vs Test SELL% diff: {diff:.1f} points")
    if diff > 10:
        print("   🔴 VERDICT: Significant regime shift between train and test periods —")
        print("      the market may have trended differently in each split, which can")
        print("      partially explain a directional bias collapse.")
    else:
        print("   🟢 VERDICT: Train/test label balance is reasonably consistent.")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"🩺 Running model collapse diagnostics... Device: {device}")

    features_df, labels = load_data()

    scaler = np.load('weights/feature_scaler.npz')
    f_min, f_max = scaler['min'], scaler['max']
    feat_arr = (features_df.values - f_min) / (f_max - f_min + 1e-7)

    total_len = len(feat_arr)
    test_start = int(total_len * 0.60) + EMBARGO_GAP + int(total_len * 0.20) + EMBARGO_GAP
    X_test = feat_arr[test_start:]
    num_features = X_test.shape[1]

    mamba = MambaQuantModel(input_dim=num_features, d_model=128, n_layers=3, n_classes=3).to(device)
    mamba.load_state_dict(torch.load("weights/mamba_best.pt", map_location=device, weights_only=True))

    tcn = TCNQuantModel(input_dim=num_features, num_channels=[64, 128, 128], n_classes=3).to(device)
    tcn.load_state_dict(torch.load("weights/tcn_best.pt", map_location=device, weights_only=True))

    test_1_input_sensitivity(mamba, tcn, device, X_test, num_features)
    test_2_xgb_feature_importance()
    test_3_split_label_balance(labels)

    print("\n" + "=" * 70)
    print("🩺 Diagnostics complete. Read the 🔴/🟢 verdicts above in order —")
    print("   Test 1 tells you WHERE the collapse is (Mamba? TCN? both?).")
    print("   Test 2 tells you whether XGBoost is compounding the problem.")
    print("   Test 3 rules in/out a simple train/test regime-shift explanation.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()