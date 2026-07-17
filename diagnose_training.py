"""
diagnose_training.py — 🩺 GRADSCALER OVERFLOW DIAGNOSTIC
Trains Mamba on a SMALL subset (few thousand rows) for a few epochs, with
GradScaler skip-counting instrumented in, so we can see in ~1-2 minutes:
  1) Is train loss actually decreasing epoch-to-epoch? (real learning happening)
  2) What % of optimizer steps are being silently skipped due to FP16 overflow?

Run: python3 diagnose_training.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import logging

from train_models import (
    apply_vectorized_triple_barrier, build_quant_features,
    QuantDataset, SEQUENCE_LEN, TIME_BARRIER, RANDOM_SEED
)
from mamba_model import MambaQuantModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("Diagnose")

SNAPSHOT_PATH = 'data/btc_1m_snapshot.pkl'
SUBSET_ROWS = 8000       # small enough to finish in ~1-2 min
TEST_EPOCHS = 5
BATCH_SIZE = 32
LR = 3e-4


def run_variant(X, y, num_features, device, use_amp: bool, label: str):
    print("\n" + "=" * 70)
    print(f"🔬 VARIANT: {label} (AMP/FP16 {'ENABLED' if use_amp else 'DISABLED — pure FP32'})")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    ds = QuantDataset(X, y, SEQUENCE_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = MambaQuantModel(input_dim=num_features, d_model=128, n_layers=3, n_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=use_amp)

    for epoch in range(TEST_EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        skipped_steps = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            with autocast('cuda', enabled=use_amp):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            if use_amp:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_after < scale_before:
                    skipped_steps += 1
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        skip_pct = (skipped_steps / max(1, n_batches)) * 100
        print(f"   Epoch {epoch+1}/{TEST_EPOCHS} | Avg Train Loss: {avg_loss:.4f} | "
              f"Batches: {n_batches} | Skipped steps (overflow): {skipped_steps} ({skip_pct:.1f}%)")

    return avg_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"🩺 Device: {device}")

    if not os.path.exists(SNAPSHOT_PATH):
        raise FileNotFoundError(f"{SNAPSHOT_PATH} හමු නොවීය.")

    raw_df = pd.read_pickle(SNAPSHOT_PATH)
    log.info(f"📂 Loaded raw_df: {len(raw_df):,} rows — using first {SUBSET_ROWS:,} rows only for this quick test")

    raw_subset = raw_df.iloc[:SUBSET_ROWS].reset_index(drop=True)
    labels = apply_vectorized_triple_barrier(raw_subset.copy())
    features_df = build_quant_features(raw_subset.copy())

    valid_len = len(raw_subset) - TIME_BARRIER
    features_df = features_df.iloc[:valid_len].reset_index(drop=True)
    labels = labels[:valid_len]

    feat_raw = features_df.values
    f_min = feat_raw.min(axis=0)
    f_max = feat_raw.max(axis=0)
    X = (feat_raw - f_min) / (f_max - f_min + 1e-7)
    y = labels
    num_features = X.shape[1]

    log.info(f"📊 Subset ready: {len(X):,} rows, {num_features} features")

    # Variant A: AMP/FP16 enabled — same as your actual train_deep_model()
    loss_amp = run_variant(X, y, num_features, device, use_amp=(device.type == 'cuda'), label="Your current setup")

    # Variant B: AMP disabled — pure FP32, to isolate whether FP16 is the cause
    loss_fp32 = run_variant(X, y, num_features, device, use_amp=False, label="FP32 control")

    print("\n" + "=" * 70)
    print("🩺 FINAL DIAGNOSIS")
    print("=" * 70)
    print(f"   AMP/FP16 final epoch loss:  {loss_amp:.4f}")
    print(f"   FP32 control final epoch loss: {loss_fp32:.4f}")

    if loss_amp > 0.65 and loss_fp32 < 0.5:
        print("\n   🔴 CONFIRMED: AMP/FP16 path fails to learn, FP32 path DOES learn.")
        print("      -> GradScaler overflow-skip is very likely killing your real training runs.")
        print("      -> Fix: disable AMP for Mamba/TCN training, or add loss/gradient clipping")
        print("         + investigate why FP16 overflows (check for exploding activations,")
        print("         possibly from the mamba_model.py dt_proj re-init issue flagged earlier).")
    elif loss_amp > 0.65 and loss_fp32 > 0.65:
        print("\n   🔴 BOTH variants failed to learn — FP16 isn't the (sole) cause.")
        print("      -> Root cause is likely elsewhere: mamba_model.py's self.apply(self._init_weights)")
        print("         overwriting mamba_ssm's internal dt_proj initialization (flagged earlier),")
        print("         learning rate, or a data/label alignment issue.")
    else:
        print("\n   🟢 Both variants show loss decreasing — training mechanics look OK on this subset.")
        print("      -> The full-dataset collapse may be a different issue (e.g. subprocess")
        print("         re-seeding, class-weight scale, or something specific to the full run).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
