"""
overfit_test.py — 🩺 CAN THE MODEL EVEN MEMORIZE A TINY DATASET?
Classic ML debugging technique: if a model can't drive loss near-zero on a
TINY dataset it should easily memorize, something structural is broken
(dead gradients, frozen layers, wrong loss/label alignment, etc.) — it's
not a "not enough signal" problem, it's a "not learning at all" problem.

Run: python3 overfit_test.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from train_models import (
    apply_vectorized_triple_barrier, build_quant_features,
    QuantDataset, SEQUENCE_LEN, TIME_BARRIER, RANDOM_SEED
)
from mamba_model import MambaQuantModel
from tcn_model import TCNQuantModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("OverfitTest")

SNAPSHOT_PATH = 'data/btc_1m_snapshot.pkl'
TINY_ROWS = 700          # -> ~200 sliding-window samples with SEQUENCE_LEN=500
OVERFIT_EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3                # higher LR — fine for a deliberate overfit test


def run_overfit(model_cls, model_name, X, y, num_features, device, **model_kwargs):
    print("\n" + "=" * 70)
    print(f"🔬 OVERFIT TEST: {model_name}")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    ds = QuantDataset(X, y, SEQUENCE_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"   Dataset size: {len(ds)} samples, {len(loader)} batches/epoch")

    model = model_cls(input_dim=num_features, n_classes=3, **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)  # no weight decay — pure overfit test
    criterion = nn.CrossEntropyLoss()

    grad_norm_first_batch = None

    for epoch in range(OVERFIT_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            if epoch == 0 and batch_idx == 0:
                # Capture gradient norms on the very first backward pass —
                # tells us whether gradients are actually reaching each part of the model.
                total_norm = 0.0
                zero_grad_params = 0
                total_params_with_grad = 0
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        g_norm = p.grad.data.norm(2).item()
                        total_norm += g_norm ** 2
                        total_params_with_grad += 1
                        if g_norm < 1e-8:
                            zero_grad_params += 1
                grad_norm_first_batch = total_norm ** 0.5
                print(f"   [First backward pass] Total grad norm: {grad_norm_first_batch:.6f} | "
                      f"Params with ~zero grad: {zero_grad_params}/{total_params_with_grad}")

            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / len(loader)
        acc = correct / total * 100

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:>3}/{OVERFIT_EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {acc:.1f}%")

    print(f"\n   Final Loss: {avg_loss:.4f} | Final Train Acc: {acc:.1f}%")
    if avg_loss < 0.15 and acc > 90:
        print(f"   🟢 VERDICT: {model_name} CAN memorize this tiny dataset — gradient flow is healthy.")
        print(f"      -> The full-dataset collapse is more likely a signal/scale/hyperparameter")
        print(f"         issue than a structural bug in this model.")
    elif avg_loss > 0.6:
        print(f"   🔴 VERDICT: {model_name} CANNOT memorize even {len(ds)} samples after {OVERFIT_EPOCHS} epochs.")
        print(f"      -> This points to a structural problem: dead/vanishing gradients,")
        print(f"         a frozen layer, or a label/feature misalignment bug.")
    else:
        print(f"   ⚠️  VERDICT: Partial memorization — some learning capacity, but not full.")
        print(f"      -> Borderline; worth longer overfit run or LR tuning before concluding.")

    return avg_loss, acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"🩺 Device: {device}")

    raw_df = pd.read_pickle(SNAPSHOT_PATH)
    raw_subset = raw_df.iloc[:TINY_ROWS].reset_index(drop=True)
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

    log.info(f"📊 Tiny dataset ready: {len(X):,} rows -> ~{len(X)-SEQUENCE_LEN} sliding-window samples")
    u, c = np.unique(y, return_counts=True)
    log.info(f"📊 Tiny-set label distribution: {dict(zip(u.astype(int), c))}")

    run_overfit(MambaQuantModel, "MAMBA", X, y, num_features, device,
                d_model=128, n_layers=3)
    run_overfit(TCNQuantModel, "TCN", X, y, num_features, device,
                num_channels=[64, 128, 128])

    print("\n" + "=" * 70)
    print("🩺 If BOTH models fail to overfit even this tiny dataset, the bug is")
    print("   almost certainly structural (gradient flow / label alignment / a")
    print("   frozen or misconfigured layer) rather than a data-scale problem.")
    print("   If ONE model overfits fine and the other doesn't, that isolates")
    print("   the bug to the failing architecture specifically.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
