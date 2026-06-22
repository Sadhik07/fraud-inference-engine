"""Synthetic card-transaction generator with an IEEE-CIS-like schema.

The real engine trains on the IEEE-CIS Fraud Detection dataset (~590K rows). That
data is competition-licensed, so this module synthesizes transactions with the
same shape: card/device entities, merchant categories, amounts, and — critically
— *fraud rings* where a small set of devices/cards collude. Ring structure is what
the graph features in `features/graph_features.py` are designed to surface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


def generate(cfg=CONFIG) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.data.seed)
    n = cfg.data.n_transactions

    card_id = rng.integers(0, cfg.data.n_cards, n)
    device_id = rng.integers(0, cfg.data.n_devices, n)
    merchant_cat = rng.integers(0, 20, n)
    hour = rng.integers(0, 24, n)
    hour_bucket = (hour // 6)  # night / morning / afternoon / evening

    # legit amounts: log-normal; timestamps over ~30 days
    amount = np.round(rng.lognormal(3.0, 1.0, n), 2)
    ts = np.sort(rng.uniform(0, 30 * 24 * 3600, n))

    is_fraud = np.zeros(n, dtype=int)

    # ---- inject fraud rings -------------------------------------------------
    # a ring = few devices hammering many cards with off-hours, higher amounts.
    # Budget the number of ring transactions so overall fraud ~ cfg.fraud_rate,
    # reserving a thin slice for non-ring point fraud below.
    ring_budget = int(cfg.data.fraud_rate * n * 0.85)
    n_rings = max(4, n // 1500)
    per_ring = max(10, ring_budget // n_rings)
    for _ in range(n_rings):
        ring_devices = rng.integers(0, cfg.data.n_devices, rng.integers(2, 6))
        ring_cards = rng.integers(0, cfg.data.n_cards, rng.integers(20, 120))
        k = int(rng.integers(max(8, per_ring // 2), per_ring + 1))
        idx = rng.integers(0, n, k)
        device_id[idx] = rng.choice(ring_devices, k)
        card_id[idx] = rng.choice(ring_cards, k)
        amount[idx] = np.round(rng.lognormal(4.2, 0.7, k), 2)  # larger
        hour[idx] = rng.integers(0, 5, k)                      # off-hours
        hour_bucket[idx] = 0
        is_fraud[idx] = 1

    # a thin layer of non-ring (point) fraud
    point = rng.random(n) < (cfg.data.fraud_rate * 0.15)
    is_fraud[point] = 1
    amount[point] *= rng.uniform(2, 5, point.sum())

    df = pd.DataFrame({
        "txn_id": np.arange(n),
        "ts": ts,
        "card_id": card_id,
        "device_id": device_id,
        "merchant_cat": merchant_cat,
        "hour": hour,
        "hour_bucket": hour_bucket,
        "amount": amount,
        "is_fraud": is_fraud,
    })
    return df


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(f"\nrows={len(df):,}  fraud={df.is_fraud.mean()*100:.2f}%")
