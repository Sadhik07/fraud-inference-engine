"""CPU smoke + correctness tests for the fraud engine."""
import numpy as np
import torch

from src.config import CONFIG
from src.data.synthetic_tx import generate
from src.features.graph_features import derive_features
from src.models.tabular_net import TabularFraudNet


def _small():
    cfg = CONFIG
    cfg.data.n_transactions = 4000
    cfg.data.n_cards = 800
    cfg.data.n_devices = 500
    return cfg


def test_generator_has_fraud():
    df = generate(_small())
    assert 0.0 < df.is_fraud.mean() < 0.5
    assert {"card_id", "device_id", "amount", "is_fraud"}.issubset(df.columns)


def test_graph_features_present():
    df = derive_features(generate(_small()))
    for c in ("card_degree", "device_degree", "ring_size",
              "amount_zscore", "txn_velocity_1h", "device_share", "time_since_prev"):
        assert c in df.columns
    # ring members should have larger components on average than non-fraud
    assert df.loc[df.is_fraud == 1, "ring_size"].mean() >= df.loc[df.is_fraud == 0, "ring_size"].mean()


def test_model_forward_and_backward():
    card = {"card_id": 20, "device_id": 20, "merchant_cat": 20, "hour_bucket": 20}
    net = TabularFraudNet(card, n_numeric=len(CONFIG.data.num_features))
    cats = torch.randint(0, 20, (16, 4))  # all columns share cardinality 20 here
    nums = torch.randn(16, len(CONFIG.data.num_features))
    logit = net(cats, nums)
    assert logit.shape == (16,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, torch.randint(0, 2, (16,)).float())
    loss.backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)
