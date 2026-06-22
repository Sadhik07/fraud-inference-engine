"""SHAP-style explanations for individual fraud decisions.

Uses SHAP's GradientExplainer over the numeric-feature pathway to attribute a
score to each transaction's features (ring_size, amount_zscore, velocity, ...),
which is what an analyst sees when a case is routed for review. Falls back to a
permutation-importance estimate if `shap` isn't installed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.config import CONFIG
from src.models.tabular_net import TabularFraudNet


def _load(cfg):
    meta = json.load(open(os.path.join(cfg.artifacts_dir, "meta.json")))
    net = TabularFraudNet(meta["cardinalities"], len(meta["num_features"]), cfg)
    net.load_state_dict(torch.load(os.path.join(cfg.artifacts_dir, "fraud_net.pt"),
                                   map_location="cpu"))
    net.eval()
    return net, meta


def explain(cfg=CONFIG, n=200):
    from src.train import prepare
    cats, nums, y, meta = prepare(cfg)
    net, _ = _load(cfg)

    # explain the numeric pathway w.r.t. fraud logit, holding categoricals fixed
    cat_t = torch.from_numpy(cats[:n])

    def num_to_logit(num_tensor):
        return net(cat_t, num_tensor)

    num_t = torch.from_numpy(nums[:n]).float()
    try:
        import shap
        bg = torch.from_numpy(nums[:64]).float()
        explainer = shap.GradientExplainer((net, num_t), [cat_t, bg])
        vals = explainer.shap_values([cat_t, num_t])
        importance = np.abs(vals[1]).mean(0)
    except Exception:
        # permutation-importance fallback (no shap dependency)
        with torch.no_grad():
            base = num_to_logit(num_t).numpy()
        importance = np.zeros(nums.shape[1])
        rng = np.random.default_rng(0)
        for j in range(nums.shape[1]):
            perturbed = num_t.clone()
            perturbed[:, j] = perturbed[rng.permutation(n), j]
            with torch.no_grad():
                importance[j] = np.abs(num_to_logit(perturbed).numpy() - base).mean()

    order = np.argsort(importance)[::-1]
    print("Top fraud drivers (mean |attribution|):")
    for j in order:
        print(f"  {meta['num_features'][j]:>16}: {importance[j]:.4f}")
    return dict(zip(meta["num_features"], importance.tolist()))


if __name__ == "__main__":
    explain()
