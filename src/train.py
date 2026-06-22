"""Train the tabular fraud model on synthetic transactions with graph features.

Encodes categoricals to contiguous ids, standardizes numerics, trains with a
class-weighted BCE loss, and reports PR-AUC / ROC-AUC on a held-out split. Saves
the checkpoint + preprocessing metadata for ONNX export and benchmarking.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import CONFIG
from src.data.synthetic_tx import generate
from src.features.graph_features import derive_features
from src.models.tabular_net import TabularFraudNet

CAT = list(CONFIG.data.cat_features)
NUM = list(CONFIG.data.num_features)


def prepare(cfg=CONFIG):
    df = derive_features(generate(cfg))
    encoders = {}
    for c in CAT:
        codes, uniques = df[c].factorize()
        df[c + "_code"] = codes
        encoders[c] = len(uniques)
    cats = df[[c + "_code" for c in CAT]].to_numpy(np.int64)
    nums = df[NUM].to_numpy(np.float32)
    nums = np.nan_to_num(nums, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = nums.mean(0), nums.std(0) + 1e-6
    nums = (nums - mu) / sd
    nums = np.clip(nums, -8.0, 8.0).astype(np.float32)  # tame heavy-tailed outliers
    y = df.is_fraud.to_numpy(np.float32)
    meta = {"cardinalities": {c: int(encoders[c]) for c in CAT},
            "num_features": NUM, "cat_features": CAT,
            "mu": mu.tolist(), "sd": sd.tolist()}
    return cats, nums, y, meta


def main(cfg=CONFIG):
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    cats, nums, y, meta = prepare(cfg)
    n = len(y); idx = np.random.default_rng(0).permutation(n)
    cut = int(n * 0.8)
    tr, te = idx[:cut], idx[cut:]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = TensorDataset(torch.from_numpy(cats[tr]), torch.from_numpy(nums[tr]),
                       torch.from_numpy(y[tr]))
    dl = DataLoader(ds, batch_size=cfg.model.batch_size, shuffle=True)

    net = TabularFraudNet(meta["cardinalities"], len(NUM), cfg).to(device)
    pos_weight = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)], device=device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.model.lr)

    for epoch in range(cfg.model.epochs):
        net.train(); tot = 0.0
        for cb, nb, yb in dl:
            cb, nb, yb = cb.to(device), nb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(net(cb, nb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            opt.step()
            tot += loss.item() * len(yb)
        print(f"epoch {epoch + 1}/{cfg.model.epochs}  loss={tot / len(tr):.4f}")

    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(cats[te]).to(device),
                     torch.from_numpy(nums[te]).to(device)).cpu().numpy()
    proba = 1 / (1 + np.exp(-logits))
    pr = average_precision_score(y[te], proba)
    roc = roc_auc_score(y[te], proba)
    print(f"\nheld-out PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}")

    torch.save(net.state_dict(), os.path.join(cfg.artifacts_dir, "fraud_net.pt"))
    meta["metrics"] = {"pr_auc": float(pr), "roc_auc": float(roc)}
    json.dump(meta, open(os.path.join(cfg.artifacts_dir, "meta.json"), "w"), indent=2)
    print(f"saved -> {cfg.artifacts_dir}/fraud_net.pt")
    return pr, roc


if __name__ == "__main__":
    main()
