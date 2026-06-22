"""Tabular fraud model: entity embeddings + feature attention.

Categorical entities (card, device, merchant category, hour bucket) get learned
embeddings; numeric features pass through a small MLP. A self-attention block over
the per-feature token representations lets the model weight, e.g., ring_size and
amount_zscore together. Output is a single fraud logit.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CONFIG


class TabularFraudNet(nn.Module):
    def __init__(self, cat_cardinalities: dict[str, int], n_numeric: int, cfg=CONFIG):
        super().__init__()
        m = cfg.model
        self.cat_names = list(cat_cardinalities.keys())
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(card + 1, m.emb_dim) for name, card in cat_cardinalities.items()
        })
        self.num_proj = nn.Sequential(
            nn.Linear(n_numeric, m.emb_dim), nn.ReLU(), nn.LayerNorm(m.emb_dim)
        )
        # each categorical + the numeric block is one "token"
        self.n_tokens = len(self.cat_names) + 1
        self.attn = nn.MultiheadAttention(m.emb_dim, num_heads=4, dropout=m.dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_tokens * m.emb_dim, m.hidden), nn.ReLU(), nn.Dropout(m.dropout),
            nn.Linear(m.hidden, 1),
        )

    def forward(self, cats: torch.Tensor, nums: torch.Tensor):
        # cats: (B, n_cat) long ; nums: (B, n_numeric) float
        tokens = []
        for i, name in enumerate(self.cat_names):
            tokens.append(self.embeddings[name](cats[:, i]))
        tokens.append(self.num_proj(nums))
        x = torch.stack(tokens, dim=1)        # (B, n_tokens, emb_dim)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        return self.head(x).squeeze(-1)       # (B,) logit


if __name__ == "__main__":
    card = {"card_id": 6000, "device_id": 4000, "merchant_cat": 20, "hour_bucket": 4}
    net = TabularFraudNet(card, n_numeric=len(CONFIG.data.num_features))
    cats = torch.randint(0, 20, (8, 4))
    nums = torch.randn(8, len(CONFIG.data.num_features))
    print("logit shape:", net(cats, nums).shape,
          "| params:", sum(p.numel() for p in net.parameters()))
