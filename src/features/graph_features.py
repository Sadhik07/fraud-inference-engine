"""Graph-derived linkage features over a card/device bipartite graph.

Production stores the entity graph in Neo4j; here we build the same bipartite
card<->device graph in NetworkX and derive features that expose collusion rings:
node degrees, shared-device ratio, and connected-component ("ring") size. These
features are concatenated with the per-transaction numeric features before the
model sees them. `to_cypher()` reproduces the graph in Neo4j.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_graph(df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    cards = [f"c{c}" for c in df.card_id.to_numpy()]
    devs = [f"d{d}" for d in df.device_id.to_numpy()]
    edges = list(zip(cards, devs))
    g.add_edges_from(edges)
    return g


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add graph + behavioral features used by the model."""
    df = df.sort_values("ts").reset_index(drop=True).copy()
    g = build_graph(df)

    # connected-component id and size -> ring membership signal
    comp_id = {}
    comp_size = {}
    for i, comp in enumerate(nx.connected_components(g)):
        for nd in comp:
            comp_id[nd] = i
            comp_size[nd] = len(comp)

    card_node = "c" + df.card_id.astype(str)
    dev_node = "d" + df.device_id.astype(str)

    df["card_degree"] = [g.degree[n] for n in card_node]
    df["device_degree"] = [g.degree[n] for n in dev_node]
    df["ring_size"] = [comp_size.get(n, 1) for n in dev_node]

    # behavioral: amount z-score within card, velocity, device sharing, recency
    df["amount_zscore"] = df.groupby("card_id")["amount"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-6))
    df["txn_velocity_1h"] = _velocity(df, window_s=3600)
    dev_counts = df.device_id.value_counts()
    card_counts = df.card_id.value_counts()
    df["device_share"] = df.device_id.map(dev_counts) / df.card_id.map(card_counts).clip(lower=1)
    df["time_since_prev"] = df.groupby("card_id")["ts"].diff().fillna(1e6)
    return df


def _velocity(df: pd.DataFrame, window_s: float) -> np.ndarray:
    """Count of prior transactions on the same card within a trailing window."""
    out = np.zeros(len(df))
    last_idx = {}
    times = df.ts.to_numpy()
    cards = df.card_id.to_numpy()
    history: dict[int, list[float]] = {}
    for i in range(len(df)):
        c = cards[i]
        h = history.setdefault(c, [])
        # drop old
        while h and times[i] - h[0] > window_s:
            h.pop(0)
        out[i] = len(h)
        h.append(times[i])
    return out


def to_cypher_sample(df: pd.DataFrame, limit: int = 200) -> str:
    lines = []
    for _, r in df.head(limit).iterrows():
        lines.append(
            f"MERGE (c:Card {{id:{int(r.card_id)}}}) "
            f"MERGE (d:Device {{id:{int(r.device_id)}}}) "
            f"MERGE (c)-[:USED_ON {{amount:{r.amount}}}]->(d);"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from src.data.synthetic_tx import generate

    df = derive_features(generate())
    cols = ["card_degree", "device_degree", "ring_size", "amount_zscore",
            "txn_velocity_1h", "device_share", "time_since_prev"]
    print(df[cols + ["is_fraud"]].describe().round(2))
