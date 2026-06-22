"""Configuration for the GPU-accelerated fraud detection inference engine."""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    n_transactions: int = 60_000   # synthetic volume (real IEEE-CIS ≈ 590K)
    fraud_rate: float = 0.035      # close to IEEE-CIS positive rate
    n_cards: int = 6_000
    n_devices: int = 4_000
    seed: int = 13
    # categorical (entity) features get embeddings; numeric features are dense
    cat_features: tuple = ("card_id", "device_id", "merchant_cat", "hour_bucket")
    num_features: tuple = ("amount", "amount_zscore", "txn_velocity_1h",
                            "device_share", "card_degree", "device_degree",
                            "ring_size", "time_since_prev")


@dataclass
class ModelConfig:
    emb_dim: int = 16
    hidden: int = 96
    attn_dim: int = 64
    dropout: float = 0.2
    epochs: int = 4
    batch_size: int = 512
    lr: float = 2e-3


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    artifacts_dir: str = "artifacts"


CONFIG = Config()
