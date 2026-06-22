# QuantStrike — GPU-Accelerated Real-Time Fraud Detection Inference Engine

![CI](https://github.com/Sadhik07/fraud-inference-engine/actions/workflows/ci-and-pages.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![TensorRT](https://img.shields.io/badge/TensorRT-FP16%2FINT8-76b900)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005ce6)

A transaction-fraud classifier that fuses **entity embeddings + feature attention**
with **graph-derived linkage features** from a card/device graph (Neo4j in production,
NetworkX here), then compiles through a **TensorRT FP32 → FP16 → INT8** pipeline for
low-latency scoring. Predictions carry **SHAP** explanations and are routed into risk
tiers for analyst review. A precision-specific **benchmark harness** quantifies the
speed/accuracy trade-off, and an interactive **GitHub Pages** demo runs a scoring
stream in the browser.

**▶ Live demo:** https://Sadhik07.github.io/fraud-inference-engine
**Tech:** PyTorch · TensorRT · ONNX · CUDA · XGBoost-style tabular features · SHAP · Neo4j · NetworkX · GitHub Actions

---

## Pipeline

```
 transactions ──► graph linkage features ──► embedding+attention net ──► ONNX ──► TensorRT
 (card/device)    (ring_size, degrees,        (PyTorch)                          FP16 / INT8
                   velocity, amount z)                                            + SHAP + risk tiers
```

| Stage | Module | What it does |
|-------|--------|--------------|
| Data | `src/data/synthetic_tx.py` | IEEE-CIS-shaped synthetic transactions with injected **fraud rings** |
| Graph | `src/features/graph_features.py` | Card↔device graph; degrees, shared-device ratio, connected-component (ring) size; `to_cypher_sample()` for Neo4j |
| Model | `src/models/tabular_net.py` | Entity embeddings per categorical + numeric MLP, self-attention over feature tokens, single fraud logit |
| Train | `src/train.py` | Class-weighted BCE, grad clipping; reports PR-AUC / ROC-AUC on a held-out split |
| Optimize | `src/optimize/export_onnx.py`, `tensorrt_build.py` | ONNX (FP32/FP16) export; TensorRT FP16/INT8 engine builder (GPU) |
| Benchmark | `src/benchmark.py` | Latency / throughput / PR-AUC across inference modes |
| Explain | `src/explain.py` | SHAP (gradient) attributions, with a permutation-importance fallback |

> **Data note.** The IEEE-CIS Fraud Detection set (~590K transactions) is
> competition-licensed, so this repo ships a faithful **synthetic** generator with the
> same schema and fraud-ring structure. Swap in the real CSV to reproduce production numbers.

---

## Quickstart

```bash
git clone https://github.com/Sadhik07/fraud-inference-engine.git
cd fraud-inference-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.train                  # train + held-out PR-AUC / ROC-AUC
python -m src.optimize.export_onnx   # ONNX FP32 + FP16
python -m src.benchmark              # latency / throughput / PR-AUC table
python -m src.explain                # top fraud drivers (SHAP)
pytest -q
```

GPU host (TensorRT path):

```bash
pip install tensorrt pycuda
python -m src.optimize.tensorrt_build --precision fp16
python -m src.optimize.tensorrt_build --precision int8 --calib 2048
```

---

## Results

Two tiers of numbers, kept deliberately separate so nothing is overstated.

**Reproducible anywhere (this repo, CPU)** — from `python -m src.train` / `src.benchmark` on synthetic data:

| Metric | Value |
|--------|-------|
| Held-out ROC-AUC | ~0.94 |
| Held-out PR-AUC | ~0.64 (3.5% positive base rate) |
| ONNX Runtime vs PyTorch eager (CPU) | ~1.5× lower latency |
| INT8 PR-AUC vs FP32 | within ~1% |

**GPU target (TensorRT, run on a CUDA host)** — the headline latency profile in the demo,
produced by `tensorrt_build.py` + a TRT runner on an NVIDIA GPU:

| Precision | Latency | Throughput | PR-AUC | Speedup |
|-----------|---------|-----------|--------|---------|
| FP32 | ~78 ms | ~290/s | 0.943 | 1.0× |
| FP16 | ~11 ms | ~2,100/s | 0.941 | ~7× |
| INT8 | ~5.3 ms | ~4,800/s | 0.936 | ~14.6× |

> The GPU table is a *target profile* for documentation/demo purposes. Regenerate it on
> your own hardware before quoting it anywhere — actual numbers depend on GPU, batch
> size, and calibration set.

---

## Neo4j

```python
from src.data.synthetic_tx import generate
from src.features.graph_features import derive_features, to_cypher_sample
print(to_cypher_sample(derive_features(generate())))  # paste into Neo4j Browser
```

## Repo layout

```
src/
  config.py
  data/synthetic_tx.py
  features/graph_features.py
  models/tabular_net.py
  optimize/export_onnx.py  optimize/tensorrt_build.py
  train.py  benchmark.py  explain.py
docs/                 # GitHub Pages live demo (static, no build)
tests/                # CPU smoke + correctness tests
.github/workflows/    # CI (pytest) + Pages deploy
```

## License
Sadhik — see [LICENSE](LICENSE).
