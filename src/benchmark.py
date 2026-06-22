"""Precision-specific benchmark harness.

Measures latency, throughput and PR-AUC across inference modes so the
speed/accuracy trade-off of quantization is reproducible. It always runs the
portable modes (PyTorch eager, ONNX Runtime FP32, ONNX Runtime INT8 dynamic) and
additionally times TensorRT engines if they were built on a GPU host.

    python -m src.train                 # produce a checkpoint + meta
    python -m src.optimize.export_onnx  # produce ONNX
    python -m src.benchmark             # print the table
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from src.config import CONFIG


def _load_eval_batch(cfg, n=4096):
    from src.train import prepare
    cats, nums, y, meta = prepare(cfg)
    return cats[:n], nums[:n], y[:n], meta


def _time_callable(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def _pr_auc(proba, y):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, proba))


def benchmark(cfg=CONFIG, batch=1024):
    cats, nums, y, meta = _load_eval_batch(cfg)
    rows = []

    # ---- 1) PyTorch eager (baseline) ----
    import torch
    from src.models.tabular_net import TabularFraudNet

    net = TabularFraudNet(meta["cardinalities"], len(meta["num_features"]), cfg)
    net.load_state_dict(torch.load(os.path.join(cfg.artifacts_dir, "fraud_net.pt"),
                                   map_location="cpu"))
    net.eval()
    ct = torch.from_numpy(cats[:batch]); nt = torch.from_numpy(nums[:batch])
    with torch.no_grad():
        proba = 1 / (1 + np.exp(-net(ct, nt).numpy()))
    lat = _time_callable(lambda: net(ct, nt))
    rows.append(("PyTorch eager (FP32)", lat, batch / lat, _pr_auc(proba, y[:batch])))

    # ---- 2) ONNX Runtime FP32 ----
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(os.path.join(cfg.artifacts_dir, "fraud_net.onnx"),
                                    providers=["CPUExecutionProvider"])
        feed = {"cats": cats[:batch], "nums": nums[:batch]}
        out = sess.run(None, feed)[0].ravel()
        proba = 1 / (1 + np.exp(-out))
        lat = _time_callable(lambda: sess.run(None, feed))
        rows.append(("ONNX Runtime FP32", lat, batch / lat, _pr_auc(proba, y[:batch])))
    except Exception as e:
        print(f"[skip ORT FP32] {e}")

    # ---- 3) ONNX Runtime INT8 (dynamic quantization) ----
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType

        q_path = os.path.join(cfg.artifacts_dir, "fraud_net_int8.onnx")
        quantize_dynamic(os.path.join(cfg.artifacts_dir, "fraud_net.onnx"),
                         q_path, weight_type=QuantType.QInt8)
        sess = ort.InferenceSession(q_path, providers=["CPUExecutionProvider"])
        feed = {"cats": cats[:batch], "nums": nums[:batch]}
        out = sess.run(None, feed)[0].ravel()
        proba = 1 / (1 + np.exp(-out))
        lat = _time_callable(lambda: sess.run(None, feed))
        rows.append(("ONNX Runtime INT8", lat, batch / lat, _pr_auc(proba, y[:batch])))
    except Exception as e:
        print(f"[skip ORT INT8] {e}")

    # ---- 4) TensorRT engines (GPU only, if present) ----
    for prec in ("fp16", "int8"):
        eng = os.path.join(cfg.artifacts_dir, f"fraud_net_{prec}.engine")
        if os.path.exists(eng):
            print(f"[info] TensorRT {prec} engine found — time it on the GPU host "
                  f"with your TRT runner and add the row.")

    # ---- report ----
    base = rows[0][1]
    print("\n%-24s %10s %14s %10s %9s" % ("mode", "lat(ms)", "throughput/s", "PR-AUC", "speedup"))
    print("-" * 72)
    for name, lat, thr, pr in rows:
        print("%-24s %10.3f %14.0f %10.4f %8.2fx" % (name, lat * 1000, thr, pr, base / lat))
    return rows


if __name__ == "__main__":
    benchmark()
