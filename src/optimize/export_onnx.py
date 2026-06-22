"""Export the trained fraud model to ONNX (FP32) and FP16 variants.

ONNX is the portable handoff format consumed by both ONNX Runtime (CPU/GPU) and
the TensorRT engine builder in tensorrt_build.py.
"""
from __future__ import annotations

import json
import os

import torch

from src.config import CONFIG
from src.models.tabular_net import TabularFraudNet


def export(cfg=CONFIG):
    meta = json.load(open(os.path.join(cfg.artifacts_dir, "meta.json")))
    net = TabularFraudNet(meta["cardinalities"], len(meta["num_features"]), cfg)
    net.load_state_dict(torch.load(os.path.join(cfg.artifacts_dir, "fraud_net.pt"),
                                   map_location="cpu"))
    net.eval()

    cats = torch.zeros(1, len(meta["cat_features"]), dtype=torch.long)
    nums = torch.zeros(1, len(meta["num_features"]), dtype=torch.float32)
    onnx_path = os.path.join(cfg.artifacts_dir, "fraud_net.onnx")
    kwargs = dict(
        input_names=["cats", "nums"], output_names=["logit"],
        dynamic_axes={"cats": {0: "batch"}, "nums": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(net, (cats, nums), onnx_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(net, (cats, nums), onnx_path, **kwargs)
    print(f"exported {onnx_path}")

    try:
        import onnx
        from onnxconverter_common import float16

        m16 = float16.convert_float_to_float16(onnx.load(onnx_path), keep_io_types=True)
        fp16_path = os.path.join(cfg.artifacts_dir, "fraud_net_fp16.onnx")
        onnx.save(m16, fp16_path)
        print(f"exported {fp16_path}")
    except Exception as e:  # pragma: no cover
        print(f"[skip fp16] {e}")


if __name__ == "__main__":
    export()
