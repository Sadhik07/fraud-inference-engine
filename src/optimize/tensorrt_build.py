"""Build TensorRT engines (FP16 / INT8) from the exported ONNX model.

REQUIRES an NVIDIA GPU with CUDA + TensorRT installed — this is the GPU path that
produces the headline FP16/INT8 latency numbers. On a CPU-only machine this script
will detect the missing runtime and tell you to use `benchmark.py` (ONNX Runtime),
which produces real CPU numbers everywhere.

Install (GPU host):
    pip install tensorrt pycuda
    # CUDA 12.x + matching TensorRT 10.x

Usage:
    python -m src.optimize.tensorrt_build --precision fp16
    python -m src.optimize.tensorrt_build --precision int8 --calib 2048
"""
from __future__ import annotations

import argparse
import os

from src.config import CONFIG


def build(precision: str = "fp16", calib: int = 1024, cfg=CONFIG):
    onnx_path = os.path.join(cfg.artifacts_dir, "fraud_net.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError("Export first: python -m src.optimize.export_onnx")

    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not installed (CPU-only host).")
        print("This script needs an NVIDIA GPU + TensorRT. For portable, reproducible")
        print("numbers on any machine run:  python -m src.benchmark")
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # NOTE: attach an Int8 calibrator over `calib` representative batches here.
        # See src/benchmark.py for how calibration data is assembled.

    engine = builder.build_serialized_network(network, config)
    out = os.path.join(cfg.artifacts_dir, f"fraud_net_{precision}.engine")
    with open(out, "wb") as f:
        f.write(engine)
    print(f"built {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp16", choices=["fp16", "int8"])
    ap.add_argument("--calib", type=int, default=1024)
    args = ap.parse_args()
    build(args.precision, args.calib)
