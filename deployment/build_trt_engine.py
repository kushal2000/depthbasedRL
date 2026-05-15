"""Build Fast-FoundationStereo TensorRT engines from ONNX exports.

Example:

    ONNX_DIR=/home/tylerlum/github_repos/Fast-FoundationStereo/weights/23-36-37/onnx_384x224_iters4
    python /home/tylerlum/github_repos/Fast-FoundationStereo/scripts/make_onnx.py \
      --model_dir /home/tylerlum/github_repos/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
      --save_path "${ONNX_DIR}" \
      --height 224 --width 384 --valid_iters 4 --max_disp 192
    python deployment/build_trt_engine.py --onnx_dir "${ONNX_DIR}"
"""

from __future__ import annotations

import argparse
import sys
import time

import tensorrt as trt


def build(onnx_path: str, engine_path: str, *, fp16: bool = True, workspace_gb: int = 4) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for idx in range(parser.num_errors):
            print(parser.get_error(idx), file=sys.stderr)
        raise RuntimeError(f"TensorRT ONNX parse failed for {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) * (1 << 30))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError(f"TensorRT build returned None for {onnx_path}")
    engine_bytes = bytes(serialized)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"built {engine_path} ({len(engine_bytes) / 1024 / 1024:.1f} MiB, {elapsed:.1f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", required=True)
    parser.add_argument("--fp16", type=int, default=1)
    parser.add_argument("--workspace_gb", type=int, default=4)
    args = parser.parse_args()

    for name in ("feature_runner", "post_runner"):
        print(f"=== building {name} ===", flush=True)
        build(
            f"{args.onnx_dir}/{name}.onnx",
            f"{args.onnx_dir}/{name}.engine",
            fp16=bool(args.fp16),
            workspace_gb=int(args.workspace_gb),
        )


if __name__ == "__main__":
    main()
