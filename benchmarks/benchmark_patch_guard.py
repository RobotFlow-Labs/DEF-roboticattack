#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from def_roboticattack.pipeline.runtime import DefenseRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline patch guard runtime")
    parser.add_argument("--backend", default="auto", help="auto|cuda|mlx|cpu")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    return parser.parse_args()


def make_inputs(runtime: DefenseRuntime, batch_size: int, height: int, width: int):
    if runtime.backend_info.name == "cuda":
        try:
            import torch

            device = runtime.backend_info.torch_device
            return torch.rand(batch_size, 3, height, width, device=device)
        except Exception:
            return np.random.rand(batch_size, 3, height, width).astype(np.float32)
    return np.random.rand(batch_size, 3, height, width).astype(np.float32)


def main() -> None:
    args = parse_args()
    runtime = DefenseRuntime(backend=args.backend)
    images = make_inputs(runtime, args.batch_size, args.height, args.width)

    for _ in range(args.warmup):
        runtime.sanitize_and_score(images)

    samples = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        runtime.sanitize_and_score(images)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        samples.append(dt_ms)

    mean_ms = statistics.mean(samples)
    std_ms = statistics.pstdev(samples)
    throughput = args.batch_size / (mean_ms / 1000.0)

    out = {
        "backend": runtime.backend_info.name,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "latency_ms_mean": mean_ms,
        "latency_ms_std": std_ms,
        "throughput_samples_per_sec": throughput,
    }

    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"patch_guard_{runtime.backend_info.name}.json"
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    print(f"saved: {out_file}")


if __name__ == "__main__":
    main()
