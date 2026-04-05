"""Evaluation script for PatchDetectorNet defense model.

Tests detection accuracy, latency, and throughput on synthetic adversarial data.
Generates a comprehensive report in JSON and prints summary.

Usage:
    python -m def_roboticattack.evaluate --checkpoint /path/to/best.pth
    python -m def_roboticattack.evaluate --checkpoint /path/to/best.pth --num-samples 10000
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from def_roboticattack.data.synthetic import SyntheticPatchDataset
from def_roboticattack.models.patch_detector import PatchDetectorNet
from def_roboticattack.pipeline.runtime import DefenseRuntime


@torch.no_grad()
def evaluate_accuracy(
    model: PatchDetectorNet,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate classification accuracy, precision, recall, F1."""
    model.eval()
    tp = fp = tn = fn = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        probs = model.predict_proba(images).squeeze(-1)
        preds = (probs > 0.5).float()

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


@torch.no_grad()
def evaluate_latency(
    model: PatchDetectorNet,
    device: torch.device,
    batch_size: int = 8,
    image_size: int = 224,
    warmup: int = 50,
    iterations: int = 500,
) -> dict:
    """Benchmark inference latency and throughput."""
    model.eval()
    dummy = torch.rand(batch_size, 3, image_size, image_size, device=device)

    # Warmup
    for _ in range(warmup):
        _ = model(dummy)
    torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(latencies)
    std_ms = statistics.pstdev(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    throughput = batch_size / (mean_ms / 1000.0)

    return {
        "batch_size": batch_size,
        "latency_ms_mean": round(mean_ms, 3),
        "latency_ms_std": round(std_ms, 3),
        "latency_ms_p50": round(p50, 3),
        "latency_ms_p99": round(p99, 3),
        "throughput_samples_per_sec": round(throughput, 1),
    }


@torch.no_grad()
def evaluate_full_pipeline(
    runtime: DefenseRuntime,
    device: torch.device,
    batch_size: int = 8,
    image_size: int = 224,
    iterations: int = 200,
) -> dict:
    """Benchmark full defense pipeline (sanitize + heuristic + neural)."""
    dummy = torch.rand(batch_size, 3, image_size, image_size, device=device)

    # Warmup
    for _ in range(20):
        runtime.full_defense(dummy)

    latencies = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runtime.full_defense(dummy)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(latencies)
    throughput = batch_size / (mean_ms / 1000.0)

    return {
        "pipeline_latency_ms_mean": round(mean_ms, 3),
        "pipeline_throughput_samples_per_sec": round(throughput, 1),
    }


def run_eval(checkpoint_path: str, num_samples: int = 5000, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchDetectorNet(in_channels=3).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())

    # Eval dataset (different seed than training)
    eval_ds = SyntheticPatchDataset(
        num_samples=num_samples,
        image_size=224,
        patch_ratio_range=(0.01, 0.20),
        attack_prob=0.5,
        seed=999,
    )
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"[EVAL] checkpoint={checkpoint_path}")
    print(f"[EVAL] model_params={param_count:,}")
    print(f"[EVAL] eval_samples={num_samples}")
    print()

    # 1. Accuracy
    acc_metrics = evaluate_accuracy(model, eval_loader, device)
    print(f"[ACCURACY] acc={acc_metrics['accuracy']:.4f} "
          f"prec={acc_metrics['precision']:.4f} "
          f"rec={acc_metrics['recall']:.4f} "
          f"f1={acc_metrics['f1']:.4f}")
    print(f"  TP={acc_metrics['tp']} FP={acc_metrics['fp']} "
          f"TN={acc_metrics['tn']} FN={acc_metrics['fn']}")

    # 2. Model-only latency
    lat_metrics = evaluate_latency(model, device, batch_size=8)
    print(f"\n[LATENCY] mean={lat_metrics['latency_ms_mean']:.2f}ms "
          f"p50={lat_metrics['latency_ms_p50']:.2f}ms "
          f"p99={lat_metrics['latency_ms_p99']:.2f}ms "
          f"throughput={lat_metrics['throughput_samples_per_sec']:.0f} samples/s")

    # 3. Full pipeline latency
    runtime = DefenseRuntime(backend="cuda")
    runtime.load_model(checkpoint_path)
    pipe_metrics = evaluate_full_pipeline(runtime, device, batch_size=8)
    print(f"\n[PIPELINE] mean={pipe_metrics['pipeline_latency_ms_mean']:.2f}ms "
          f"throughput={pipe_metrics['pipeline_throughput_samples_per_sec']:.0f} samples/s")

    # Combine report
    gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    report = {
        "checkpoint": str(checkpoint_path),
        "model_params": param_count,
        "eval_samples": num_samples,
        "gpu": gpu_name,
        **acc_metrics,
        **lat_metrics,
        **pipe_metrics,
    }

    # Save report
    report_dir = Path("/mnt/artifacts-datai/reports/DEF-roboticattack")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "eval_report.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(f"\n[SAVED] {report_file}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_eval(args.checkpoint, args.num_samples, args.batch_size)


if __name__ == "__main__":
    main()
