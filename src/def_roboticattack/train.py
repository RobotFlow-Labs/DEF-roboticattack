"""Training script for PatchDetectorNet defense model.

Usage:
    python -m def_roboticattack.train --config configs/paper.toml
    python -m def_roboticattack.train --config configs/debug.toml --max-steps 5
    python -m def_roboticattack.train --config configs/paper.toml --resume /path/to/best.pth
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import tomllib
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from def_roboticattack.data.synthetic import SyntheticPatchDataset
from def_roboticattack.models.patch_detector import PatchDetectorNet

REQUIRED_CONFIG_SECTIONS = ("training", "model", "data", "checkpoint", "early_stopping", "logging")


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]


class CheckpointManager:
    def __init__(
        self, save_dir: str, keep_top_k: int = 2, metric: str = "val_loss", mode: str = "min"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = max(1, keep_top_k)  # M4: ensure at least 1
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int):
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)

        best_val, best_path = self.history[0]
        best_dest = self.save_dir / "best.pth"
        shutil.copy2(best_path, best_dest)
        return best_dest


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        improved = (
            (metric < self.best - self.min_delta)
            if self.mode == "min"
            else (metric > self.best + self.min_delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def load_config(path: str) -> dict:
    """Load and validate TOML config."""
    with open(path, "rb") as f:
        config = tomllib.load(f)
    # M8: validate required sections
    for section in REQUIRED_CONFIG_SECTIONS:
        if section not in config:
            print(f"[ERROR] Config {path} missing required section: [{section}]")
            sys.exit(1)
    return config


def find_batch_size(
    model: nn.Module, device: torch.device, image_size: int, target_util: float = 0.65
) -> int:
    """Auto-detect optimal batch size targeting GPU VRAM utilization.

    Simulates a full training step (forward + backward) to account for grad memory.
    """
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_bytes = target_util * total_mem
    max_bs = 512

    def _try_bs(bs: int) -> bool:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            dummy = torch.rand(bs, 3, image_size, image_size, device=device)
            labels = torch.rand(bs, 1, device=device)
            out = model(dummy)
            loss = criterion(out, labels)
            loss.backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device)
            return peak <= target_bytes
        except RuntimeError:
            model.zero_grad(set_to_none=True)
            return False
        finally:
            # M9: always clean up
            torch.cuda.empty_cache()

    bs = 8
    best_bs = bs
    while bs <= max_bs:
        if _try_bs(bs):
            best_bs = bs
            bs *= 2
        else:
            break

    lo, hi = best_bs, min(bs, max_bs)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if _try_bs(mid):
            lo = mid
        else:
            hi = mid

    torch.cuda.empty_cache()
    return lo


def train(config: dict, resume: str | None = None, max_steps: int | None = None):
    tc = config["training"]
    mc = config["model"]
    dc = config["data"]
    cc = config["checkpoint"]
    lc = config["logging"]

    seed = tc["seed"]
    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device_type = "cuda" if use_cuda else "cpu"  # M1: device-aware amp
    model = PatchDetectorNet(in_channels=mc["in_channels"]).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    image_size = mc["image_size"]
    if tc["batch_size"] == "auto":
        if not use_cuda:
            batch_size = 32
        else:
            batch_size = find_batch_size(model, device, image_size, target_util=0.65)
    else:
        batch_size = int(tc["batch_size"])

    # H5: gradient accumulation
    grad_accum_steps = tc.get("gradient_accumulation", 1)

    train_ds = SyntheticPatchDataset(
        num_samples=dc["num_train_samples"],
        image_size=image_size,
        patch_ratio_range=(dc["patch_ratio_min"], dc["patch_ratio_max"]),
        attack_prob=dc["attack_prob"],
        seed=seed,
    )
    val_ds = SyntheticPatchDataset(
        num_samples=dc["num_val_samples"],
        image_size=image_size,
        patch_ratio_range=(dc["patch_ratio_min"], dc["patch_ratio_max"]),
        attack_prob=dc["attack_prob"],
        seed=seed + 1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=dc["num_workers"], pin_memory=dc["pin_memory"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=dc["num_workers"], pin_memory=dc["pin_memory"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"]
    )

    total_steps = tc["epochs"] * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, tc["warmup_steps"], total_steps)
    criterion = nn.BCEWithLogitsLoss()
    use_amp = use_cuda and tc["precision"] == "fp16"
    scaler = GradScaler(device_type, enabled=use_amp)  # M1: device-aware

    ckpt_mgr = CheckpointManager(
        save_dir=cc["save_dir"], keep_top_k=cc["keep_top_k"],
        metric=cc["metric"], mode=cc["mode"],
    )

    es_cfg = config["early_stopping"]
    early_stop = (
        EarlyStopping(patience=es_cfg["patience"], min_delta=es_cfg["min_delta"])
        if es_cfg["enabled"]
        else None
    )

    log_dir = Path(lc["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = log_dir / "metrics.jsonl"

    start_epoch = 0
    global_step = 0
    if resume:
        ckpt = torch.load(
            resume, map_location=device, weights_only=False
        )  # needs optimizer/scheduler state
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]
        print(f"[RESUME] from epoch={start_epoch}, step={global_step}")

    gpu_name = torch.cuda.get_device_name(device) if use_cuda else "CPU"
    gpu_count = torch.cuda.device_count() if use_cuda else 0
    print(f"[BATCH] batch_size={batch_size} (grad_accum={grad_accum_steps})")
    print(f"[GPU] {gpu_count}x {gpu_name}")
    print(f"[DATA] train={len(train_ds)} samples, val={len(val_ds)} samples")
    print(f"[MODEL] {param_count:,} parameters")
    print(f"[TRAIN] {tc['epochs']} epochs, lr={tc['learning_rate']}, optimizer={tc['optimizer']}")
    print(f"[CKPT] save every {cc['save_every_n_steps']} steps, keep best {cc['keep_top_k']}")

    t_start = time.time()
    last_ckpt_step = -1  # M13: track last checkpoint step to avoid duplicates

    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            if max_steps is not None and global_step >= max_steps:
                break

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

            with autocast(device_type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
                # H5: scale loss for gradient accumulation
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            if torch.isnan(loss):
                print("[FATAL] Loss is NaN — stopping training")
                print("[FIX] Reduce lr by 10x, check data for corrupt samples")
                return

            scaler.scale(loss).backward()

            # H5: only step optimizer every grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                batch_loss = loss.item() * grad_accum_steps
                epoch_loss += batch_loss

                preds = (torch.sigmoid(logits) > 0.5).float()
                epoch_correct += (preds == labels).sum().item()
                epoch_total += labels.shape[0]

                if global_step % lc["log_every_n_steps"] == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"  step={global_step} loss={batch_loss:.4f} lr={lr:.2e}")

                if global_step % cc["save_every_n_steps"] == 0:
                    val_loss = _validate(
                        model, val_loader, criterion, device, device_type, use_amp
                    )
                    state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "val_loss": val_loss,
                    }
                    ckpt_mgr.save(state, val_loss, global_step)
                    last_ckpt_step = global_step
                    model.train()

        if max_steps is not None and global_step >= max_steps:
            break

        # Epoch-level validation + checkpoint (skip if we just checkpointed at this step)
        val_loss = _validate(model, val_loader, criterion, device, device_type, use_amp)
        train_acc = epoch_correct / max(1, epoch_total)
        train_loss_avg = epoch_loss / max(1, len(train_loader) // grad_accum_steps)
        elapsed = time.time() - t_start

        log_entry = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss_avg,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
        }
        with open(metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(
            f"[EPOCH {epoch}/{tc['epochs']}] "
            f"train_loss={train_loss_avg:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} time={elapsed:.1f}s"
        )

        # M13: avoid duplicate checkpoint if step-level just saved
        if global_step != last_ckpt_step:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "step": global_step,
                "val_loss": val_loss,
            }
            ckpt_mgr.save(state, val_loss, global_step)
            last_ckpt_step = global_step

        if early_stop and early_stop.step(val_loss):
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs. Stopping.")
            break

    total_time = time.time() - t_start
    print(f"[DONE] Training complete in {total_time:.1f}s, {global_step} steps")
    print(f"[CKPT] Best checkpoint: {ckpt_mgr.save_dir / 'best.pth'}")


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    device_type: str,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)
        with autocast(device_type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * labels.shape[0]
        n += labels.shape[0]
    return total_loss / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps (smoke test)")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
