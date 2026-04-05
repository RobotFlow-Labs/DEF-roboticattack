"""CUDA-accelerated defense operations.

Wraps compiled CUDA kernels from kernels/cuda/ with Python-friendly API
and CPU fallbacks for dual-compute compatibility.
"""
from __future__ import annotations

from pathlib import Path

import torch

_KERNELS = None


def _load_cuda_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS

    kernel_dir = Path(__file__).resolve().parents[3] / "kernels" / "cuda"
    so_files = list(kernel_dir.glob("roboticattack_cuda_kernels*.so"))
    if not so_files:
        return None

    import importlib.util

    spec = importlib.util.spec_from_file_location("roboticattack_cuda_kernels", so_files[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _KERNELS = mod
    return _KERNELS


def fused_patch_apply(
    image: torch.Tensor,
    patch: torch.Tensor,
    pos_y: int,
    pos_x: int,
) -> torch.Tensor:
    """Apply adversarial patch to image. CUDA-accelerated with CPU fallback.

    Args:
        image: [C, H, W] single image tensor
        patch: [C, pH, pW] patch tensor
        pos_y, pos_x: top-left position for patch placement
    """
    if image.is_cuda:
        kernels = _load_cuda_kernels()
        if kernels is not None:
            return kernels.fused_patch_apply(image, patch, pos_y, pos_x)

    # CPU fallback
    output = image.clone()
    pH, pW = patch.shape[1], patch.shape[2]
    output[:, pos_y : pos_y + pH, pos_x : pos_x + pW] = patch
    return output


def fused_patch_apply_batch(
    images: torch.Tensor,
    patch: torch.Tensor,
    positions: list[tuple[int, int]],
) -> torch.Tensor:
    """Batch patch application. Applies same patch at different positions per image.

    Args:
        images: [B, C, H, W]
        patch: [C, pH, pW]
        positions: list of (pos_y, pos_x) per image
    """
    B = images.shape[0]
    outputs = []
    for i in range(B):
        py, px = positions[i] if i < len(positions) else positions[-1]
        outputs.append(fused_patch_apply(images[i], patch, py, px))
    return torch.stack(outputs)


def fused_action_perturb(
    actions: torch.Tensor,
    grads: torch.Tensor,
    step_size: float,
    eps: float,
) -> torch.Tensor:
    """PGD-style action perturbation. CUDA-accelerated with CPU fallback.

    Used to simulate adversarial action-space attacks for defense training.

    Args:
        actions: [N, D] action vectors
        grads: [N, D] gradient vectors
        step_size: PGD step size
        eps: L-infinity constraint radius
    """
    if actions.is_cuda:
        kernels = _load_cuda_kernels()
        if kernels is not None:
            return kernels.fused_action_perturb(actions, grads, step_size, eps)

    # CPU fallback: FGSM-style sign step + projection
    sign = torch.sign(grads)
    perturbed = actions + step_size * sign
    delta = (perturbed - actions).clamp(-eps, eps)
    return actions + delta
