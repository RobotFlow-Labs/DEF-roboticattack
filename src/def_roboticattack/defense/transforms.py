"""Input-domain sanitization transforms for VLA adversarial patch defense."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def clamp_patch_intensity(images: torch.Tensor, percentile: float = 99.5) -> torch.Tensor:
    """Clamp extreme per-image activations that are common in patch artifacts."""
    if images.ndim != 4:
        raise ValueError("Expected images tensor with shape [B, C, H, W]")

    b = images.shape[0]
    flat = images.view(b, -1)
    hi = torch.quantile(flat, percentile / 100.0, dim=1, keepdim=True).view(b, 1, 1, 1)
    lo = torch.quantile(flat, (100.0 - percentile) / 100.0, dim=1, keepdim=True).view(b, 1, 1, 1)
    return images.clamp(min=lo, max=hi)


def gaussian_blur_3x3(images: torch.Tensor, blur_strength: float = 0.15) -> torch.Tensor:
    """Low-cost smoothing to suppress high-frequency patch edges."""
    if images.ndim != 4:
        raise ValueError("Expected images tensor with shape [B, C, H, W]")

    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=images.device,
        dtype=images.dtype,
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)
    kernel = kernel.repeat(images.shape[1], 1, 1, 1)

    smoothed = F.conv2d(images, kernel, padding=1, groups=images.shape[1])
    return (1.0 - blur_strength) * images + blur_strength * smoothed
