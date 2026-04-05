"""Heuristic patch anomaly detection via edge energy analysis.

Uses Sobel edge magnitude as a signal: adversarial patches create
abnormally sharp edges. Scoring uses a fixed reference threshold
(not per-batch statistics) so it works correctly at batch_size=1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    score: list[float]
    flagged: list[bool]


class PatchAnomalyDetector:
    def __init__(self, edge_threshold: float = 0.15):
        """Initialize detector with fixed edge magnitude threshold.

        Args:
            edge_threshold: absolute edge magnitude above which an image
                is flagged as potentially containing an adversarial patch.
                Determined empirically from clean VLA image distribution.
        """
        self.edge_threshold = edge_threshold

    def score_torch(self, images) -> DetectionResult:
        import torch
        import torch.nn.functional as F

        if images.ndim != 4:
            raise ValueError("Expected tensor shape [B, C, H, W]")

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=images.device,
            dtype=images.dtype,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(images, sobel_x.repeat(images.shape[1], 1, 1, 1), padding=1, groups=images.shape[1])
        gy = F.conv2d(images, sobel_y.repeat(images.shape[1], 1, 1, 1), padding=1, groups=images.shape[1])
        mag = torch.sqrt(gx * gx + gy * gy).mean(dim=(1, 2, 3))

        flagged = mag > self.edge_threshold

        return DetectionResult(
            score=mag.detach().cpu().tolist(),
            flagged=flagged.detach().cpu().tolist(),
        )

    def score_numpy(self, images: np.ndarray) -> DetectionResult:
        if images.ndim != 4:
            raise ValueError("Expected ndarray shape [B, C, H, W]")

        dx = np.abs(np.diff(images, axis=3, prepend=images[:, :, :, :1]))
        dy = np.abs(np.diff(images, axis=2, prepend=images[:, :, :1, :]))
        mag = np.sqrt(dx * dx + dy * dy).mean(axis=(1, 2, 3))

        flagged = mag > self.edge_threshold

        return DetectionResult(
            score=mag.astype(float).tolist(),
            flagged=flagged.astype(bool).tolist(),
        )
