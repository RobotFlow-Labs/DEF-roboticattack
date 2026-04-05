"""Thin integration adapter for OpenVLA defense."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from def_roboticattack.pipeline.runtime import DefenseRuntime


class OpenVLAGuard:
    """Thin integration adapter that can be inserted before OpenVLA forward pass."""

    def __init__(self, backend: str = "auto"):
        self.runtime = DefenseRuntime(backend=backend)

    def _to_nchw(self, pixel_values: Any):
        try:
            import torch
            import torchvision.transforms as T

            if isinstance(pixel_values, torch.Tensor):
                if pixel_values.ndim != 4:
                    raise ValueError("Expected tensor pixel_values with shape [B, C, H, W]")
                return pixel_values

            if isinstance(pixel_values, Iterable):
                tensors = []
                to_tensor = T.ToTensor()
                for item in pixel_values:
                    tensors.append(to_tensor(item))
                return torch.stack(tensors, dim=0)
        except ImportError:
            pass  # torch/torchvision not available — fall through to numpy path

        arr = np.asarray(pixel_values)
        if arr.ndim == 4 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))
        if arr.ndim != 4:
            raise ValueError("Unsupported pixel_values format")
        return arr

    def sanitize(self, pixel_values: Any):
        images = self._to_nchw(pixel_values)
        sanitized, detection = self.runtime.sanitize_and_score(images)
        return sanitized, detection
