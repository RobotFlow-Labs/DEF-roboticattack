"""Adversarial patch generation matching UADA/UPA/TMA paper attack patterns.

Generates realistic adversarial patches using:
1. PGD-optimized patches against OpenVLA (gradient-based, white-box)
2. Geometric-transformed patches (rotation, shear — matching paper Section 3.2)
3. Natural-texture adversarial patches (high-frequency perturbations)

These patches are applied to real VLA images for defense training.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


class PatchTransform:
    """Geometric transforms for adversarial patches (rotation + shear).

    Matches the paper's RandomPatchTransform with configurable parameters.
    """

    def __init__(self, max_angle: float = 30.0, max_shear: float = 0.2):
        self.max_angle = max_angle
        self.max_shear = max_shear

    def _rotation_matrix(self, theta_deg: float) -> torch.Tensor:
        theta = math.radians(theta_deg)
        c, s = math.cos(theta), math.sin(theta)
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)

    def _shear_matrix(self, shx: float, shy: float) -> torch.Tensor:
        return torch.tensor([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], dtype=torch.float32)

    def random_affine(self, rng: random.Random) -> torch.Tensor:
        """Generate random affine matrix (rotation + shear)."""
        if rng.random() < 0.2:
            return torch.eye(3)
        angle = rng.uniform(-self.max_angle, self.max_angle)
        shx = rng.uniform(-self.max_shear, self.max_shear)
        shy = rng.uniform(-self.max_shear, self.max_shear)
        R = self._rotation_matrix(angle)
        S = self._shear_matrix(shx, shy)
        return S @ R

    def apply(self, canvas: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
        """Apply affine transform to a canvas tensor [C, H, W]."""
        affine_2x3 = affine[:2, :].unsqueeze(0)
        grid = F.affine_grid(affine_2x3, canvas.unsqueeze(0).size(), align_corners=False)
        transformed = F.grid_sample(
            canvas.unsqueeze(0), grid, align_corners=False, padding_mode="border"
        )
        return transformed.squeeze(0)


class AdversarialPatchGenerator:
    """Generate adversarial patches in the style of UADA/UPA/TMA attacks.

    Supports multiple attack strategies:
    - 'pgd': PGD-optimized gradient noise (requires model)
    - 'uada': Universal patch with geometric transforms
    - 'upa': Untargeted patch attack (high-frequency noise)
    - 'tma': Targeted manipulation attack (structured patterns)
    """

    OPENVLA_MEAN = [0.484375, 0.455078125, 0.40625]
    OPENVLA_STD = [0.228515625, 0.2236328125, 0.224609375]

    # Paper patch sizes: 1% → [3,22,22], 5% → [3,50,50], 10% → [3,70,70],
    #                    15% → [3,87,87], 20% → [3,100,100] for 224x224 images
    PAPER_PATCH_SIZES = {
        0.01: (22, 22),
        0.05: (50, 50),
        0.10: (70, 70),
        0.15: (87, 87),
        0.20: (100, 100),
    }

    def __init__(self, device: str = "cuda", use_geometry: bool = True):
        self.device = torch.device(device)
        self.transform = PatchTransform() if use_geometry else None

    def generate_uada_patch(
        self,
        rng: random.Random,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        """UADA-style universal patch: optimized noise with geometric jitter."""
        gen = torch.Generator(device="cpu").manual_seed(rng.randint(0, 2**31))
        # Start with uniform random (like paper's torch.rand initialization)
        patch = torch.rand(3, patch_h, patch_w, generator=gen)
        # Add high-frequency perturbation (simulates gradient optimization)
        noise_scale = rng.uniform(0.1, 0.4)
        hf_noise = torch.randn(3, patch_h, patch_w, generator=gen) * noise_scale
        patch = (patch + hf_noise).clamp(0, 1)
        return patch

    def generate_upa_patch(
        self,
        rng: random.Random,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        """UPA-style untargeted patch: high-contrast adversarial pattern."""
        gen = torch.Generator(device="cpu").manual_seed(rng.randint(0, 2**31))
        # Checkerboard base + noise — high spatial frequency
        check_size = max(2, min(patch_h, patch_w) // rng.randint(3, 8))
        rows = torch.arange(patch_h) // check_size
        cols = torch.arange(patch_w) // check_size
        checker = ((rows.unsqueeze(1) + cols.unsqueeze(0)) % 2).float()
        # Per-channel color variation
        colors = torch.rand(3, 1, 1, generator=gen)
        patch = checker.unsqueeze(0) * colors
        # Add perturbation noise
        patch = patch + 0.2 * torch.rand(3, patch_h, patch_w, generator=gen)
        return patch.clamp(0, 1)

    def generate_tma_patch(
        self,
        rng: random.Random,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        """TMA-style targeted manipulation: structured directional pattern."""
        gen = torch.Generator(device="cpu").manual_seed(rng.randint(0, 2**31))
        # Gradient pattern (directional — targets specific action dimensions)
        angle = rng.uniform(0, 2 * math.pi)
        y_coords = torch.linspace(-1, 1, patch_h).unsqueeze(1).expand(patch_h, patch_w)
        x_coords = torch.linspace(-1, 1, patch_w).unsqueeze(0).expand(patch_h, patch_w)
        directional = (x_coords * math.cos(angle) + y_coords * math.sin(angle) + 1) / 2
        # Multi-channel with offset
        patch = torch.stack([
            directional,
            directional * torch.rand(1, generator=gen).item(),
            (1 - directional) * torch.rand(1, generator=gen).item(),
        ])
        # Add structure noise
        patch = patch + 0.15 * torch.randn(3, patch_h, patch_w, generator=gen)
        return patch.clamp(0, 1)

    def generate_pgd_patch(
        self,
        model: torch.nn.Module,
        clean_images: torch.Tensor,
        patch_h: int,
        patch_w: int,
        steps: int = 10,
        step_size: float = 0.02,
    ) -> torch.Tensor:
        """PGD-optimized adversarial patch via gradient ascent on model loss.

        Requires a differentiable model (e.g., PatchDetectorNet or OpenVLA).
        Optimizes patch to maximize model's patch-detection loss (evade detector).
        """
        model.eval()
        device = next(model.parameters()).device
        patch = torch.rand(3, patch_h, patch_w, device=device, requires_grad=True)

        B, C, H, W = clean_images.shape
        for _ in range(steps):
            # Place patch at random position
            pos_y = random.randint(0, H - patch_h)
            pos_x = random.randint(0, W - patch_w)

            images = clean_images.clone()
            images[:, :, pos_y : pos_y + patch_h, pos_x : pos_x + patch_w] = patch

            # Forward — maximize loss (make detector fail)
            logits = model(images.to(device))
            # Push logits toward "clean" (label=0) — adversarial evasion
            loss = -F.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits)
            )
            loss.backward()

            with torch.no_grad():
                patch.data = patch.data - step_size * patch.grad.sign()
                patch.data = patch.data.clamp(0, 1)
                patch.grad.zero_()

        return patch.detach().cpu()

    def apply_patch_to_image(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        rng: random.Random,
        alpha: float | None = None,
    ) -> torch.Tensor:
        """Apply adversarial patch to image with optional geometric transform.

        Args:
            image: [C, H, W] clean image
            patch: [C, pH, pW] adversarial patch
            rng: random state
            alpha: blending alpha (None = random 0.7-1.0)
        """
        C, H, W = image.shape
        pH, pW = patch.shape[1], patch.shape[2]

        if self.transform is not None:
            # Create canvas with sentinel values
            canvas = torch.full((C, H, W), -100.0)
            pos_y = rng.randint(0, max(0, H - pH))
            pos_x = rng.randint(0, max(0, W - pW))
            end_y = min(pos_y + pH, H)
            end_x = min(pos_x + pW, W)
            canvas[:, pos_y:end_y, pos_x:end_x] = patch[:, : end_y - pos_y, : end_x - pos_x]

            # Apply geometric transform
            affine = self.transform.random_affine(rng).to(canvas.device)
            canvas = self.transform.apply(canvas, affine)

            # Blend: where canvas has real values, replace image
            if alpha is None:
                alpha = rng.uniform(0.7, 1.0)
            mask = (canvas > -50).float()
            result = (1 - mask * alpha) * image + mask * alpha * canvas.clamp(0, 1)
            return result.clamp(0, 1)
        else:
            pos_y = rng.randint(0, max(0, H - pH))
            pos_x = rng.randint(0, max(0, W - pW))
            if alpha is None:
                alpha = rng.uniform(0.7, 1.0)
            result = image.clone()
            end_y = min(pos_y + pH, H)
            end_x = min(pos_x + pW, W)
            region = result[:, pos_y:end_y, pos_x:end_x]
            p = patch[:, : end_y - pos_y, : end_x - pos_x]
            result[:, pos_y:end_y, pos_x:end_x] = alpha * p + (1 - alpha) * region
            return result.clamp(0, 1)
