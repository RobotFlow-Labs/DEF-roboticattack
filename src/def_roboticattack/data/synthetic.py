"""Synthetic adversarial patch dataset for defense model training.

Generates clean images + adversarial-patched images for binary classification.
Simulates UADA/UPA/TMA attack patterns from the paper:
- Random patch sizes (1-20% of image area)
- Random positions
- Various patch textures (noise, gradient, natural-looking)

Uses per-index seeding for reproducibility across DataLoader workers.
"""
from __future__ import annotations

import math
import random

import torch
from torch.utils.data import Dataset


class SyntheticPatchDataset(Dataset):
    """Generate synthetic clean/patched image pairs for defense training.

    Label 0 = clean, Label 1 = adversarial patch present.
    Uses per-sample deterministic seeding so results are identical
    regardless of DataLoader num_workers.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        image_size: int = 224,
        patch_ratio_range: tuple[float, float] = (0.01, 0.20),
        attack_prob: float = 0.5,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.patch_ratio_range = patch_ratio_range
        self.attack_prob = attack_prob
        self.base_seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def _make_rng(self, idx: int) -> tuple[random.Random, torch.Generator]:
        """Create per-sample RNG pair from index — reproducible across workers."""
        sample_seed = hash((self.base_seed, idx)) & 0x7FFFFFFF
        rng = random.Random(sample_seed)
        gen = torch.Generator().manual_seed(sample_seed)
        return rng, gen

    def _make_clean_image(self, rng: random.Random, gen: torch.Generator) -> torch.Tensor:
        """Generate a plausible clean image (structured noise simulating robot workspace)."""
        img = torch.rand(3, self.image_size, self.image_size, generator=gen)
        x = torch.linspace(0, 1, self.image_size)
        y = torch.linspace(0, 1, self.image_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        angle = rng.uniform(0, 2 * math.pi)
        gradient = (grid_x * math.cos(angle) + grid_y * math.sin(angle)).unsqueeze(0)
        img = 0.6 * img + 0.4 * gradient.expand(3, -1, -1)
        return img.clamp(0, 1)

    def _make_patch(self, rng: random.Random, gen: torch.Generator, ph: int, pw: int) -> torch.Tensor:
        """Generate adversarial-like patch with high-frequency content."""
        style = rng.choice(["noise", "gradient", "checkerboard", "perturbed"])
        if style == "noise":
            patch = torch.rand(3, ph, pw, generator=gen)
        elif style == "gradient":
            x = torch.linspace(0, 1, pw).unsqueeze(0).expand(ph, -1)
            patch = x.unsqueeze(0).expand(3, -1, -1) * torch.rand(3, 1, 1, generator=gen)
        elif style == "checkerboard":
            check_size = max(2, min(ph, pw) // 4)
            rows = torch.arange(ph) // check_size
            cols = torch.arange(pw) // check_size
            checker = ((rows.unsqueeze(1) + cols.unsqueeze(0)) % 2).float()
            patch = checker.unsqueeze(0).expand(3, -1, -1)
            patch = patch * torch.rand(3, 1, 1, generator=gen)
        else:
            base = torch.rand(3, 1, 1, generator=gen).expand(3, ph, pw)
            noise = 0.3 * torch.rand(3, ph, pw, generator=gen)
            patch = (base + noise).clamp(0, 1)
        return patch.clamp(0, 1)

    def __getitem__(self, idx: int) -> dict:
        rng, gen = self._make_rng(idx)
        img = self._make_clean_image(rng, gen)
        is_attack = rng.random() < self.attack_prob

        if is_attack:
            ratio = rng.uniform(*self.patch_ratio_range)
            patch_area = ratio * self.image_size * self.image_size
            patch_side = int(math.sqrt(patch_area))
            patch_side = max(4, min(patch_side, self.image_size - 4))

            patch = self._make_patch(rng, gen, patch_side, patch_side)
            pos_y = rng.randint(0, self.image_size - patch_side)
            pos_x = rng.randint(0, self.image_size - patch_side)

            alpha = rng.uniform(0.7, 1.0)
            img[:, pos_y : pos_y + patch_side, pos_x : pos_x + patch_side] = (
                alpha * patch
                + (1 - alpha) * img[:, pos_y : pos_y + patch_side, pos_x : pos_x + patch_side]
            )
            label = 1
        else:
            label = 0

        return {"image": img, "label": torch.tensor(label, dtype=torch.float32)}
