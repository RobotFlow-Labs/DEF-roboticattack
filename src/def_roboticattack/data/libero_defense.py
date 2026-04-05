"""LIBERO-based defense dataset for adversarial patch detection training.

Loads real VLA manipulation frames from LIBERO and applies adversarial patches
to create a realistic binary classification dataset (clean vs patched).
Uses actual robot workspace images instead of synthetic noise.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


LIBERO_ROOT = Path("/mnt/forge-data/datasets/lerobot--libero")
FRAMES_DIR = LIBERO_ROOT / "extracted_frames" / "observation.images.image"


def _discover_frames(frames_dir: Path, max_frames: int = 0) -> list[Path]:
    """Discover all JPEG frames under the LIBERO extracted_frames directory."""
    frames = sorted(frames_dir.rglob("*.jpg"))
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


class LIBERODefenseDataset(Dataset):
    """Real LIBERO frames + adversarial patch augmentation for defense training.

    Label 0 = clean real frame, Label 1 = adversarial patch applied.
    """

    def __init__(
        self,
        frames_dir: Path = FRAMES_DIR,
        image_size: int = 224,
        patch_ratio_range: tuple[float, float] = (0.01, 0.20),
        attack_prob: float = 0.5,
        max_frames: int = 0,
        seed: int = 42,
    ):
        self.frames = _discover_frames(frames_dir, max_frames)
        if not self.frames:
            raise FileNotFoundError(f"No .jpg frames found in {frames_dir}")

        self.image_size = image_size
        self.patch_ratio_range = patch_ratio_range
        self.attack_prob = attack_prob
        self.base_seed = seed

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.frames)

    def _make_rng(self, idx: int) -> random.Random:
        return random.Random(hash((self.base_seed, idx)) & 0x7FFFFFFF)

    def _make_patch(self, rng: random.Random, ph: int, pw: int) -> torch.Tensor:
        """Generate adversarial-like patch with high-frequency content."""
        gen = torch.Generator().manual_seed(rng.randint(0, 2**31))
        style = rng.choice(["noise", "gradient", "checkerboard", "perturbed", "stripe"])

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
            patch = checker.unsqueeze(0).expand(3, -1, -1) * torch.rand(3, 1, 1, generator=gen)
        elif style == "stripe":
            # Vertical stripes — common in adversarial patches
            stripe_w = max(2, pw // 6)
            cols = torch.arange(pw) // stripe_w
            stripes = (cols % 2).float().unsqueeze(0).expand(ph, -1)
            patch = stripes.unsqueeze(0).expand(3, -1, -1) * torch.rand(3, 1, 1, generator=gen)
        else:
            base = torch.rand(3, 1, 1, generator=gen).expand(3, ph, pw).clone()
            noise = 0.3 * torch.rand(3, ph, pw, generator=gen)
            patch = (base + noise).clamp(0, 1)

        return patch.clamp(0, 1)

    def __getitem__(self, idx: int) -> dict:
        rng = self._make_rng(idx)

        # Load real LIBERO frame
        img_path = self.frames[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)  # [3, H, W] in [0, 1]

        is_attack = rng.random() < self.attack_prob

        if is_attack:
            ratio = rng.uniform(*self.patch_ratio_range)
            patch_area = ratio * self.image_size * self.image_size
            # Allow rectangular patches (M10: not just square)
            aspect = rng.uniform(0.5, 2.0)
            ph = int(math.sqrt(patch_area * aspect))
            pw = int(math.sqrt(patch_area / aspect))
            ph = max(4, min(ph, self.image_size - 4))
            pw = max(4, min(pw, self.image_size - 4))

            patch = self._make_patch(rng, ph, pw)
            pos_y = rng.randint(0, self.image_size - ph)
            pos_x = rng.randint(0, self.image_size - pw)

            alpha = rng.uniform(0.7, 1.0)
            region = img_tensor[:, pos_y : pos_y + ph, pos_x : pos_x + pw]
            img_tensor[:, pos_y : pos_y + ph, pos_x : pos_x + pw] = (
                alpha * patch + (1 - alpha) * region
            )
            label = 1
        else:
            label = 0

        return {"image": img_tensor, "label": torch.tensor(label, dtype=torch.float32)}
