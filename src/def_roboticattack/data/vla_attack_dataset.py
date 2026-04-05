"""VLA adversarial attack dataset for defense training.

Combines real images from LIBERO + COCO with UADA/UPA/TMA-style adversarial
patches applied via geometric transforms. This is the production training
dataset for the defense model.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from def_roboticattack.attacks.patch_gen import AdversarialPatchGenerator


LIBERO_FRAMES = Path("/mnt/forge-data/datasets/lerobot--libero/extracted_frames/observation.images.image")
COCO_VAL = Path("/mnt/forge-data/datasets/coco/val2017")


def _discover_images(dirs: list[Path], extensions: tuple = (".jpg", ".jpeg", ".png"), max_per_dir: int = 0) -> list[Path]:
    """Discover image files from multiple directories."""
    images = []
    for d in dirs:
        if not d.exists():
            continue
        found = sorted(f for f in d.rglob("*") if f.suffix.lower() in extensions)
        if max_per_dir > 0:
            found = found[:max_per_dir]
        images.extend(found)
    return images


class VLAAttackDataset(Dataset):
    """Real VLA images + multi-strategy adversarial patches for defense training.

    Sources:
    - LIBERO robot manipulation frames (256x256)
    - COCO val2017 natural images (diverse backgrounds)

    Attack types (matching paper):
    - UADA: Universal adversarial patch with geometric transforms
    - UPA: Untargeted high-frequency patch
    - TMA: Targeted manipulation with directional gradients
    - PGD: gradient-optimized evasion patches (if model provided)

    Label 0 = clean, Label 1 = attacked
    """

    ATTACK_TYPES = ["uada", "upa", "tma"]

    def __init__(
        self,
        image_size: int = 224,
        patch_ratio_range: tuple[float, float] = (0.01, 0.20),
        attack_prob: float = 0.5,
        use_geometry: bool = True,
        max_libero_frames: int = 50000,
        max_coco_images: int = 5000,
        seed: int = 42,
    ):
        # Discover real images
        libero_images = _discover_images([LIBERO_FRAMES], max_per_dir=max_libero_frames)
        coco_images = _discover_images([COCO_VAL], max_per_dir=max_coco_images)
        self.images = libero_images + coco_images
        if not self.images:
            raise FileNotFoundError("No images found in LIBERO or COCO directories")

        self.image_size = image_size
        self.patch_ratio_range = patch_ratio_range
        self.attack_prob = attack_prob
        self.base_seed = seed

        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self.patch_gen = AdversarialPatchGenerator(device="cpu", use_geometry=use_geometry)

    def __len__(self) -> int:
        return len(self.images)

    def _make_rng(self, idx: int) -> random.Random:
        return random.Random(hash((self.base_seed, idx)) & 0x7FFFFFFF)

    def __getitem__(self, idx: int) -> dict:
        rng = self._make_rng(idx)

        # Load real image
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        is_attack = rng.random() < self.attack_prob

        if is_attack:
            # Random patch size from paper's distribution
            ratio = rng.uniform(*self.patch_ratio_range)
            patch_area = ratio * self.image_size * self.image_size
            # Allow rectangular patches
            aspect = rng.uniform(0.6, 1.67)
            ph = int(math.sqrt(patch_area * aspect))
            pw = int(math.sqrt(patch_area / aspect))
            ph = max(4, min(ph, self.image_size - 4))
            pw = max(4, min(pw, self.image_size - 4))

            # Random attack type
            attack_type = rng.choice(self.ATTACK_TYPES)
            if attack_type == "uada":
                patch = self.patch_gen.generate_uada_patch(rng, ph, pw)
            elif attack_type == "upa":
                patch = self.patch_gen.generate_upa_patch(rng, ph, pw)
            else:
                patch = self.patch_gen.generate_tma_patch(rng, ph, pw)

            # Apply patch with geometric transform
            img_tensor = self.patch_gen.apply_patch_to_image(img_tensor, patch, rng)
            label = 1
        else:
            label = 0

        return {"image": img_tensor, "label": torch.tensor(label, dtype=torch.float32)}
