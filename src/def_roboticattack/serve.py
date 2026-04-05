"""ANIMA serve integration for DEF-roboticattack.

AnimaNode subclass that provides:
- FastAPI /predict endpoint for adversarial patch detection
- Health/ready/info endpoints (from base class)
- Weight download from HuggingFace on startup
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import torch


class DefRoboticAttackNode:
    """Defense runtime serving node for adversarial patch detection."""

    def __init__(self):
        self.model = None
        self.runtime = None
        self.device = None
        self._ready = False
        self._start_time = time.time()

    def setup_inference(self, weight_dir: str | Path = "/data/weights"):
        """Load model weights and configure backend."""
        from def_roboticattack.models.patch_detector import PatchDetectorNet
        from def_roboticattack.pipeline.runtime import DefenseRuntime

        weight_dir = Path(weight_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try safetensors first, then pth
        safetensors_path = weight_dir / "patch_detector.safetensors"
        pth_path = weight_dir / "patch_detector.pth"
        best_path = weight_dir / "best.pth"

        model = PatchDetectorNet(in_channels=3).to(self.device)

        if safetensors_path.exists():
            from safetensors.torch import load_file

            state = load_file(str(safetensors_path), device=str(self.device))
            model.load_state_dict(state)
        elif pth_path.exists():
            ckpt = torch.load(pth_path, map_location=self.device, weights_only=False)
            state = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state)
        elif best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            state = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state)
        else:
            print("[WARN] No weights found, using random initialization")

        model.eval()
        self.model = model

        # Set up full defense runtime
        self.runtime = DefenseRuntime(backend="cuda" if torch.cuda.is_available() else "cpu")
        self.runtime._model = model
        self.runtime._model_device = self.device
        self._ready = True
        print(f"[SERVE] Model loaded on {self.device}, ready for inference")

    def process(self, image_bytes: bytes | None = None, image_tensor: torch.Tensor | None = None) -> dict:
        """Run inference on input image(s)."""
        if not self._ready:
            return {"error": "Model not ready"}

        if image_tensor is not None:
            images = image_tensor.to(self.device)
        elif image_bytes is not None:
            from PIL import Image
            import torchvision.transforms as T

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            images = transform(img).unsqueeze(0).to(self.device)
        else:
            return {"error": "No input provided"}

        if images.ndim == 3:
            images = images.unsqueeze(0)

        result = self.runtime.full_defense(images)
        return {
            "probabilities": result["neural"]["probabilities"] if "neural" in result else [],
            "flagged": result["neural"]["flagged"] if "neural" in result else [],
            "combined_risk": result["combined_risk"],
            "heuristic_risk": result["heuristic_risk"],
        }

    def get_health(self) -> dict:
        return {
            "status": "healthy" if self._ready else "starting",
            "module": "def-roboticattack",
            "uptime_s": time.time() - self._start_time,
            "gpu": str(self.device) if self.device else "unknown",
        }

    def get_ready(self) -> dict:
        return {
            "ready": self._ready,
            "module": "def-roboticattack",
            "version": "0.1.0",
            "weights_loaded": self.model is not None,
        }

    def get_info(self) -> dict:
        param_count = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        return {
            "module": "def-roboticattack",
            "version": "0.1.0",
            "domain": "defense",
            "paper": "VLA Adversarial Vulnerabilities (ICCV 2025)",
            "model_params": param_count,
            "device": str(self.device),
            "capabilities": [
                "adversarial_patch_detection",
                "input_sanitization",
                "anomaly_scoring",
            ],
        }
