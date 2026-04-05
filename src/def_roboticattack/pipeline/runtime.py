"""Defense runtime — composed pipeline for VLA adversarial patch defense.

Integrates:
- PatchDetectorNet (trained CNN for patch detection)
- Input sanitization transforms (clamp + blur)
- Heuristic anomaly scoring (Sobel edge magnitude, fixed threshold)
- CUDA-accelerated ops when available
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from def_roboticattack.config import DefenseConfig
from def_roboticattack.defense.detector import DetectionResult, PatchAnomalyDetector
from def_roboticattack.defense.transforms import clamp_patch_intensity, gaussian_blur_3x3
from def_roboticattack.device import BackendInfo, resolve_backend


class DefenseRuntime:
    def __init__(self, config: DefenseConfig | None = None, backend: str | None = None):
        self.config = config or DefenseConfig()
        self.backend_info: BackendInfo = resolve_backend(backend)
        self.detector = PatchAnomalyDetector(edge_threshold=self.config.edge_threshold)
        self._model = None
        self._model_device = None

    def load_model(self, checkpoint_path: str | Path | None = None):
        """Load trained PatchDetectorNet for neural patch detection."""
        import torch

        from def_roboticattack.models.patch_detector import PatchDetectorNet

        device = torch.device(self.backend_info.torch_device)
        model = PatchDetectorNet(in_channels=3).to(device)

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            state = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state)

        model.eval()
        self._model = model
        self._model_device = device

    def sanitize_and_score(self, images):
        """Run baseline sanitization + anomaly scoring."""
        if hasattr(images, "ndim") and not isinstance(images, np.ndarray):
            sanitized = clamp_patch_intensity(images, self.config.clamp_percentile)
            sanitized = gaussian_blur_3x3(sanitized, self.config.blur_strength)
            detection = self.detector.score_torch(sanitized)
            return sanitized, detection

        np_images = np.asarray(images)
        detection = self.detector.score_numpy(np_images)
        return np_images, detection

    def detect_patches(self, images) -> dict:
        """Run trained neural patch detector. Returns probabilities and flags."""
        import torch

        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model(checkpoint_path) first.")

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        images = images.to(self._model_device)
        if images.ndim == 3:
            images = images.unsqueeze(0)

        with torch.no_grad():
            probs = self._model.predict_proba(images).squeeze(-1)

        return {
            "probabilities": probs.cpu().tolist(),
            "flagged": (probs > 0.5).cpu().tolist(),
            "max_prob": probs.max().item(),
        }

    def full_defense(self, images) -> dict:
        """Full defense pipeline: sanitize + heuristic score + neural detection."""
        sanitized, heuristic_detection = self.sanitize_and_score(images)
        heuristic_risk = self.aggregate_risk(heuristic_detection)

        result = {
            "sanitized": sanitized,
            "heuristic": heuristic_detection,
            "heuristic_risk": heuristic_risk,
        }

        if self._model is not None:
            neural_result = self.detect_patches(sanitized)
            result["neural"] = neural_result
            # Both heuristic_risk and neural max_prob are now in [0, 1]
            result["combined_risk"] = max(heuristic_risk, neural_result["max_prob"])
        else:
            result["combined_risk"] = heuristic_risk

        return result

    @staticmethod
    def aggregate_risk(detection: DetectionResult) -> float:
        """Return max score normalized to [0, 1] range."""
        if not detection.score:
            return 0.0
        # Edge magnitude scores are typically in [0, 0.5] range for natural images.
        # Normalize to [0, 1] with a soft saturation at 0.3.
        raw = float(max(detection.score))
        return min(1.0, raw / 0.3)
