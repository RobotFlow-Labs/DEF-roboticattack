from __future__ import annotations

import numpy as np

from def_roboticattack.config import DefenseConfig
from def_roboticattack.defense.detector import DetectionResult, PatchAnomalyDetector
from def_roboticattack.defense.transforms import clamp_patch_intensity, gaussian_blur_3x3
from def_roboticattack.device import BackendInfo, resolve_backend


class DefenseRuntime:
    def __init__(self, config: DefenseConfig | None = None, backend: str | None = None):
        self.config = config or DefenseConfig()
        self.backend_info: BackendInfo = resolve_backend(backend)
        self.detector = PatchAnomalyDetector(threshold_z=self.config.anomaly_threshold_z)

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

    @staticmethod
    def aggregate_risk(detection: DetectionResult) -> float:
        if not detection.zscore:
            return 0.0
        return float(max(detection.zscore))
