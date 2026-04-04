from dataclasses import dataclass


@dataclass(frozen=True)
class DefenseConfig:
    """Runtime knobs for baseline patch-defense scaffolding."""

    anomaly_threshold_z: float = 2.5
    clamp_percentile: float = 99.5
    blur_strength: float = 0.15
