from dataclasses import dataclass


@dataclass(frozen=True)
class DefenseConfig:
    """Runtime knobs for baseline patch-defense scaffolding."""

    edge_threshold: float = 0.15
    clamp_percentile: float = 99.5
    blur_strength: float = 0.15
