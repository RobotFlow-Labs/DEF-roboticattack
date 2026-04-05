from dataclasses import dataclass
from pathlib import Path


@dataclass
class DefenseConfig:
    """Runtime knobs for baseline patch-defense scaffolding.

    M14: Not frozen — allows runtime overrides and construction from TOML.
    """

    edge_threshold: float = 0.15
    clamp_percentile: float = 99.5
    blur_strength: float = 0.15

    @classmethod
    def from_toml(cls, path: str | Path) -> "DefenseConfig":
        """Load defense config from a TOML file's [defense] section if present."""
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        section = data.get("defense", {})
        return cls(
            edge_threshold=section.get("edge_threshold", cls.edge_threshold),
            clamp_percentile=section.get("clamp_percentile", cls.clamp_percentile),
            blur_strength=section.get("blur_strength", cls.blur_strength),
        )
