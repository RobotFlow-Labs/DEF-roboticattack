from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: str
    split: str
    batch_size: int


def build_dataset_spec(
    name: str,
    root: str,
    split: str = "train",
    batch_size: int = 8,
) -> DatasetSpec:
    return DatasetSpec(name=name, root=root, split=split, batch_size=batch_size)


def maybe_openvla_compat_name(name: str) -> str:
    if name in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
        return f"{name}_no_noops"
    return name


def describe_dataset(name: str) -> str:
    mapping = {
        "bridge_orig": "BridgeData V2 original split",
        "libero_spatial": "LIBERO Spatial",
        "libero_object": "LIBERO Object",
        "libero_goal": "LIBERO Goal",
        "libero_10": "LIBERO 10-task",
    }
    return mapping.get(name, "Unknown dataset")
