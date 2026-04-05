import importlib.util
import os
from dataclasses import dataclass
from typing import Literal

BackendName = Literal["cuda", "mlx", "cpu"]

# Prevent OpenMP duplicate library errors in heterogeneous Python stacks
# (e.g., numpy + torch both link against libiomp). Safe to set globally.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass(frozen=True)
class BackendInfo:
    name: BackendName
    torch_device: str
    supports_half: bool
    reason: str


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _mlx_available() -> bool:
    return importlib.util.find_spec("mlx") is not None


def resolve_backend(preferred: str | None = None) -> BackendInfo:
    raw = (preferred or os.getenv("ANIMA_BACKEND", "auto")).lower().strip()
    if raw not in {"auto", "cuda", "mlx", "cpu"}:
        raise ValueError(f"Unsupported backend '{raw}'. Expected auto|cuda|mlx|cpu")

    cuda_ok = _torch_cuda_available()
    mlx_ok = _mlx_available()

    if raw == "cuda":
        if not cuda_ok:
            raise RuntimeError("ANIMA_BACKEND=cuda requested but CUDA torch backend is unavailable")
        return BackendInfo("cuda", "cuda", True, "Requested CUDA and CUDA is available")

    if raw == "mlx":
        if not mlx_ok:
            raise RuntimeError("ANIMA_BACKEND=mlx requested but MLX is unavailable")
        return BackendInfo("mlx", "cpu", True, "Requested MLX and MLX is available")

    if raw == "cpu":
        return BackendInfo("cpu", "cpu", False, "Requested CPU")

    if cuda_ok:
        return BackendInfo("cuda", "cuda", True, "Auto-selected CUDA")
    if mlx_ok:
        return BackendInfo("mlx", "cpu", True, "Auto-selected MLX")
    return BackendInfo("cpu", "cpu", False, "Auto-selected CPU fallback")
