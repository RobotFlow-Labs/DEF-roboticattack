# DEF-roboticattack

ANIMA Wave-8 defense module scaffold for adversarial robustness of VLA systems.

## Highlights
- Python 3.11+
- Build backend: `hatchling`
- Runtime backend selection via `ANIMA_BACKEND=auto|cuda|mlx`
- Initial defense runtime: input sanitization + patch anomaly scoring
- Kernel/IP stubs for CUDA, MLX, and Triton

## Layout
- `src/def_roboticattack/` core package
- `kernels/` kernel optimization tracks
- `benchmarks/` benchmark harness and results
- `tasks/` executable task slices
- `repositories/roboticAttack/` upstream attack code reference
- `papers/` local paper PDFs

## Quick Start (uv)
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### Backend Info
```bash
ANIMA_BACKEND=auto def-roboticattack backend-info
```

### Dry Run
```bash
ANIMA_BACKEND=auto def-roboticattack dry-run --batch-size 4 --height 224 --width 224
```

### Benchmark
```bash
python benchmarks/benchmark_patch_guard.py --backend auto --iterations 200 --batch-size 8
```

