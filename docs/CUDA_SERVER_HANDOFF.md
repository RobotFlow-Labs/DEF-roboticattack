# CUDA Server Handoff

## Goal
Move DEF-roboticattack scaffold to CUDA server and verify runtime and benchmark execution under Python 3.11.

## Preconditions
- NVIDIA driver installed (`nvidia-smi` works)
- Python `3.11` installed
- `uv` installed

## Steps
1. `bash scripts/cuda_server_setup.sh`
2. `bash scripts/cuda_smoke.sh`
3. `bash scripts/cuda_benchmark.sh`

## Expected Artifacts
- Virtual environment: `.venv/`
- Benchmark JSON: `benchmarks/results/patch_guard_cuda.json`

## Notes
- Current state is defense scaffold + baseline runtime.
- Full OpenVLA evaluation-time integration and optimized kernels remain next phase tasks.
