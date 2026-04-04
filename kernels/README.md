# Custom Kernels — DEF-roboticattack

This folder contains ANIMA-owned optimization tracks for defense runtime acceleration.

## Structure
- `cuda/fused_patch_blend.cu` — CUDA scaffold for fused blend path
- `mlx/fused_patch_blend.metal` — MLX/Metal scaffold for backend parity
- `triton/fused_patch_blend.py` — Triton scaffold for fast iteration

## Optimization Targets
1. Fused patch blend + normalization
2. Fused edge-gradient/anomaly feature extraction
3. INT8/FP16 inference-time defense kernels
4. Memory-optimized image batch path for edge devices

## IP Note
Every optimized kernel in this folder is ANIMA proprietary IP.
