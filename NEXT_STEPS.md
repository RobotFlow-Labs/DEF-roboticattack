# NEXT_STEPS — DEF-roboticattack
## Last Updated: 2026-04-04
## Status: READY_FOR_CUDA_SERVER — baseline defense scaffold operational
## MVP Readiness: 45%

### DONE
- [x] Clone upstream repo: https://github.com/William-wAng618/roboticAttack
- [x] Read paper (ICCV 2025 + arXiv copy)
- [x] Create device.py dual-compute abstraction (`ANIMA_BACKEND=auto|cuda|mlx|cpu`)
- [x] Break PRD into tasks/ (single executable tasks)
- [x] Identify kernel optimization targets
- [x] Scaffold CUDA/MLX/Triton kernel tracks
- [x] Scaffold baseline defense runtime (sanitize + detect)
- [x] Add CUDA server setup/smoke/benchmark scripts
- [x] Add CUDA handoff runbook

### TODO (next phase)
- [ ] Set up full OpenVLA training/eval environment and datasets on CUDA server
- [ ] Verify baseline runs on CUDA with real OpenVLA checkpoints
- [ ] Integrate defense runtime directly into upstream OpenVLA eval pipeline
- [ ] Build custom CUDA kernels for critical path
- [ ] Build MLX Metal equivalent kernels
- [ ] Benchmark: baseline vs optimized
- [ ] Integration test on Jetson Orin NX (if applicable)

### Blockers
- No code blockers in scaffold.
- External dependency blockers for full pipeline: model weights + datasets + server resources.

### Models/Datasets Needed
- OpenVLA checkpoints (base + LIBERO variants)
- BridgeData V2 / LIBERO preprocessed data
- See DOWNLOAD_MANIFEST.md Wave 8 section
