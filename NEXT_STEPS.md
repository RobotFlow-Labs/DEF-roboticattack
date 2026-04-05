# NEXT_STEPS — DEF-roboticattack
## Last Updated: 2026-04-05
## Status: TRAINING COMPLETE — defense model trained, exported, pushed to HF
## MVP Readiness: 85%

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
- [x] Build PatchDetectorNet (multi-branch CNN: freq+edge+spatial)
- [x] Build synthetic adversarial patch dataset generator
- [x] Integrate compiled CUDA kernels (fused_patch_apply, fused_action_perturb)
- [x] Build training script with warmup cosine + checkpointing + early stopping
- [x] Build evaluation pipeline (accuracy + latency benchmarks)
- [x] Train defense model (92.7% acc, 99.6% precision, F1=0.92)
- [x] Export: pth + safetensors + ONNX + TRT FP16 + TRT FP32
- [x] Push to HuggingFace: ilessio-aiflowlab/DEF-roboticattack
- [x] Docker serving infrastructure (Dockerfile.serve, docker-compose, anima_module.yaml)
- [x] Config-driven training (configs/paper.toml, configs/debug.toml)

### TODO (next phase)
- [ ] Train on real OpenVLA image data (BridgeData V2 / LIBERO) for production accuracy
- [ ] Build MLX Metal equivalent kernels for Apple Silicon parity
- [ ] Integration test: plug defense into upstream OpenVLA eval pipeline
- [ ] Benchmark TRT FP16 inference on Jetson Orin NX
- [ ] Add adversarial training loop (jointly train detector + patch generator)
- [ ] Add multi-attack-type evaluation (UADA vs UPA vs TMA detection rates)

### Metrics (Trained Model)
- **Accuracy**: 92.7%
- **Precision**: 99.6% (very low false positive rate)
- **Recall**: 85.9%
- **F1**: 0.92
- **Model params**: 19,841
- **Inference latency**: 5.28ms (model-only, batch=8, L4)
- **Pipeline throughput**: 434 samples/s (full defense stack)
- **Training time**: 953s (11 epochs, early stop, 50K synthetic samples)

### Exports
- `patch_detector.pth` (91.7 KB)
- `patch_detector.safetensors` (81.7 KB)
- `patch_detector.onnx` (81.4 KB)
- `patch_detector_fp16.trt` (250.7 KB)
- `patch_detector_fp32.trt` (287.3 KB)

### HuggingFace
- Repo: https://huggingface.co/ilessio-aiflowlab/DEF-roboticattack
