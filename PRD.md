# PRD — DEF-roboticattack (Wave 8)

## 1. Objective
Build a defense-only ANIMA module that detects and mitigates adversarial patch attacks against VLA systems (UADA, UPA, TMA threat model), with dual-compute support on CUDA and MLX, and explicit kernel/IP optimization tracks.

## 2. Source Inputs
- Upstream code: `repositories/roboticAttack` (checked out `main`)
- Local paper copies:
  - `papers/Wang_ICCV2025_VLA_Adversarial.pdf`
  - `papers/Wang_arXiv_2411.13587.pdf`
- Upstream attack paths analyzed:
  - `VLAAttacker/UADA_wrapper.py`
  - `VLAAttacker/UPA_wrapper.py`
  - `VLAAttacker/TMA_wrapper.py`
  - `VLAAttacker/white_patch/UADA.py`
  - `VLAAttacker/white_patch/UPA.py`
  - `VLAAttacker/white_patch/TMA.py`
  - `VLAAttacker/white_patch/appply_random_transform.py`
  - `VLAAttacker/white_patch/openvla_dataloader.py`

## 3. Problem Statement
The upstream repo provides attack implementations and evaluation scripts, but no production defense stack, no backend-abstraction for MLX/CUDA parity, and no ANIMA-style kernel optimization pipeline for defensive runtime.

## 4. Scope
### In Scope
- Defense scaffold package (Python 3.11, hatchling, uv workflow)
- Backend abstraction for `ANIMA_BACKEND=mlx|cuda|auto`
- Input sanitization and anomaly scoring baseline
- Kernel/IP scaffold in `kernels/` for CUDA + MLX + Triton track
- Benchmark harness in `benchmarks/` for latency/throughput/memory framework
- Task slicing and execution-ready work items in `tasks/`

### Out of Scope (for this scaffold phase)
- Full training/eval reproduction of upstream attack claims
- Final optimized custom kernels (production performance)
- Complete OpenVLA inference integration and Jetson deployment

## 5. Users and Stakeholders
- ANIMA defense engineers
- Robotics security researchers
- Edge-deployment optimization team

## 6. Functional Requirements
1. Module must run on Python `>=3.11`.
2. Module must expose backend resolution logic for MLX/CUDA.
3. Defense runtime must support at least:
   - Patch-like perturbation sanitization pass
   - Patch anomaly scoring pass
4. Benchmark harness must measure runtime for defense passes.
5. Kernel directories must include explicit optimization targets and API placeholders.
6. Task files must be granular, independent, and directly executable.

## 7. Non-Functional Requirements
- Deterministic behavior for baseline runs (seeded where applicable)
- Clear failure messages when requested backend is unavailable
- Minimal dependencies in scaffold phase
- File and module naming aligned with ANIMA conventions

## 8. Architecture (Scaffold)
- `src/def_roboticattack/device.py`
  - Runtime backend resolution and capability metadata
- `src/def_roboticattack/defense/transforms.py`
  - Input-domain sanitization ops (baseline)
- `src/def_roboticattack/defense/detector.py`
  - Patch anomaly scoring (baseline)
- `src/def_roboticattack/pipeline/runtime.py`
  - Composed defense runtime API
- `src/def_roboticattack/data/rlds_adapter.py`
  - Dataset spec placeholders aligned with Bridge/LIBERO style
- `benchmarks/benchmark_patch_guard.py`
  - Baseline benchmark runner with JSON output
- `kernels/*`
  - CUDA/MLX/Triton kernel stubs for owned IP path

## 9. Defense Strategy (Initial)
- Input-side patch suppression via intensity clamping and blur smoothing
- Output-side anomaly score via edge-energy z-score
- Combined risk output to gate downstream policy inference

## 10. Kernel Optimization Plan
### CUDA
- Fused patch blend + normalization
- Optional mixed precision (FP16/BF16)

### MLX/Metal
- Equivalent fused blend kernel
- Backend parity checks vs CUDA path

### Quantization
- INT8/FP16 defense runtime compatibility path

## 11. Benchmark Plan
- Baseline (Python/Torch ops)
- Candidate optimized kernel path (future)
- Metrics:
  - Latency (mean/std)
  - Throughput (samples/sec)
  - Peak memory (where supported)

## 12. Acceptance Criteria (Scaffold Phase)
- `PRD.md` complete
- `tasks/TASK-*.md` created and prioritized
- Python package scaffold present and importable
- `device.py` resolves `mlx|cuda|cpu` robustly
- Benchmark script emits JSON report
- Kernel stubs present in all required subfolders

## 13. Risks
- Upstream code quality issues (e.g., transform module formatting/robustness)
- Heavy model dependencies limit local reproduction speed
- CUDA and MLX parity can diverge without strict test fixtures

## 14. Milestones
1. Intake + PRD + task slicing
2. Dual-backend and defense baseline scaffold
3. OpenVLA integration layer
4. Kernel optimization implementation
5. Quantized runtime and deployment hardening

