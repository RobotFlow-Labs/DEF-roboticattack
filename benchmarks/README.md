# Benchmarks — DEF-roboticattack

## Baseline vs Optimized
| Metric | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| Latency (ms) | — | — | — |
| Throughput (samples/s) | — | — | — |
| Memory (MB) | — | — | — |
| Model Size (MB) | — | — | — |

## Hardware Targets
- Mac Studio M-series (MLX)
- RTX 6000 Pro Blackwell (CUDA)
- Jetson Orin NX (edge deployment)
- Jetson AGX Orin (edge deployment)

## Baseline Command
```bash
python benchmarks/benchmark_patch_guard.py --backend auto --iterations 200 --batch-size 8
```

## Output
- JSON reports saved under `benchmarks/results/patch_guard_<backend>.json`
