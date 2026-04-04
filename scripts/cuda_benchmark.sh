#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

python benchmarks/benchmark_patch_guard.py --backend cuda --warmup 30 --iterations 300 --batch-size 8 --height 224 --width 224
