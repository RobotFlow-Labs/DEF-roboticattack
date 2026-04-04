#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/cuda_server_setup.sh
bash scripts/cuda_smoke.sh
bash scripts/cuda_benchmark.sh

echo "[OK] CUDA handoff checks passed"
