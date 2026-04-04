#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

ANIMA_BACKEND=cuda .venv/bin/def-roboticattack backend-info
ANIMA_BACKEND=cuda .venv/bin/def-roboticattack dry-run --batch-size 4 --height 224 --width 224
