#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is required. Install uv first."
  exit 1
fi

PY_BIN="python3.11"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "[ERROR] python3.11 not found on server"
  exit 1
fi

uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install -e .
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

echo "[INFO] Backend probe"
ANIMA_BACKEND=auto .venv/bin/def-roboticattack backend-info

echo "[INFO] Setup complete"
