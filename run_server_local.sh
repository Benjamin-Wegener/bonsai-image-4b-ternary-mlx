#!/bin/zsh
set -euo pipefail

DEMO_DIR="/Users/user/dev/Bonsai-Image-Demo"
PYTHON="$DEMO_DIR/.venv/bin/python"
MODEL_DIR="/Users/user/models/safetensors/bonsai-image-4B-ternary-mlx"
PORT=12345

export BONSAI_MODEL_DIR="$MODEL_DIR"
export MFLUX_STUDIO_BAKED_MODEL_PATH="$MODEL_DIR"

# Kill anything running on 12345
pkill -9 -f "server_12345" 2>/dev/null || true
lsof -ti :$PORT | xargs kill -9 2>/dev/null || true

cd "$DEMO_DIR"
exec "$PYTHON" -m uvicorn server_12345:app --host 127.0.0.1 --port $PORT
