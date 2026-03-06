#!/usr/bin/env bash
# macOS Apple Silicon 전용 — PaddleOCR-VL mlx-vlm 서버
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python -m mlx_vlm.server \
    --port 8111
