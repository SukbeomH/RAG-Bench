#!/usr/bin/env bash
# CUDA 전용 — GPU 서버에서만 실행 가능
set -euo pipefail
cd "$(dirname "$0")"
exec uv run uvicorn server:app --host 0.0.0.0 --port 8001
