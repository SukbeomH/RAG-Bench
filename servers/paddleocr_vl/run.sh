#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# SSL 인증서 설정 (사내 프록시 환경)
CERT="${SSL_CERT_BUNDLE:-/Users/sukbeom/Documents/cert/combined-ca-bundle.pem}"
if [[ -f "$CERT" ]]; then
    export SSL_CERT_FILE="$CERT"
    export REQUESTS_CA_BUNDLE="$CERT"
    export CURL_CA_BUNDLE="$CERT"
fi

# 모델 소스 연결 체크 스킵 (오프라인/프록시 환경)
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

exec uv run uvicorn server:app --host 0.0.0.0 --port 8000
