#!/usr/bin/env bash
# Docling 격리 venv 생성 스크립트
# 사용: cd isolated_backends/docling && bash setup_venv.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv-docling"

echo "=== Docling 격리 venv 생성 ==="
echo "경로: ${VENV_DIR}"

python3.12 -m venv "${VENV_DIR}"
echo "✓ venv 생성 완료"

"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install 'docling>=2.75' pymupdf
echo "✓ 의존성 설치 완료"

echo ""
echo "=== 검증 ==="
"${VENV_DIR}/bin/python" -c "import docling; print('docling OK')"
"${VENV_DIR}/bin/python" -c "import fitz; print('PyMuPDF OK')"
echo ""
echo "✓ 설정 완료. worker.py 실행 가능."
