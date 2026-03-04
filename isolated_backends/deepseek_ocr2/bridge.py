"""
DeepSeek-OCR-2 로컬 백엔드.
격리 venv의 Python으로 deepseek_ocr2_worker.py를 subprocess 실행.

격리 venv 경로: pdf_parser/backends/.venv-deepseek/
  → setup_deepseek_venv.sh 로 생성
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

BACKENDS_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = BACKENDS_DIR / "deepseek_ocr2_worker.py"

# 격리 venv의 python 경로
VENV_PYTHON = BACKENDS_DIR / ".venv-deepseek" / "bin" / "python"


def _get_python() -> str:
    """격리 venv 우선, 없으면 시스템 python."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    # fallback: 현재 실행 중인 python (transformers 버전 호환 여부 주의)
    import sys
    print(
        f"  [DeepSeek] ⚠️  격리 venv 없음 ({VENV_PYTHON}). "
        "setup_deepseek_venv.sh 실행 권장. 현재 Python 사용.",
        flush=True,
    )
    return sys.executable


def convert_pdf(pdf_path: str, output_path: str) -> str:
    """DeepSeek-OCR-2 worker subprocess 실행 → Markdown 반환."""
    python = _get_python()

    env = os.environ.copy()
    cert = "/Users/sukbeom/Documents/cert/combined-ca-bundle.pem"
    if os.path.exists(cert):
        env["SSL_CERT_FILE"] = cert
        env["REQUESTS_CA_BUNDLE"] = cert

    print(f"  [DeepSeek-OCR-2] worker 실행: {Path(python).parent.parent.name}/python", flush=True)

    result = subprocess.run(
        [python, str(WORKER_SCRIPT), str(pdf_path)],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"DeepSeek-OCR-2 worker 실패\nSTDERR: {result.stderr[-2000:]}\nSTDOUT: {result.stdout[-500:]}"
        )

    stdout = result.stdout
    START = "---OUTPUT_START---"
    END = "---OUTPUT_END---"

    if START not in stdout or END not in stdout:
        raise RuntimeError(
            f"출력 토큰 없음. STDOUT:\n{stdout[:1000]}\nSTDERR:\n{result.stderr[-500:]}"
        )

    json_str = stdout[stdout.find(START) + len(START): stdout.find(END)].strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 파싱 실패: {e}\n내용: {json_str[:500]}")

    if "error" in data:
        raise RuntimeError(f"DeepSeek-OCR-2 내부 오류: {data['error']}")

    # {페이지번호: markdown} → 단일 파일
    pages = {int(k): v for k, v in data.items()}
    combined = "\n\n---\n\n".join(
        f"# Page {n}\n\n{md}" for n, md in sorted(pages.items())
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"  ✓ 저장: {output_path}", flush=True)
    return combined
