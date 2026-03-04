"""
Docling 격리 subprocess bridge.
별도 venv의 Python으로 worker.py를 실행하여 의존성 충돌 우회.

격리 venv 경로: isolated_backends/docling/.venv-docling/
  → setup_venv.sh 로 생성
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

BACKENDS_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = BACKENDS_DIR / "worker.py"
VENV_PYTHON = BACKENDS_DIR / ".venv-docling" / "bin" / "python"


def _get_python() -> str:
    """격리 venv python 경로 반환. 없으면 에러."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    raise FileNotFoundError(
        f"Docling 격리 venv 없음: {VENV_PYTHON}\n"
        "  → cd isolated_backends/docling && bash setup_venv.sh"
    )


def convert_pdf(pdf_path: str) -> dict[int, str]:
    """Docling worker subprocess 실행 → {페이지번호: markdown} 반환."""
    python = _get_python()

    print(
        f"  [docling] subprocess 실행: {VENV_PYTHON.parent.parent.name}/python",
        flush=True,
    )

    result = subprocess.run(
        [python, str(WORKER_SCRIPT), str(pdf_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Docling worker 실패 (rc={result.returncode})\n"
            f"STDERR: {result.stderr[-2000:]}\n"
            f"STDOUT: {result.stdout[-500:]}"
        )

    stdout = result.stdout
    START = "---OUTPUT_START---"
    END = "---OUTPUT_END---"

    if START not in stdout or END not in stdout:
        raise RuntimeError(
            f"출력 토큰 없음.\nSTDOUT:\n{stdout[:1000]}\nSTDERR:\n{result.stderr[-500:]}"
        )

    json_str = stdout[stdout.find(START) + len(START) : stdout.find(END)].strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 파싱 실패: {e}\n내용: {json_str[:500]}")

    if "error" in data:
        raise RuntimeError(f"Docling 내부 오류: {data['error']}")

    return {int(k): v for k, v in data.items()}
