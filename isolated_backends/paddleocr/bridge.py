"""
PaddleOCR Backend (Python 3.12 benchmark runtime)
Calls the Python 3.13 PaddleOCR pipeline in an isolated venv.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# Paths
PADDLEOCR_DIR = Path(
    os.environ.get(
        "PADDLEOCR_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "PaddleOCR"),
    )
)
WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"


def convert_pdf(pdf_path: str, output_path: str) -> str:
    """
    Runs the paddleocr doc_parser using the isolated uv Python 3.13 venv.
    Captures JSON output and converts it to a markdown file.
    """
    venv_python = PADDLEOCR_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(f"PaddleOCR venv not found: {venv_python}")
    cmd = [str(venv_python), str(WORKER_SCRIPT), str(pdf_path)]

    print(
        f"  [PaddleOCR] Running isolated worker: {venv_python.name} {WORKER_SCRIPT.name}",
        flush=True,
    )

    env = os.environ.copy()
    # Remove parent VIRTUAL_ENV to avoid uv ignoring the project venv
    env.pop("VIRTUAL_ENV", None)
    cert = os.environ.get("SSL_CERT_BUNDLE", "")
    if cert and os.path.exists(cert):
        env["SSL_CERT_FILE"] = cert
        env["REQUESTS_CA_BUNDLE"] = cert

    result = subprocess.run(
        cmd, cwd=PADDLEOCR_DIR, env=env, capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"PaddleOCR Subprocess Failed: {result.stderr}\n\nSTDOUT: {result.stdout}"
        )

    output_txt = result.stdout
    start_token = "---OUTPUT_START---"
    end_token = "---OUTPUT_END---"

    if start_token not in output_txt or end_token not in output_txt:
        raise RuntimeError(f"Could not find output tokens. STDOUT: {output_txt[:1000]}")

    start_idx = output_txt.find(start_token) + len(start_token)
    end_idx = output_txt.find(end_token)

    json_str = output_txt[start_idx:end_idx].strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON decode failed: {e}\nJSON payload: {json_str[:500]}")

    if "error" in data:
        raise RuntimeError(f"PaddleOCR Internal Error: {data['error']}")

    markdown_pages = {int(k): v for k, v in data.items()}

    combined = "\n\n---\n\n".join(
        f"# Page {page_num}\n\n{content}"
        for page_num, content in sorted(markdown_pages.items())
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"  ✓ Saved completely: {output_path}")
    return combined
