"""
E2E 테스트: PDF → (페이지 분할) → PaddleOCR 구조화 worker → output_formatter → Upstage JSON.
10페이지씩 분할 처리하여 output_structure/ 기존 파일과 동일한 패턴으로 저장.

Usage:
    python run_e2e_test.py <pdf_path> [output_dir]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Paths
PADDLEOCR_DIR = Path(
    os.environ.get(
        "PADDLEOCR_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "PaddleOCR"),
    )
)
WORKER_SCRIPT = Path(__file__).resolve().parent / "worker_structured.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_formatter import format_to_upstage


def split_pdf(pdf_path: str, start: int, end: int, out_path: str) -> int:
    """PDF에서 start~end 페이지(0-based)를 추출. 추출된 페이지 수 반환."""
    import fitz

    src = fitz.open(pdf_path)
    dst = fitz.open()
    actual_end = min(end, len(src) - 1)
    for i in range(start, actual_end + 1):
        dst.insert_pdf(src, from_page=i, to_page=i)
    dst.save(out_path)
    count = len(dst)
    dst.close()
    src.close()
    return count


def run_worker(pdf_path: str) -> list[dict]:
    """격리 venv에서 worker_structured.py 실행, 블록 단위 JSON 반환."""
    venv_python = PADDLEOCR_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(f"PaddleOCR venv not found: {venv_python}")

    cmd = [str(venv_python), str(WORKER_SCRIPT), str(pdf_path)]

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    cert = os.environ.get("SSL_CERT_BUNDLE", "")
    if cert and os.path.exists(cert):
        env["SSL_CERT_FILE"] = cert
        env["REQUESTS_CA_BUNDLE"] = cert

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=PADDLEOCR_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - t0
    print(f"    Worker done in {elapsed:.1f}s (exit={result.returncode})", flush=True)

    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[:500]}", flush=True)
        raise RuntimeError(f"Worker failed: {result.stderr[:300]}")

    output_txt = result.stdout
    start_token = "---OUTPUT_START---"
    end_token = "---OUTPUT_END---"

    if start_token not in output_txt or end_token not in output_txt:
        print(f"    STDOUT (first 500): {output_txt[:500]}", flush=True)
        raise RuntimeError("Output tokens not found")

    start_idx = output_txt.find(start_token) + len(start_token)
    end_idx = output_txt.find(end_token)
    json_str = output_txt[start_idx:end_idx].strip()

    data = json.loads(json_str)
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Worker error: {data['error']}")

    return data


def get_pdf_page_count(pdf_path: str) -> int:
    """PDF 페이지 수."""
    try:
        import fitz

        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except ImportError:
        pass
    try:
        r = subprocess.run(
            ["mdls", "-name", "kMDItemNumberOfPages", pdf_path],
            capture_output=True,
            text=True,
        )
        match = re.search(r"(\d+)", r.stdout)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 48


def fix_page_indices(raw_blocks: list[dict], global_start: int) -> list[dict]:
    """분할 PDF의 0-based page_index를 원본 PDF 기준으로 보정."""
    for page_data in raw_blocks:
        page_data["page_index"] = page_data["page_index"] + global_start
    return raw_blocks


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <pdf_path> [output_dir]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_structure"
    os.makedirs(output_dir, exist_ok=True)

    total_pages = get_pdf_page_count(pdf_path)
    print(f"PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Output dir: {output_dir}")

    pdf_name = Path(pdf_path).stem
    base_name = re.sub(r"[^\w가-힣]", "_", pdf_name)
    base_name = re.sub(r"_+", "_", base_name).strip("_")

    chunk_size = 10
    all_results = []

    with tempfile.TemporaryDirectory(prefix="paddleocr_e2e_") as tmpdir:
        for chunk_start in range(0, total_pages, chunk_size):
            chunk_end = min(chunk_start + chunk_size - 1, total_pages - 1)
            page_label = f"{chunk_start:04d}_{chunk_end:04d}"

            print(f"\n{'=' * 60}")
            print(f"Chunk: pages {chunk_start}-{chunk_end} ({page_label})")
            print(f"{'=' * 60}")

            # 1. PDF 분할
            chunk_pdf = os.path.join(tmpdir, f"chunk_{page_label}.pdf")
            n_pages = split_pdf(pdf_path, chunk_start, chunk_end, chunk_pdf)
            print(f"  Split: {n_pages} pages → {chunk_pdf}", flush=True)

            # 2. Worker 실행
            try:
                raw_blocks = run_worker(chunk_pdf)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            # 3. page_index 보정 (분할 PDF 0-based → 원본 기준)
            raw_blocks = fix_page_indices(raw_blocks, chunk_start)

            # 4. Upstage 포맷 변환
            upstage_json = format_to_upstage(raw_blocks, model="paddleocr-vl")

            # 5. usage 보정
            upstage_json["usage"]["pages"] = total_pages
            upstage_json["usage"]["standard"] = list(
                range(chunk_start + 1, chunk_end + 2)
            )

            # 6. 원자적 저장
            out_file = os.path.join(
                output_dir, f"paddleocr_{base_name}_{page_label}.json"
            )
            tmp_file = out_file + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(upstage_json, f, ensure_ascii=False, indent=2)
            os.rename(tmp_file, out_file)

            elements = len(upstage_json["elements"])
            cats = {}
            for e in upstage_json["elements"]:
                cats[e["category"]] = cats.get(e["category"], 0) + 1

            print(f"  Saved: {out_file}")
            print(f"  Elements: {elements}")
            print(f"  Categories: {cats}")

            all_results.append(
                {
                    "chunk": page_label,
                    "file": out_file,
                    "elements": elements,
                    "categories": cats,
                }
            )

    # 요약
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    total_elements = sum(r["elements"] for r in all_results)
    expected_chunks = (total_pages + chunk_size - 1) // chunk_size
    print(f"Chunks: {len(all_results)}/{expected_chunks}")
    print(f"Total elements: {total_elements}")
    for r in all_results:
        print(f"  {r['chunk']}: {r['elements']} elements — {r['categories']}")


if __name__ == "__main__":
    main()
