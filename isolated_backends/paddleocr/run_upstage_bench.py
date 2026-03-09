"""
Upstage Document Parse API로 벤치마크 PDF를 처리하여 raw JSON 저장.

Usage:
    python run_upstage_bench.py [output_dir]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

UPSTAGE_API_URL = "https://api.upstage.ai/v1/document-digitization"
BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_pdfs"
API_KEY = os.environ.get("UPSTAGE_API_KEY", "")


def call_upstage(pdf_path: Path) -> dict:
    """Upstage Document Parse API 호출, 전체 raw JSON 반환."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": "document-parse",
        "output_formats": '["markdown", "text", "html"]',
        "ocr": "auto",
        "coordinates": "true",
        "base64_encoding": '["table", "figure", "chart"]',
    }

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    response = requests.post(
        UPSTAGE_API_URL,
        headers=headers,
        files={"document": (pdf_path.name, pdf_bytes, "application/pdf")},
        data=data,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def main():
    if not API_KEY:
        print("ERROR: UPSTAGE_API_KEY 환경변수가 필요합니다.")
        print("  export UPSTAGE_API_KEY=up_xxx")
        sys.exit(1)

    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else BENCH_DIR / "upstage_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(BENCH_DIR.glob("*.pdf"))
    print(f"PDFs: {len(pdfs)}")
    print(f"Output: {output_dir}")
    print()

    for pdf in pdfs:
        out_file = output_dir / f"{pdf.stem}.json"
        if out_file.exists():
            print(f"SKIP (exists): {pdf.name}")
            continue

        print(f"Processing: {pdf.name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = call_upstage(pdf)
            elapsed = time.time() - t0

            # 원자적 저장
            tmp = str(out_file) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            os.rename(tmp, str(out_file))

            elements = len(result.get("elements", []))
            pages = result.get("usage", {}).get("pages", "?")
            print(f"OK ({elapsed:.1f}s, {pages} pages, {elements} elements)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}")

        # Rate limit 대비
        time.sleep(2)

    print()
    print("Done.")
    for f in sorted(output_dir.glob("*.json")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
