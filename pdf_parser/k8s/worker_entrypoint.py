#!/usr/bin/env python3
"""
PDF Parser K8s Worker Entrypoint

K8s Job에서 실행되는 PDF → Markdown 변환 워커.

환경변수:
  INPUT_DIR       — PDF 파일이 있는 PVC 경로 (기본: /input)
  OUTPUT_DIR      — Markdown 출력 PVC 경로 (기본: /output)
  PARSE_MODE      — "hybrid" | "document" (기본: hybrid)
  GEMINI_API_KEY  — Gemini VLM API 키 (VLM 모드 필요)
  BATCH_ID        — 배치 식별자 (로깅용)
  FILE_PATTERN    — 처리할 파일 glob 패턴 (기본: *.pdf)
"""

import json
import os
import sys
import time
from pathlib import Path


def main():
    input_dir = Path(os.environ.get("INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    parse_mode = os.environ.get("PARSE_MODE", "hybrid")
    api_key = os.environ.get("GEMINI_API_KEY")
    batch_id = os.environ.get("BATCH_ID", "unknown")
    file_pattern = os.environ.get("FILE_PATTERN", "*.pdf")

    print(f"\n{'=' * 60}")
    print(f" PDF Parser Worker")
    print(f"{'=' * 60}")
    print(f"  Batch ID   : {batch_id}")
    print(f"  Input      : {input_dir}")
    print(f"  Output     : {output_dir}")
    print(f"  Mode       : {parse_mode}")
    print(f"  Pattern    : {file_pattern}")
    print(f"  Gemini API : {'SET' if api_key else 'NOT SET'}")
    print()

    # 입력 검증
    if not input_dir.exists():
        print(f"ERROR: 입력 디렉토리 없음: {input_dir}")
        sys.exit(1)

    pdf_files = sorted(input_dir.glob(file_pattern))
    if not pdf_files:
        print(f"WARN: PDF 파일 없음: {input_dir}/{file_pattern}")
        # 빈 결과 기록 후 정상 종료
        _write_result(output_dir, batch_id, [], [])
        return

    print(f"  PDF 파일 수: {len(pdf_files)}")
    for f in pdf_files:
        print(f"    - {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    print()

    # pdf_parser 모듈 임포트
    sys.path.insert(0, "/app/pdf_parser")
    from smart_router import route_pdf
    from quality_checker import check_quality

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []
    t_total = time.time()

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{'─' * 40}")
        print(f"  [{i}/{len(pdf_files)}] {pdf_file.name}")
        print(f"{'─' * 40}")

        t0 = time.time()
        try:
            md_path = route_pdf(
                str(pdf_file),
                str(output_dir),
                api_key=api_key,
                mode=parse_mode,
            )

            elapsed = time.time() - t0
            quality = check_quality(md_path)

            result = {
                "file": pdf_file.name,
                "output": Path(md_path).name,
                "mode": parse_mode,
                "elapsed_s": round(elapsed, 1),
                "word_count": quality["word_count"],
                "has_headers": quality["has_headers"],
                "has_tables": quality["has_tables"],
                "has_formulas": quality["has_formulas"],
            }
            results.append(result)

            print(f"  완료: {elapsed:.1f}s | {quality['word_count']} words")

        except Exception as e:
            elapsed = time.time() - t0
            error_info = {
                "file": pdf_file.name,
                "error": str(e),
                "elapsed_s": round(elapsed, 1),
            }
            errors.append(error_info)
            print(f"  ERROR: {e}")

    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  완료: {len(results)}/{len(pdf_files)} 성공, {len(errors)} 실패")
    print(f"  총 소요: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")

    # 결과 JSON 기록
    _write_result(output_dir, batch_id, results, errors)


def _write_result(output_dir: Path, batch_id: str, results: list, errors: list):
    """결과를 원자적으로 JSON 파일에 기록."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "batch_id": batch_id,
        "total_files": len(results) + len(errors),
        "success": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }

    result_path = output_dir / f"parse_result_{batch_id}.json"
    tmp_path = result_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    tmp_path.rename(result_path)
    print(f"  결과 저장: {result_path}")

    # DONE 시그널
    done_path = output_dir / "DONE"
    done_path.write_text(f"{batch_id}\n")


if __name__ == "__main__":
    main()
