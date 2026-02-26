"""
PDF Parser K8s 벤치마크 워커 엔트리포인트.

단일 (backend × pdf) 조합을 실행하고 결과를 /results PVC에 저장.

환경변수:
  PDF_BACKEND        : 사용할 백엔드
                       (pymupdf|docling|openai|openai-4.1|upstage|upstage-enhanced|
                        granite-vision|got-ocr2|paddleocr-vl)
  PDF_FILE           : 처리할 PDF 파일명 (benchmark_pdfs/ 내 파일명)
  PARSE_MODE         : 파싱 모드 (direct|document|hybrid), 기본 direct
  RESULTS_DIR        : 결과 저장 경로 (기본: /results)
  BENCHMARK_PDFS_DIR : PDF 및 GT 파일 위치 (기본: /app/benchmark_pdfs)

  # API 키 (Secret에서 주입)
  OPENAI_API_KEY     : OpenAI API 키 (openai 백엔드)
  UPSTAGE_API_KEY    : Upstage API 키 (upstage 백엔드)

  # OCR 백엔드 엔드포인트 (K8s 내부 서비스, 기본값 사용 가능)
  OPENSOURCE_VLM_ENDPOINT : 오픈소스 VLM 서비스 URL 오버라이드
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 경로 설정 ────────────────────────────────────────────────────────────────
sys.path.insert(0, "/app")

RESULTS_DIR   = Path(os.environ.get("RESULTS_DIR", "/results"))
BENCH_DIR     = Path(os.environ.get("BENCHMARK_PDFS_DIR", "/app/benchmark_pdfs"))
GT_DIR        = BENCH_DIR / "gt"


# ── 백엔드 디스패처 (runner.py와 동일 로직) ───────────────────────────────────

def run_backend(backend: str, pdf_path: Path, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "pymupdf":
        import category1_simple as cat
        return cat.convert_pdf(str(pdf_path), str(output_path))

    elif backend == "docling":
        import category2_medium as cat
        converter = cat.build_converter()
        return cat.convert_pdf(str(pdf_path), str(output_path), converter=converter)

    elif backend == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY 환경변수 필요")
        import category3_openai as cat
        pages = cat.convert_pdf(str(pdf_path), key, model="gpt-4o")
        cat.save_markdown(pages, str(output_path))
        return output_path.read_text(encoding="utf-8")

    elif backend == "openai-4.1":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY 환경변수 필요")
        import category3_openai as cat
        pages = cat.convert_pdf(str(pdf_path), key, model="gpt-4.1")
        cat.save_markdown(pages, str(output_path))
        return output_path.read_text(encoding="utf-8")

    elif backend == "upstage":
        key = os.environ.get("UPSTAGE_API_KEY", "")
        if not key:
            raise ValueError("UPSTAGE_API_KEY 환경변수 필요")
        import category3_upstage as cat
        pages = cat.convert_pdf(str(pdf_path), key, mode="auto")
        cat.save_markdown(pages, str(output_path))
        return output_path.read_text(encoding="utf-8")

    elif backend == "upstage-enhanced":
        key = os.environ.get("UPSTAGE_API_KEY", "")
        if not key:
            raise ValueError("UPSTAGE_API_KEY 환경변수 필요")
        import category3_upstage as cat
        pages = cat.convert_pdf(str(pdf_path), key, mode="enhanced")
        cat.save_markdown(pages, str(output_path))
        return output_path.read_text(encoding="utf-8")

    elif backend in ("granite-vision", "got-ocr2", "paddleocr-vl"):
        # K8s 내부 OCR 서비스 호출 (OpenAI-compatible API)
        import category3_opensource as cat
        pages = cat.convert_pdf(str(pdf_path), api_key="ollama", backend_key=backend)
        cat.save_markdown(pages, str(output_path))
        return output_path.read_text(encoding="utf-8")

    else:
        raise ValueError(f"알 수 없는 백엔드: {backend}")


def _save_metrics(
    result_dir: Path,
    backend: str,
    pdf_name: str,
    mode: str,
    pred_text: str,
    gt_text: str | None,
    speed_s: float,
    error: str | None = None,
) -> None:
    """metrics.json 원자적 저장 + DONE 파일 생성."""
    from benchmark.evaluator import evaluate_document

    if error:
        data = {
            "backend": backend,
            "pdf_name": pdf_name,
            "mode": mode,
            "error": error,
            "pages": [],
            "summary": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    else:
        result = evaluate_document(
            pred_text=pred_text,
            gt_text=gt_text,
            speed_s=speed_s,
            backend=backend,
            pdf_name=pdf_name,
            mode=mode,
        )
        data = {
            "backend": backend,
            "pdf_name": pdf_name,
            "mode": mode,
            "error": None,
            "pages": [
                {
                    "page":        p.page,
                    "text_ned":    round(p.text_ned, 4) if p.text_ned >= 0 else None,
                    "table_teds":  round(p.table_teds, 4) if p.table_teds >= 0 else None,
                    "speed_s":     round(p.speed_s, 3),
                    "word_count":  p.word_count,
                    "has_headers": p.has_headers,
                    "has_tables":  p.has_tables,
                    "has_formulas":p.has_formulas,
                }
                for p in result.pages
            ],
            "summary": {
                "avg_text_ned":   round(result.avg_text_ned, 4)   if result.pages else None,
                "avg_table_teds": round(result.avg_table_teds, 4) if result.avg_table_teds >= 0 else None,
                "avg_speed_s":    round(result.avg_speed, 3)      if result.pages else None,
                "total_time_s":   round(result.total_time_s, 3)   if result.pages else None,
                "total_words":    result.total_words,
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    tmp = result_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(result_dir / "metrics.json")
    (result_dir / "DONE").touch()
    print(f"[DONE] metrics.json → {result_dir}")


def main() -> None:
    backend  = os.environ.get("PDF_BACKEND", "")
    pdf_name = os.environ.get("PDF_FILE", "")
    mode     = os.environ.get("PARSE_MODE", "direct")

    if not backend or not pdf_name:
        print("오류: PDF_BACKEND, PDF_FILE 환경변수를 설정하세요.", file=sys.stderr)
        sys.exit(1)

    pdf_path = BENCH_DIR / pdf_name
    if not pdf_path.exists():
        print(f"오류: PDF 없음: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # GT 로드
    from benchmark.spec import GT_MAP
    gt_name = GT_MAP.get(pdf_name)
    gt_text: str | None = None
    if gt_name:
        gt_path = GT_DIR / gt_name
        if gt_path.exists():
            gt_text = gt_path.read_text(encoding="utf-8")

    # 결과 디렉토리
    pdf_stem = pdf_name.replace(".pdf", "").replace("_", "-")
    label = f"{backend}-{pdf_stem}-{mode}"[:63]
    result_dir  = RESULTS_DIR / label
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "output.md"

    print(f"[START] backend={backend}  pdf={pdf_name}  mode={mode}")

    t0 = time.perf_counter()
    try:
        pred_text = run_backend(backend, pdf_path, output_path)
    except Exception as e:
        speed_s = time.perf_counter() - t0
        print(f"[ERROR] {e}", file=sys.stderr)
        _save_metrics(result_dir, backend, pdf_name, mode,
                      pred_text="", gt_text=None,
                      speed_s=speed_s, error=str(e))
        sys.exit(1)

    speed_s = time.perf_counter() - t0
    _save_metrics(result_dir, backend, pdf_name, mode,
                  pred_text=pred_text, gt_text=gt_text,
                  speed_s=speed_s)

    print(f"[DONE]  {label}  {speed_s:.1f}s")


if __name__ == "__main__":
    main()
