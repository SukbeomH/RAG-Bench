"""
로컬 벤치마크 러너

autorag_parsers 레지스트리를 통해 PDF → Markdown 변환 후 평가.
K8s 없이 빠른 검증용.

사용법:
    python -m autorag_pdf_eval.runner --preset quick
    python -m autorag_pdf_eval.runner --preset phase1 --output ./bench_results
    python -m autorag_pdf_eval.runner --backend pymupdf --pdf text_only.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

# .env 자동 로드 (python-dotenv 없어도 동작)
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

_cert_bundle = os.environ.get("SSL_CERT_BUNDLE", "")
if _cert_bundle and os.path.exists(_cert_bundle):
    os.environ.setdefault("SSL_CERT_FILE", _cert_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _cert_bundle)

from autorag_pdf_eval.evaluator import BenchResult, evaluate_document
from autorag_pdf_eval.spec import Backend, BenchSpec, GT_MAP, get_preset


# ── 경로 설정 ─────────────────────────────────────────────────────────────────

BENCH_DIR = Path(__file__).parent.parent / "benchmark_pdfs"
GT_DIR = BENCH_DIR / "gt"


# ── 백엔드 디스패처 ───────────────────────────────────────────────────────────


def run_backend(
    backend: Backend, pdf_path: Path, output_path: Path, api_key: str | None
) -> str:
    """
    autorag_parsers 레지스트리로 PDF → Markdown 변환.

    반환: 변환된 마크다운 문자열
    """
    from autorag_parsers import get_parser

    output_path.parent.mkdir(parents=True, exist_ok=True)

    parser = get_parser(backend)
    result = parser.convert(str(pdf_path))
    md_text = result.full_markdown

    output_path.write_text(md_text, encoding="utf-8")
    return md_text


# ── 단일 스펙 실행 ────────────────────────────────────────────────────────────


def run_spec(
    spec: BenchSpec,
    output_dir: Path,
    api_key: str | None = None,
    verbose: bool = True,
) -> BenchResult:
    """
    단일 BenchSpec 실행 → BenchResult 반환.

    산출물:
        {output_dir}/{spec.label}/output.md
        {output_dir}/{spec.label}/metrics.json
    """
    pdf_path = BENCH_DIR / spec.pdf_name
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 없음: {pdf_path}")

    result_dir = output_dir / spec.label
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "output.md"

    gt_text: str | None = None
    if spec.gt_name:
        gt_path = GT_DIR / spec.gt_name
        if gt_path.exists():
            gt_text = gt_path.read_text(encoding="utf-8")

    if verbose:
        print(f"  [{spec.backend:<10}] {spec.pdf_name}", end="  ", flush=True)

    # 변환 실행 + 시간 측정
    t0 = time.perf_counter()
    try:
        pred_text = run_backend(spec.backend, pdf_path, output_path, api_key)
    except NotImplementedError as e:
        print(f"SKIP — {e}")
        result = BenchResult(
            backend=spec.backend, pdf_name=spec.pdf_name, mode=spec.mode
        )
        _save_metrics(result, result_dir)
        return result
    except Exception as e:
        print(f"ERROR — {e}")
        result = BenchResult(
            backend=spec.backend, pdf_name=spec.pdf_name, mode=spec.mode
        )
        _save_metrics(result, result_dir, error=str(e))
        return result

    speed_s = time.perf_counter() - t0

    # 평가
    result = evaluate_document(
        pred_text=pred_text,
        gt_text=gt_text,
        speed_s=speed_s,
        backend=spec.backend,
        pdf_name=spec.pdf_name,
        mode=spec.mode,
    )

    _save_metrics(result, result_dir)

    if verbose:
        ned_str = (
            f"NED={result.avg_edit_dist:.3f}"
            if result.avg_edit_dist is not None
            else "NED=  N/A"
        )
        teds_str = (
            f"TEDS-H={result.avg_teds_html:.3f}"
            if result.avg_teds_html is not None
            else "TEDS-H=  N/A"
        )
        print(f"{ned_str}  {teds_str}  {speed_s:.1f}s  {result.total_words}w")

    return result


def _save_metrics(
    result: BenchResult, result_dir: Path, error: str | None = None
) -> None:
    """metrics.json 원자적 기록."""

    def _omnidoc_dict(o) -> dict | None:
        if o is None:
            return None
        return {
            "edit_dist": o.edit_dist,
            "bleu": o.bleu,
            "meteor": o.meteor,
            "teds_html": o.teds_html,
        }

    data = {
        "backend": result.backend,
        "pdf_name": result.pdf_name,
        "mode": result.mode,
        "error": error,
        "pages": [
            {
                "page": p.page,
                "speed_s": round(p.speed_s, 3),
                "word_count": p.word_count,
                "has_headers": p.has_headers,
                "has_tables": p.has_tables,
                "has_formulas": p.has_formulas,
                "omnidoc": _omnidoc_dict(p.omnidoc),
            }
            for p in result.pages
        ],
        "summary": {
            "avg_edit_dist": result.avg_edit_dist,
            "avg_bleu": result.avg_bleu,
            "avg_meteor": result.avg_meteor,
            "avg_teds_html": result.avg_teds_html,
            "avg_speed_s": round(result.avg_speed, 3) if result.pages else None,
            "total_time_s": round(result.total_time_s, 3) if result.pages else None,
            "total_words": result.total_words,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = result_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(result_dir / "metrics.json")
    (result_dir / "DONE").touch()


# ── 전체 실행 루프 ────────────────────────────────────────────────────────────


def run_all(
    specs: list[BenchSpec],
    output_dir: Path,
    api_key: str | None = None,
    verbose: bool = True,
    delay_s: float = 0.0,
) -> list[BenchResult]:
    """스펙 목록을 순서대로 실행 후 결과 목록 반환."""
    results: list[BenchResult] = []
    total = len(specs)

    print(f"\n{'=' * 70}")
    print(f"벤치마크 시작 — {total}개 조합 | 출력: {output_dir}")
    if delay_s > 0:
        print(f"스펙 간 딜레이: {delay_s}s")
    print(f"{'=' * 70}\n")

    for idx, spec in enumerate(specs, 1):
        if verbose:
            print(f"[{idx:>3}/{total}] ", end="")
        results.append(run_spec(spec, output_dir, api_key=api_key, verbose=verbose))
        if delay_s > 0 and idx < total:
            time.sleep(delay_s)

    _print_summary(results)
    return results


def _print_summary(results: list[BenchResult]) -> None:
    """백엔드×문서유형 요약 테이블 출력."""
    print(f"\n{'=' * 100}")
    print("결과 요약")
    print(f"{'=' * 100}")
    print(
        f"{'백엔드':<12} {'PDF':<35} {'NED':>6} "
        f"{'BLEU':>6} {'METEOR':>7} {'TEDS-H':>7} {'속도':>7} {'단어':>6}"
    )
    print("-" * 100)
    for r in results:
        ned = f"{r.avg_edit_dist:.3f}" if r.avg_edit_dist is not None else "  N/A"
        bleu = f"{r.avg_bleu:.1f}" if r.avg_bleu is not None else "  N/A"
        meteor = f"{r.avg_meteor:.1f}" if r.avg_meteor is not None else "    N/A"
        teds_h = f"{r.avg_teds_html:.3f}" if r.avg_teds_html is not None else "    N/A"
        spd = f"{r.avg_speed:.2f}s"
        w = str(r.total_words)
        print(
            f"{r.backend:<12} {r.pdf_name:<35} {ned:>6} "
            f"{bleu:>6} {meteor:>7} {teds_h:>7} {spd:>7} {w:>6}"
        )
    print(f"{'=' * 100}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF 파서 로컬 벤치마크 러너",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m benchmark.runner --preset quick
  python -m benchmark.runner --preset phase1 --output ./bench_results
  python -m benchmark.runner --backend docling --pdf table_native.pdf
        """,
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=[
            "quick",
            "phase1",
            "phase2",
            "vlm",
            "upstage-only",
            "ocr",
            "deepseek",
            "vlm-all",
            "tables",
            "graphs",
        ],
        help="사전 정의된 조합 집합",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=[
            "pymupdf",
            "docling",
            "openai",
            "openai-4.1",
            "upstage",
            "upstage-enhanced",
            "paddleocr-vl",
            "deepseek-ocr2",
        ],
        help="단일 백엔드 지정 (--pdf와 함께 사용)",
    )
    parser.add_argument(
        "--pdf", default=None, help="단일 PDF 파일명 (--backend와 함께 사용)"
    )
    parser.add_argument(
        "--mode", default="direct", choices=["direct", "document", "hybrid"]
    )
    parser.add_argument(
        "--output", default="./bench_results", help="결과 저장 디렉토리"
    )
    parser.add_argument("--quiet", action="store_true", help="진행 출력 억제")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="스펙 간 대기 시간(초) — API rate limit 회피용 (예: --delay 5)",
    )
    args = parser.parse_args()

    api_key = None  # 오픈소스 백엔드는 API 키 불필요 (K8s 로컬 서비스)
    output_dir = Path(args.output) / datetime.now().strftime("%Y%m%d-%H%M")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 스펙 결정
    if args.preset:
        specs = get_preset(args.preset)
    elif args.backend and args.pdf:
        specs = [
            BenchSpec(
                backend=args.backend,
                pdf_name=args.pdf,
                mode=args.mode,
                gt_name=GT_MAP.get(args.pdf),
            )
        ]
    else:
        parser.error("--preset 또는 --backend + --pdf 를 지정하세요.")

    run_all(
        specs, output_dir, api_key=api_key, verbose=not args.quiet, delay_s=args.delay
    )


if __name__ == "__main__":
    main()
