"""
로컬 벤치마크 러너

backends/ 패키지(Phase 2)가 없어도 기존 category*.py로 동작하는
로컬 실행 루프. K8s 없이 빠른 검증용.

사용법:
    # 빠른 테스트
    python -m benchmark.runner --preset quick

    # Phase 1 전체 (기존 3 백엔드 × 11 PDF)
    python -m benchmark.runner --preset phase1 --output ./bench_results

    # 특정 조합
    python -m benchmark.runner --backend pymupdf --pdf text_only.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 패키지 루트를 sys.path에 추가 (pdf_parser/ 기준 실행 지원)
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 자동 로드 (python-dotenv 없어도 동작)
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

cert_path = "/Users/sukbeom/Documents/cert/combined-ca-bundle.pem"
if os.path.exists(cert_path):
    os.environ["SSL_CERT_FILE"] = cert_path
    os.environ["REQUESTS_CA_BUNDLE"] = cert_path

from benchmark.evaluator import BenchResult, evaluate_document
from benchmark.spec import Backend, BenchSpec, GT_MAP, get_preset


# ── 경로 설정 ─────────────────────────────────────────────────────────────────

BENCH_DIR = Path(__file__).parent.parent / "benchmark_pdfs"
GT_DIR = BENCH_DIR / "gt"


# ── 백엔드 디스패처 ───────────────────────────────────────────────────────────


def run_backend(
    backend: Backend, pdf_path: Path, output_path: Path, api_key: str | None
) -> str:
    """
    지정 백엔드로 PDF → Markdown 변환.

    반환: 변환된 마크다운 문자열
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "pymupdf":
        import category1_simple as cat

        return cat.convert_pdf(str(pdf_path), str(output_path))

    elif backend == "docling":
        import category2_medium as cat

        converter = cat.build_converter()
        return cat.convert_pdf(str(pdf_path), str(output_path), converter=converter)

    elif backend == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI 백엔드는 OPENAI_API_KEY 필요 (.env 또는 환경변수)")
        import category3_openai as cat

        pages = cat.convert_pdf(str(pdf_path), key, model="gpt-4o")
        cat.save_markdown(pages, str(output_path))
        return output_path.read_text(encoding="utf-8")

    elif backend == "openai-4.1":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI 백엔드는 OPENAI_API_KEY 필요 (.env 또는 환경변수)")
        import category3_openai as cat

        pages = cat.convert_pdf(str(pdf_path), key, model="gpt-4.1")
        cat.save_markdown(pages, str(output_path))
        return output_path.read_text(encoding="utf-8")

    elif backend == "upstage":
        key = os.environ.get("UPSTAGE_API_KEY")
        if not key:
            raise ValueError(
                "Upstage 백엔드는 UPSTAGE_API_KEY 필요 (.env 또는 환경변수)"
            )
        import category3_upstage as cat

        pages = cat.convert_pdf(str(pdf_path), key, mode="auto")
        cat.save_markdown(pages, str(output_path))
        return output_path.read_text(encoding="utf-8")

    elif backend == "upstage-enhanced":
        key = os.environ.get("UPSTAGE_API_KEY")
        if not key:
            raise ValueError(
                "Upstage 백엔드는 UPSTAGE_API_KEY 필요 (.env 또는 환경변수)"
            )
        import category3_upstage as cat

        pages = cat.convert_pdf(str(pdf_path), key, mode="enhanced")
        cat.save_markdown(pages, str(output_path))
        return output_path.read_text(encoding="utf-8")

    elif backend in ("paddleocr", "paddleocr-vl"):
        # 로컬 패들 파이프라인 호출 (subprocess Worker 이용)
        try:
            import importlib

            mod = importlib.import_module("backends.paddle_backend")
            return mod.convert_pdf(str(pdf_path), str(output_path))
        except ModuleNotFoundError:
            raise NotImplementedError(
                "backends.paddle_backend 패키지를 찾을 수 없습니다."
            )

    elif backend == "deepseek-ocr2":
        # 로컬 격리 venv subprocess 실행 (MPS/CUDA/CPU 자동 선택)
        # 사전 준비: bash pdf_parser/backends/setup_deepseek_venv.sh
        try:
            import importlib

            mod = importlib.import_module("backends.deepseek_ocr2_backend")
            return mod.convert_pdf(str(pdf_path), str(output_path))
        except ModuleNotFoundError:
            raise NotImplementedError(
                "backends.deepseek_ocr2_backend 패키지를 찾을 수 없습니다."
            )

    elif backend == "mineru":
        # Phase 2에서 backends/ 패키지 구현 예정
        try:
            import importlib

            mod = importlib.import_module("backends.mineru")
            return mod.convert_pdf(str(pdf_path), str(output_path))
        except ModuleNotFoundError:
            raise NotImplementedError(
                "'mineru' 백엔드는 Phase 2에서 구현 예정입니다. "
                "backends/ 패키지가 없습니다."
            )
    else:
        raise ValueError(f"알 수 없는 백엔드: {backend}")


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
            f"NED={result.avg_text_ned:.3f}"
            if result.pages[0].text_ned >= 0
            else "NED=  N/A"
        )
        teds_str = (
            f"TEDS={result.avg_table_teds:.3f}"
            if result.avg_table_teds >= 0
            else "TEDS=  N/A"
        )
        print(f"{ned_str}  {teds_str}  {speed_s:.1f}s  {result.total_words}w")

    return result


def _save_metrics(
    result: BenchResult, result_dir: Path, error: str | None = None
) -> None:
    """metrics.json 원자적 기록."""
    data = {
        "backend": result.backend,
        "pdf_name": result.pdf_name,
        "mode": result.mode,
        "error": error,
        "pages": [
            {
                "page": p.page,
                "text_ned": round(p.text_ned, 4) if p.text_ned >= 0 else None,
                "table_teds": round(p.table_teds, 4) if p.table_teds >= 0 else None,
                "speed_s": round(p.speed_s, 3),
                "word_count": p.word_count,
                "has_headers": p.has_headers,
                "has_tables": p.has_tables,
                "has_formulas": p.has_formulas,
            }
            for p in result.pages
        ],
        "summary": {
            "avg_text_ned": round(result.avg_text_ned, 4) if result.pages else None,
            "avg_table_teds": round(result.avg_table_teds, 4)
            if result.avg_table_teds >= 0
            else None,
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
    print(f"\n{'=' * 70}")
    print("결과 요약")
    print(f"{'=' * 70}")
    print(f"{'백엔드':<12} {'PDF':<35} {'NED':>6} {'TEDS':>6} {'속도':>7} {'단어':>6}")
    print("-" * 70)
    for r in results:
        ned = (
            f"{r.avg_text_ned:.3f}" if r.pages and r.pages[0].text_ned >= 0 else "  N/A"
        )
        teds = f"{r.avg_table_teds:.3f}" if r.avg_table_teds >= 0 else "  N/A"
        spd = f"{r.avg_speed:.2f}s"
        w = str(r.total_words)
        print(f"{r.backend:<12} {r.pdf_name:<35} {ned:>6} {teds:>6} {spd:>7} {w:>6}")
    print(f"{'=' * 70}\n")


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
            "mineru",
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

    run_all(specs, output_dir, api_key=api_key, verbose=not args.quiet, delay_s=args.delay)


if __name__ == "__main__":
    main()
