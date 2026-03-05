"""PDF parser benchmark nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autorag_pipeline.states.pdf_bench_state import PDFBenchState


def parse_all_pdfs(state: PDFBenchState) -> dict[str, Any]:
    """Parse all PDFs defined in specs using their respective backends.

    Wraps autorag_pdf_eval.runner.run_spec() for each spec.
    """
    from autorag_pdf_eval.runner import run_spec
    from autorag_pdf_eval.spec import BenchSpec

    specs = state["specs"]
    results_dir = state.get("results_dir", "/tmp/pdf_bench_results")

    parse_results: list[dict] = []
    for spec_dict in specs:
        spec = BenchSpec(**spec_dict)
        try:
            result = run_spec(spec, output_dir=results_dir)
            parse_results.append(
                {
                    "label": spec.label,
                    "backend": spec.backend,
                    "pdf_name": spec.pdf_name,
                    "total_time_s": result.total_time_s,
                    "total_words": result.total_words,
                    "pages": [
                        {
                            "page": ps.page,
                            "speed_s": ps.speed_s,
                            "word_count": ps.word_count,
                        }
                        for ps in result.pages
                    ],
                }
            )
        except Exception as e:
            parse_results.append(
                {
                    "label": spec.label,
                    "backend": spec.backend,
                    "pdf_name": spec.pdf_name,
                    "error": str(e),
                }
            )

    return {"parse_results": parse_results}


def normalize_pdfs(state: PDFBenchState) -> dict[str, Any]:
    """기존 output.md에 정규화 재적용 + 재평가.

    skip_parse=True 시 파싱을 건너뛰고 이 노드에서
    기존 결과의 output.md를 읽어 정규화 → 재평가합니다.
    skip_parse=False 시에도 동일하게 정규화가 적용됩니다
    (run_spec 내부에서 이미 처리되므로 결과만 수집).
    """
    from autorag_pdf_eval.runner import reeval_dir

    results_dir = Path(state.get("results_dir", "/tmp/pdf_bench_results"))
    skip_parse = state.get("skip_parse", False)

    normalize_results: list[dict] = []

    if skip_parse:
        # 파싱 없이 기존 output.md에 정규화 재적용
        results = reeval_dir(results_dir, verbose=True)
        for r in results:
            normalize_results.append(
                {
                    "backend": r.backend,
                    "pdf_name": r.pdf_name,
                    "avg_edit_dist": r.avg_edit_dist,
                    "avg_teds_html": r.avg_teds_html,
                    "total_words": r.total_words,
                }
            )
    else:
        # 파싱 후 정규화는 run_spec 내부에서 이미 처리됨 — parse_results 전달
        for pr in state.get("parse_results", []):
            normalize_results.append(
                {
                    "backend": pr.get("backend", ""),
                    "pdf_name": pr.get("pdf_name", ""),
                    "label": pr.get("label", ""),
                }
            )

    return {"normalize_results": normalize_results}


def evaluate_pdfs(state: PDFBenchState) -> dict[str, Any]:
    """Evaluate parsed PDF results using NED/TEDS metrics.

    Reads metrics.json files produced by parse_all_pdfs or normalize_pdfs.
    """
    from autorag_pdf_eval.report import (
        compute_backend_averages,
        compute_weighted_scores,
        load_results,
    )

    results_dir = state.get("results_dir", "/tmp/pdf_bench_results")
    metrics = load_results([Path(results_dir)])
    backend_avgs = compute_backend_averages(metrics)
    weighted_scores = compute_weighted_scores(backend_avgs)

    eval_results: list[dict] = []
    for backend, scores in weighted_scores.items():
        eval_results.append(
            {
                "backend": backend,
                "weighted_total": scores.get("total", 0),
                "averages": backend_avgs.get(backend, {}),
            }
        )

    return {"eval_results": eval_results}


def collect_summary(state: PDFBenchState) -> dict[str, Any]:
    """Generate a summary string from evaluation results."""
    eval_results = state.get("eval_results", [])

    sorted_results = sorted(
        eval_results, key=lambda x: x.get("weighted_total", 0), reverse=True
    )

    lines = ["# PDF Parser Benchmark Summary", ""]
    for i, r in enumerate(sorted_results, 1):
        lines.append(
            f"{i}. **{r['backend']}** — weighted score: {r.get('weighted_total', 0):.2f}"
        )

    return {"summary": "\n".join(lines)}
