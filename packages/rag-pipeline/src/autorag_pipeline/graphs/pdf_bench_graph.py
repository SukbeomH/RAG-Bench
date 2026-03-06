"""PDF 파서 벤치마크 파이프라인 — LangGraph StateGraph.

PDF 파서 백엔드별 정확도를 OmniDocBench 메트릭으로 정량 평가.
5종 백엔드 × N개 PDF를 파싱 → 정규화 → NED/TEDS 평가 → 요약 보고.

노드 역할:
  - parse_all_pdfs: BenchSpec에 정의된 모든 (백엔드, PDF) 조합을 파싱 (autorag_pdf_eval.runner)
  - normalize_pdfs: 파싱 결과에 마크다운 정규화 적용 (normalize.py 7개 규칙).
      skip_parse=True 시 기존 output.md에 정규화만 재적용.
  - evaluate_pdfs: NED, BLEU, METEOR, TEDS-HTML 메트릭 산출 + 가중 점수 계산
  - collect_summary: 백엔드별 종합 순위를 마크다운 요약으로 생성

사용 예::

    from autorag_pipeline.graphs.pdf_bench_graph import build_pdf_bench_graph

    graph = build_pdf_bench_graph()

    # 전체 벤치마크 (파싱 + 정규화 + 평가)
    result = graph.invoke({"specs": [...], "results_dir": "./bench_results"})

    # 기존 결과에 정규화만 재적용 (reeval)
    result = graph.invoke({"results_dir": "./bench_results", "skip_parse": True})
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from autorag_pipeline.nodes.pdf_bench import (
    collect_summary,
    evaluate_pdfs,
    normalize_pdfs,
    parse_all_pdfs,
)
from autorag_pipeline.states.pdf_bench_state import PDFBenchState


def _should_parse(state: PDFBenchState) -> str:
    """파싱 여부 분기. skip_parse=True면 정규화 노드로 직행 (reeval 모드)."""
    if state.get("skip_parse", False):
        return "normalize_pdfs"
    return "parse_all_pdfs"


def build_pdf_bench_graph() -> StateGraph:
    """PDF 파서 벤치마크 그래프를 빌드+컴파일.

    Graph flow::

        START → [skip_parse?]
          ├─ False → parse_all_pdfs → normalize_pdfs → evaluate_pdfs → collect_summary → END
          └─ True  → normalize_pdfs → evaluate_pdfs → collect_summary → END
    """
    graph = StateGraph(PDFBenchState)

    graph.add_node("parse_all_pdfs", parse_all_pdfs)
    graph.add_node("normalize_pdfs", normalize_pdfs)
    graph.add_node("evaluate_pdfs", evaluate_pdfs)
    graph.add_node("collect_summary", collect_summary)

    graph.add_conditional_edges(
        START,
        _should_parse,
        {
            "parse_all_pdfs": "parse_all_pdfs",
            "normalize_pdfs": "normalize_pdfs",
        },
    )
    graph.add_edge("parse_all_pdfs", "normalize_pdfs")
    graph.add_edge("normalize_pdfs", "evaluate_pdfs")
    graph.add_edge("evaluate_pdfs", "collect_summary")
    graph.add_edge("collect_summary", END)

    return graph.compile()
