"""PDF parser benchmark graph using LangGraph StateGraph."""

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
    """skip_parse=True 시 파싱을 건너뛰고 정규화 노드로 직행."""
    if state.get("skip_parse", False):
        return "normalize_pdfs"
    return "parse_all_pdfs"


def build_pdf_bench_graph() -> StateGraph:
    """Build and compile the PDF parser benchmark graph.

    Graph flow::

        START → [skip_parse?] → parse_all_pdfs → normalize_pdfs → evaluate_pdfs → collect_summary → END
                      ↓
                 normalize_pdfs → evaluate_pdfs → collect_summary → END
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
