"""PDF parser benchmark graph using LangGraph StateGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from autorag_pipeline.nodes.pdf_bench import (
    collect_summary,
    evaluate_pdfs,
    parse_all_pdfs,
)
from autorag_pipeline.states.pdf_bench_state import PDFBenchState


def build_pdf_bench_graph() -> StateGraph:
    """Build and compile the PDF parser benchmark graph.

    Graph flow::

        START → parse_all_pdfs → evaluate_pdfs → collect_summary → END
    """
    graph = StateGraph(PDFBenchState)

    graph.add_node("parse_all_pdfs", parse_all_pdfs)
    graph.add_node("evaluate_pdfs", evaluate_pdfs)
    graph.add_node("collect_summary", collect_summary)

    graph.add_edge(START, "parse_all_pdfs")
    graph.add_edge("parse_all_pdfs", "evaluate_pdfs")
    graph.add_edge("evaluate_pdfs", "collect_summary")
    graph.add_edge("collect_summary", END)

    return graph.compile()
