"""RAG benchmark graph using LangGraph StateGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from autorag_pipeline.nodes.bench_eval import evaluate_ragas
from autorag_pipeline.nodes.bench_prep import enrich_contextual, load_hf_data
from autorag_pipeline.nodes.bench_run import run_benchmark
from autorag_pipeline.states.rag_bench_state import RAGBenchState


def _should_prep(state: RAGBenchState) -> str:
    """Route: skip prep if data is already loaded."""
    if state.get("prep_done"):
        return "run_benchmark"
    return "load_hf_data"


def build_rag_bench_graph() -> StateGraph:
    """Build and compile the RAG benchmark pipeline graph.

    Graph flow::

        START → [conditional: prep_done?]
          → load_hf_data → enrich_contextual → run_benchmark → evaluate_ragas → END
          → run_benchmark (skip prep) → evaluate_ragas → END
    """
    graph = StateGraph(RAGBenchState)

    graph.add_node("load_hf_data", load_hf_data)
    graph.add_node("enrich_contextual", enrich_contextual)
    graph.add_node("run_benchmark", run_benchmark)
    graph.add_node("evaluate_ragas", evaluate_ragas)

    graph.add_conditional_edges(
        START,
        _should_prep,
        {"load_hf_data": "load_hf_data", "run_benchmark": "run_benchmark"},
    )
    graph.add_edge("load_hf_data", "enrich_contextual")
    graph.add_edge("enrich_contextual", "run_benchmark")
    graph.add_edge("run_benchmark", "evaluate_ragas")
    graph.add_edge("evaluate_ragas", END)

    return graph.compile()
