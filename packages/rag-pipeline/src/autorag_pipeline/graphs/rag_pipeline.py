"""Basic RAG pipeline graph using LangGraph StateGraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from autorag_pipeline.nodes.generate import generate_answer
from autorag_pipeline.nodes.index import build_and_index
from autorag_pipeline.nodes.parse import chunk_document, parse_pdf
from autorag_pipeline.nodes.retrieve import retrieve
from autorag_pipeline.states.rag_state import RAGState


def _should_index(state: RAGState) -> str:
    """Route: skip indexing if strategy is already available."""
    if state.get("index_ready"):
        return "retrieve"
    return "build_and_index"


def build_rag_pipeline() -> StateGraph:
    """Build and compile the basic RAG pipeline graph.

    Graph flow::

        START → parse_pdf → chunk_document → [conditional]
          → build_and_index → retrieve → generate_answer → END
          → retrieve (skip index) → generate_answer → END
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("parse_pdf", parse_pdf)
    graph.add_node("chunk_document", chunk_document)
    graph.add_node("build_and_index", build_and_index)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)

    # Edges
    graph.add_edge(START, "parse_pdf")
    graph.add_edge("parse_pdf", "chunk_document")
    graph.add_conditional_edges(
        "chunk_document",
        _should_index,
        {"build_and_index": "build_and_index", "retrieve": "retrieve"},
    )
    graph.add_edge("build_and_index", "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
