"""Retrieve node: vector search over indexed documents."""

from __future__ import annotations

from typing import Any

from autorag_pipeline.states.rag_state import RAGState


def retrieve(state: RAGState) -> dict[str, Any]:
    """Retrieve relevant documents for the query using the indexed strategy."""
    strategy = state["_strategy"]
    query: str = state["query"]
    k: int = state.get("k", 5)

    docs = strategy.retrieve(query, k=k)

    context_parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source_path", "unknown")
        page = doc.metadata.get("page_number", "?")
        snippet = doc.page_content[:300]
        context_parts.append(f"[{i}] ({src} p.{page}) {snippet}")

    return {
        "retrieved_docs": docs,
        "context": "\n\n".join(context_parts),
    }
