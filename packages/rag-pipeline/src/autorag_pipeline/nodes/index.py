"""Index node: build a DenseSparseStrategy vector index from chunks."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from autorag_pipeline.states.rag_state import RAGState


def build_and_index(state: RAGState) -> dict[str, Any]:
    """Convert chunks to LangChain Documents and index them.

    Creates an in-memory Qdrant index using DenseSparseStrategy.
    """
    from autorag_retrieval.strategies.dense_sparse import DenseSparseStrategy

    chunks: list[dict] = state["chunks"]

    documents = [
        Document(
            page_content=c["chunk_text"],
            metadata={
                "doc_id": c.get("doc_id", ""),
                "source_path": c.get("source_path", ""),
                "page_number": c.get("page_number", 0),
                "chunk_id": c.get("chunk_id", ""),
                "bbox": c.get("bbox"),
                "backend": c.get("backend", ""),
            },
        )
        for c in chunks
        if c.get("chunk_text", "").strip()
    ]

    strategy = DenseSparseStrategy(
        dense_model="e5",
        sparse_type="korean_bm25",
        qdrant_path=":memory:",
    )
    strategy.index(documents)

    return {
        "_strategy": strategy,
        "strategy_name": "e5+korean_bm25",
        "index_ready": True,
    }
