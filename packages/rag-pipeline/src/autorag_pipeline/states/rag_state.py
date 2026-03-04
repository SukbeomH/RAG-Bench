"""RAG pipeline state definition."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    """State for the basic RAG pipeline graph."""

    # --- Input ---
    pdf_path: str
    query: str
    backend: str  # parser backend name (e.g. "pymupdf")
    k: int
    chunk_size: int
    chunk_overlap: int

    # --- Parse output ---
    doc_id: str
    pages: list[dict]  # serialised PageResult dicts
    total_parse_time_s: float
    _conversion_result: Any  # ConversionResult (internal, not serialisable)

    # --- Chunk output ---
    chunks: list[dict]  # serialised ChunkProvenance dicts

    # --- Index output ---
    _strategy: Any  # DenseSparseStrategy instance (internal)
    strategy_name: str
    index_ready: bool

    # --- Retrieve output ---
    retrieved_docs: list[Document]
    context: str

    # --- Generate output ---
    answer: str
    citations: list[dict]

    # --- Error tracking (append-only) ---
    errors: Annotated[list[str], operator.add]
