"""RAG benchmark pipeline state definition."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class RAGBenchState(TypedDict, total=False):
    """State for the RAG benchmark graph."""

    # --- Config ---
    category: str  # general, legal, business, medical, technical
    combo_specs: list[dict]  # serialised ComboSpec list
    preset: str
    results_dir: str
    contextual_llm: str

    # --- Phase 1 (prep) output ---
    child_chunks: list[Document]
    parent_pairs: list[tuple[str, Document]]
    qa_pairs: list[dict]
    enriched_chunks: list[Document]
    prep_done: bool

    # --- Phase 2 (bench) output ---
    bench_results: list[dict]
    bench_done: bool

    # --- Error tracking ---
    errors: Annotated[list[str], operator.add]
