"""RAG benchmark Phase 1 (prep) nodes."""

from __future__ import annotations

from typing import Any

from autorag_pipeline.states.rag_bench_state import RAGBenchState


def load_hf_data(state: RAGBenchState) -> dict[str, Any]:
    """Load HuggingFace dataset and split into chunks/pairs/QA.

    Wraps the logic from worker_entrypoint.py's Phase 1:
    HFDatasetLoader → beir_to_parent_child_chunks.
    """
    from autorag_retrieval.data.loader import HFDatasetLoader
    from autorag_retrieval.data.processor import beir_to_parent_child_chunks

    category = state["category"]

    loader = HFDatasetLoader()
    dataset = loader.load(category)

    parent_pairs, child_chunks, qa_pairs = beir_to_parent_child_chunks(dataset)

    return {
        "child_chunks": child_chunks,
        "parent_pairs": parent_pairs,
        "qa_pairs": qa_pairs,
    }


def enrich_contextual(state: RAGBenchState) -> dict[str, Any]:
    """Enrich child chunks with contextual information using LLM.

    Wraps ContextualRetrievalStrategy.enrich_only().
    """
    from autorag_retrieval.strategies.contextual import ContextualRetrievalStrategy

    child_chunks = state["child_chunks"]
    contextual_llm = state.get("contextual_llm", "gpt-4o-mini")

    enriched = ContextualRetrievalStrategy.enrich_only(
        child_chunks,
        llm_model=contextual_llm,
    )

    return {
        "enriched_chunks": enriched,
        "prep_done": True,
    }
