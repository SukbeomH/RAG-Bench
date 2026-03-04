"""RAG benchmark Phase 2 (bench) nodes."""

from __future__ import annotations

from typing import Any

from autorag_pipeline.states.rag_bench_state import RAGBenchState


def run_benchmark(state: RAGBenchState) -> dict[str, Any]:
    """Run benchmark for each combo spec.

    Wraps the Phase 2 logic: build_strategy_from_spec → BenchmarkRunner.run().
    """
    from autorag_retrieval.combo.builder import build_strategy_from_spec
    from autorag_retrieval.combo.cache import IndexCacheManager
    from autorag_retrieval.combo.spec import ComboSpec
    from autorag_retrieval.runner import BenchmarkRunner

    combo_specs = state["combo_specs"]
    child_chunks = state["child_chunks"]
    parent_pairs = state["parent_pairs"]
    qa_pairs = state["qa_pairs"]
    enriched_chunks = state.get("enriched_chunks")

    queries = [qa["query"] for qa in qa_pairs]
    index_cache = IndexCacheManager()

    results: list[dict] = []
    for spec_dict in combo_specs:
        spec = ComboSpec(**spec_dict)
        strategy = build_strategy_from_spec(
            spec,
            index_cache,
            child_chunks,
            parent_pairs,
            pre_enriched=enriched_chunks,
        )

        runner = BenchmarkRunner(
            strategies=[strategy],
            queries=queries,
            k=3,
        )
        run_results = runner.run()
        results.append(
            {
                "spec_label": spec.label,
                "run_results": run_results,
            }
        )

    return {"bench_results": results}
