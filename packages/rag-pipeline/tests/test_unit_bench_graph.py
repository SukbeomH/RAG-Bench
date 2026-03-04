"""Unit tests for the RAG benchmark graph structure."""

from __future__ import annotations


def test_rag_bench_graph_compiles():
    """build_rag_bench_graph() returns a compiled graph."""
    from autorag_pipeline.graphs.rag_bench_graph import build_rag_bench_graph

    graph = build_rag_bench_graph()
    assert graph is not None


def test_rag_bench_graph_has_expected_nodes():
    """The benchmark graph contains all expected nodes."""
    from autorag_pipeline.graphs.rag_bench_graph import build_rag_bench_graph

    graph = build_rag_bench_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__",
        "__end__",
        "load_hf_data",
        "enrich_contextual",
        "run_benchmark",
        "evaluate_ragas",
    }
    assert expected.issubset(node_names)


def test_should_prep_routes_to_load():
    """When prep_done is not set, route to load_hf_data."""
    from autorag_pipeline.graphs.rag_bench_graph import _should_prep

    assert _should_prep({}) == "load_hf_data"


def test_should_prep_routes_to_run():
    """When prep_done is True, route to run_benchmark."""
    from autorag_pipeline.graphs.rag_bench_graph import _should_prep

    assert _should_prep({"prep_done": True}) == "run_benchmark"


def test_rag_bench_edge_structure():
    """Verify the benchmark graph has correct edges."""
    from autorag_pipeline.graphs.rag_bench_graph import build_rag_bench_graph

    graph = build_rag_bench_graph()
    edges = {(e.source, e.target) for e in graph.get_graph().edges}

    assert ("load_hf_data", "enrich_contextual") in edges
    assert ("enrich_contextual", "run_benchmark") in edges
    assert ("run_benchmark", "evaluate_ragas") in edges
    assert ("evaluate_ragas", "__end__") in edges
