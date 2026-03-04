"""Unit tests for the RAG pipeline graph structure."""

from __future__ import annotations



def test_rag_graph_compiles():
    """build_rag_pipeline() returns a compiled graph without errors."""
    from autorag_pipeline.graphs.rag_pipeline import build_rag_pipeline

    graph = build_rag_pipeline()
    assert graph is not None


def test_rag_graph_has_expected_nodes():
    """The compiled graph contains all expected node names."""
    from autorag_pipeline.graphs.rag_pipeline import build_rag_pipeline

    graph = build_rag_pipeline()

    # Get node names from the graph
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__",
        "__end__",
        "parse_pdf",
        "chunk_document",
        "build_and_index",
        "retrieve",
        "generate_answer",
    }
    assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"


def test_should_index_routes_to_build():
    """When index_ready is not set, route to build_and_index."""
    from autorag_pipeline.graphs.rag_pipeline import _should_index

    state = {"chunks": [{"chunk_text": "test"}]}
    assert _should_index(state) == "build_and_index"


def test_should_index_routes_to_retrieve():
    """When index_ready is True, route directly to retrieve."""
    from autorag_pipeline.graphs.rag_pipeline import _should_index

    state = {"index_ready": True}
    assert _should_index(state) == "retrieve"


def test_rag_graph_edge_structure():
    """Verify the graph has the correct edge topology."""
    from autorag_pipeline.graphs.rag_pipeline import build_rag_pipeline

    graph = build_rag_pipeline()
    drawable = graph.get_graph()

    # Check edges exist by collecting (source, target) pairs
    edges = {(e.source, e.target) for e in drawable.edges}

    assert ("__start__", "parse_pdf") in edges
    assert ("parse_pdf", "chunk_document") in edges
    assert ("build_and_index", "retrieve") in edges
    assert ("retrieve", "generate_answer") in edges
    assert ("generate_answer", "__end__") in edges
