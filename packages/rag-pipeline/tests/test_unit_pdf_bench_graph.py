"""Unit tests for the PDF parser benchmark graph structure."""

from __future__ import annotations


def test_pdf_bench_graph_compiles():
    """build_pdf_bench_graph() returns a compiled graph."""
    from autorag_pipeline.graphs.pdf_bench_graph import build_pdf_bench_graph

    graph = build_pdf_bench_graph()
    assert graph is not None


def test_pdf_bench_graph_has_expected_nodes():
    """The PDF benchmark graph contains all expected nodes."""
    from autorag_pipeline.graphs.pdf_bench_graph import build_pdf_bench_graph

    graph = build_pdf_bench_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__",
        "__end__",
        "parse_all_pdfs",
        "evaluate_pdfs",
        "collect_summary",
    }
    assert expected.issubset(node_names)


def test_pdf_bench_edge_structure():
    """Verify sequential edge structure."""
    from autorag_pipeline.graphs.pdf_bench_graph import build_pdf_bench_graph

    graph = build_pdf_bench_graph()
    edges = {(e.source, e.target) for e in graph.get_graph().edges}

    assert ("__start__", "parse_all_pdfs") in edges
    assert ("parse_all_pdfs", "evaluate_pdfs") in edges
    assert ("evaluate_pdfs", "collect_summary") in edges
    assert ("collect_summary", "__end__") in edges
