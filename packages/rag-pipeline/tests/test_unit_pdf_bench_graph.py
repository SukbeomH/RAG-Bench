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
        "normalize_pdfs",
        "evaluate_pdfs",
        "collect_summary",
    }
    assert expected.issubset(node_names)


def test_pdf_bench_edge_structure():
    """Verify edge structure including conditional routing."""
    from autorag_pipeline.graphs.pdf_bench_graph import build_pdf_bench_graph

    graph = build_pdf_bench_graph()
    edges = {(e.source, e.target) for e in graph.get_graph().edges}

    # START → conditional (parse or normalize)
    assert ("__start__", "parse_all_pdfs") in edges
    assert ("__start__", "normalize_pdfs") in edges
    # parse → normalize → evaluate → collect → END
    assert ("parse_all_pdfs", "normalize_pdfs") in edges
    assert ("normalize_pdfs", "evaluate_pdfs") in edges
    assert ("evaluate_pdfs", "collect_summary") in edges
    assert ("collect_summary", "__end__") in edges

    # 조건부 엣지 확인
    conditional_edges = {
        (e.source, e.target) for e in graph.get_graph().edges if e.conditional
    }
    assert ("__start__", "parse_all_pdfs") in conditional_edges
    assert ("__start__", "normalize_pdfs") in conditional_edges


def test_should_parse_routing():
    """_should_parse returns correct node based on skip_parse flag."""
    from autorag_pipeline.graphs.pdf_bench_graph import _should_parse

    assert _should_parse({"skip_parse": False}) == "parse_all_pdfs"
    assert _should_parse({"skip_parse": True}) == "normalize_pdfs"
    assert _should_parse({}) == "parse_all_pdfs"  # default: parse
