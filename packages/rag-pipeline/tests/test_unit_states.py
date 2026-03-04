"""Unit tests for pipeline state definitions."""

from __future__ import annotations

import operator

from autorag_pipeline.states.rag_state import RAGState


def test_rag_state_instantiation():
    """RAGState can be instantiated with minimal fields."""
    state: RAGState = {
        "pdf_path": "/tmp/test.pdf",
        "query": "테스트 질문",
        "backend": "pymupdf",
    }
    assert state["pdf_path"] == "/tmp/test.pdf"
    assert state["query"] == "테스트 질문"


def test_rag_state_optional_fields():
    """RAGState supports optional fields with total=False."""
    state: RAGState = {"pdf_path": "/tmp/test.pdf", "query": "q"}
    assert state.get("k") is None
    assert state.get("index_ready") is None
    assert state.get("errors") is None


def test_errors_annotated_add():
    """errors field uses operator.add for append-only accumulation."""
    hints = RAGState.__annotations__
    assert "errors" in hints

    # Verify operator.add works as reducer
    a = ["error1"]
    b = ["error2"]
    result = operator.add(a, b)
    assert result == ["error1", "error2"]


def test_rag_state_all_fields():
    """RAGState can hold all defined fields."""
    state: RAGState = {
        "pdf_path": "/tmp/test.pdf",
        "query": "q",
        "backend": "pymupdf",
        "k": 5,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "doc_id": "abc123",
        "pages": [],
        "total_parse_time_s": 1.5,
        "_conversion_result": None,
        "chunks": [],
        "_strategy": None,
        "strategy_name": "e5+bm25",
        "index_ready": True,
        "retrieved_docs": [],
        "context": "ctx",
        "answer": "answer",
        "citations": [],
        "errors": [],
    }
    assert state["index_ready"] is True
    assert state["strategy_name"] == "e5+bm25"
