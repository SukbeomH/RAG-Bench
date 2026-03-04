"""Unit tests for pipeline nodes with mocked dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch


# --- parse node tests ---


@dataclass
class _FakePageResult:
    page_num: int = 1
    markdown: str = "Hello world"
    backend: str = "pymupdf"
    bbox_data: list | None = None
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class _FakeConversionResult:
    pdf_path: str = "/tmp/test.pdf"
    pages: list | None = None
    total_time_s: float = 0.5
    metadata: dict | None = None

    def __post_init__(self):
        if self.pages is None:
            self.pages = [_FakePageResult()]
        if self.metadata is None:
            self.metadata = {}


@dataclass
class _FakeChunkProvenance:
    doc_id: str = "abc123"
    source_path: str = "/tmp/test.pdf"
    page_number: int = 1
    chunk_id: str = "p1_c0"
    chunk_text: str = "Hello world"
    bbox: tuple | None = None
    backend: str = "pymupdf"
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def test_parse_pdf_node():
    """parse_pdf node calls get_parser and returns expected keys."""
    fake_result = _FakeConversionResult()
    mock_parser = MagicMock()
    mock_parser.convert.return_value = fake_result

    with patch("autorag_parsers.get_parser", return_value=mock_parser):
        from autorag_pipeline.nodes.parse import parse_pdf

        state = {"pdf_path": "/tmp/test.pdf", "backend": "pymupdf"}
        result = parse_pdf(state)

    assert "doc_id" in result
    assert "pages" in result
    assert "total_parse_time_s" in result
    assert "_conversion_result" in result
    assert isinstance(result["pages"], list)


def test_chunk_document_node():
    """chunk_document node chunks the conversion result."""
    fake_chunk = _FakeChunkProvenance()
    fake_result = _FakeConversionResult()

    with patch("autorag_parsers.chunk_document", return_value=[fake_chunk]) as mock_cd:
        from autorag_pipeline.nodes.parse import chunk_document

        state = {
            "_conversion_result": fake_result,
            "chunk_size": 256,
            "chunk_overlap": 32,
        }
        result = chunk_document(state)

    assert "chunks" in result
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["chunk_text"] == "Hello world"


# --- index node tests ---


def test_build_and_index_node():
    """build_and_index creates strategy and sets index_ready."""
    mock_strategy = MagicMock()

    chunks = [
        {
            "chunk_text": "테스트 텍스트",
            "doc_id": "abc",
            "source_path": "/tmp/test.pdf",
            "page_number": 1,
            "chunk_id": "p1_c0",
            "bbox": None,
            "backend": "pymupdf",
        }
    ]

    with patch(
        "autorag_retrieval.strategies.dense_sparse.DenseSparseStrategy"
    ) as MockStrategy:
        MockStrategy.return_value = mock_strategy
        from autorag_pipeline.nodes.index import build_and_index

        state = {"chunks": chunks}
        result = build_and_index(state)

    assert result["index_ready"] is True
    assert result["strategy_name"] == "e5+korean_bm25"
    assert result["_strategy"] is mock_strategy


# --- retrieve node tests ---


def test_retrieve_node():
    """retrieve node returns docs and context string."""
    from langchain_core.documents import Document

    mock_doc = Document(
        page_content="관련 텍스트 내용",
        metadata={
            "source_path": "/tmp/test.pdf",
            "page_number": 1,
            "chunk_id": "p1_c0",
        },
    )
    mock_strategy = MagicMock()
    mock_strategy.retrieve.return_value = [mock_doc]

    from autorag_pipeline.nodes.retrieve import retrieve

    state = {"_strategy": mock_strategy, "query": "질문", "k": 3}
    result = retrieve(state)

    assert "retrieved_docs" in result
    assert "context" in result
    assert len(result["retrieved_docs"]) == 1
    mock_strategy.retrieve.assert_called_once_with("질문", k=3)


# --- generate node tests ---


def test_generate_answer_node():
    """generate_answer node calls LLM and returns answer + citations."""
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="테스트 답변입니다.")

    mock_doc = Document(
        page_content="컨텍스트 텍스트",
        metadata={
            "chunk_id": "p1_c0",
            "source_path": "/tmp/test.pdf",
            "page_number": 1,
            "bbox": None,
        },
    )

    with patch("autorag_retrieval.config.make_llm", return_value=mock_llm):
        from autorag_pipeline.nodes.generate import generate_answer

        state = {
            "query": "테스트 질문",
            "context": "[1] 컨텍스트 텍스트",
            "retrieved_docs": [mock_doc],
        }
        result = generate_answer(state)

    assert result["answer"] == "테스트 답변입니다."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "p1_c0"
