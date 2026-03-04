"""Unit tests for pdf-parsers: registry, chunking, protocol."""

from __future__ import annotations

import hashlib

import pytest

from autorag_parsers import (
    ChunkConfig,
    ChunkProvenance,
    ConversionResult,
    PageResult,
    available_backends,
    chunk_document,
    chunk_page,
    get_parser,
)
from autorag_parsers.registry import _REGISTRY, register


# ── Registry ─────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_pymupdf_registered(self):
        assert "pymupdf" in available_backends()

    def test_get_parser_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="no_such_backend"):
            get_parser("no_such_backend")

    def test_register_decorator(self):
        @register("__test_dummy__")
        class DummyParser:
            pass

        assert "__test_dummy__" in _REGISTRY
        del _REGISTRY["__test_dummy__"]

    def test_available_backends_returns_list(self):
        result = available_backends()
        assert isinstance(result, list)
        assert len(result) >= 1


# ── ChunkConfig ──────────────────────────────────────────────────────────────


class TestChunkConfig:
    def test_defaults(self):
        cfg = ChunkConfig()
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64
        assert cfg.separator == "\n\n"

    def test_custom_values(self):
        cfg = ChunkConfig(chunk_size=256, chunk_overlap=32, separator="\n")
        assert cfg.chunk_size == 256
        assert cfg.chunk_overlap == 32
        assert cfg.separator == "\n"


# ── chunk_page ───────────────────────────────────────────────────────────────


def _make_page(text: str, page_num: int = 1) -> PageResult:
    return PageResult(page_num=page_num, markdown=text, backend="test")


class TestChunkPage:
    def test_empty_page_returns_no_chunks(self):
        page = _make_page("   ")
        assert chunk_page(page, "test.pdf") == []

    def test_single_short_paragraph(self):
        page = _make_page("Hello world")
        chunks = chunk_page(page, "test.pdf")
        assert len(chunks) == 1
        assert chunks[0].chunk_text == "Hello world"
        assert chunks[0].chunk_id == "p1_c0"

    def test_chunk_provenance_fields(self):
        page = _make_page("Some text here")
        chunks = chunk_page(page, "/path/to/test.pdf")
        c = chunks[0]
        assert isinstance(c, ChunkProvenance)
        expected_doc_id = hashlib.sha256(b"/path/to/test.pdf").hexdigest()[:12]
        assert c.doc_id == expected_doc_id
        assert c.source_path == "/path/to/test.pdf"
        assert c.page_number == 1
        assert c.backend == "test"

    def test_multiple_chunks_with_overflow(self):
        # 512자보다 긴 텍스트 → 2개 이상 chunk
        para = "word " * 60  # ~300 chars per paragraph
        text = f"{para}\n\n{para}\n\n{para}"
        page = _make_page(text)
        chunks = chunk_page(page, "test.pdf", config=ChunkConfig(chunk_size=400))
        assert len(chunks) >= 2

    def test_custom_doc_id(self):
        page = _make_page("text")
        chunks = chunk_page(page, "test.pdf", doc_id="custom123")
        assert chunks[0].doc_id == "custom123"


# ── chunk_document ───────────────────────────────────────────────────────────


class TestChunkDocument:
    def test_multi_page_document(self):
        pages = [
            _make_page("Page one content", page_num=1),
            _make_page("Page two content", page_num=2),
        ]
        result = ConversionResult(pdf_path="doc.pdf", pages=pages)
        chunks = chunk_document(result)
        assert len(chunks) == 2
        assert chunks[0].page_number == 1
        assert chunks[1].page_number == 2

    def test_empty_pages_skipped(self):
        pages = [
            _make_page("Content", page_num=1),
            _make_page("   ", page_num=2),
        ]
        result = ConversionResult(pdf_path="doc.pdf", pages=pages)
        chunks = chunk_document(result)
        assert len(chunks) == 1
