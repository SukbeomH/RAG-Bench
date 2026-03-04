"""E2E tests for autorag_parsers — PDF parsing + chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from autorag_parsers import (
    ChunkConfig,
    ChunkProvenance,
    ConversionResult,
    PDFParser,
    available_backends,
    chunk_document,
    get_parser,
)


class TestPyMuPDFParse:
    def test_parse_text_only(self, sample_pdf: Path) -> None:
        parser = get_parser("pymupdf")
        result = parser.convert(str(sample_pdf))

        assert isinstance(result, ConversionResult)
        assert len(result.pages) > 0
        assert result.full_markdown.strip(), "markdown should not be empty"
        assert result.total_time_s >= 0

    def test_parse_table(self, table_pdf: Path) -> None:
        parser = get_parser("pymupdf")
        result = parser.convert(str(table_pdf))

        assert len(result.pages) > 0
        md = result.full_markdown
        assert "|" in md, "table markdown should contain pipe characters"


class TestChunking:
    def test_chunk_document_provenance(self, sample_pdf: Path) -> None:
        parser = get_parser("pymupdf")
        result = parser.convert(str(sample_pdf))
        chunks = chunk_document(result)

        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, ChunkProvenance)
            assert c.doc_id, "doc_id should not be empty"
            assert c.page_number >= 0
            assert c.chunk_id, "chunk_id should not be empty"
            assert "_" in c.chunk_id, "chunk_id should be like p0_c0"
            assert c.chunk_text.strip(), "chunk_text should not be empty"

    def test_chunk_config_params(self, sample_pdf: Path) -> None:
        parser = get_parser("pymupdf")
        result = parser.convert(str(sample_pdf))

        small = ChunkConfig(chunk_size=128, chunk_overlap=16)
        large = ChunkConfig(chunk_size=1024, chunk_overlap=64)

        chunks_small = chunk_document(result, config=small)
        chunks_large = chunk_document(result, config=large)

        assert len(chunks_small) >= len(chunks_large), (
            "smaller chunk_size should produce more chunks"
        )


class TestRegistry:
    def test_get_parser_pymupdf(self) -> None:
        parser = get_parser("pymupdf")
        assert isinstance(parser, PDFParser)
        assert parser.name == "pymupdf"

    def test_available_backends(self) -> None:
        backends = available_backends()
        assert isinstance(backends, list)
        assert "pymupdf" in backends

    def test_get_parser_invalid_raises(self) -> None:
        with pytest.raises((KeyError, ValueError)):
            get_parser("nonexistent_backend_xyz")
