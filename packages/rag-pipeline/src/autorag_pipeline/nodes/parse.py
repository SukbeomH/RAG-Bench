"""Parse and chunk nodes wrapping autorag_parsers."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from typing import Any

from autorag_pipeline.states.rag_state import RAGState


def parse_pdf(state: RAGState) -> dict[str, Any]:
    """Parse a PDF file using the specified backend.

    Reads ``pdf_path`` and ``backend`` from state, returns conversion
    metadata and the raw ConversionResult for downstream chunking.
    """
    from autorag_parsers import get_parser

    pdf_path: str = state["pdf_path"]
    backend: str = state.get("backend", "pymupdf")

    t0 = time.perf_counter()
    parser = get_parser(backend)
    result = parser.convert(pdf_path)
    elapsed = time.perf_counter() - t0

    doc_id = hashlib.sha256(pdf_path.encode()).hexdigest()[:12]
    pages = [asdict(p) for p in result.pages]

    return {
        "doc_id": doc_id,
        "pages": pages,
        "total_parse_time_s": elapsed,
        "_conversion_result": result,
    }


def chunk_document(state: RAGState) -> dict[str, Any]:
    """Chunk the parsed document into smaller pieces with provenance.

    Uses the ``_conversion_result`` produced by :func:`parse_pdf`.
    """
    from autorag_parsers import ChunkConfig
    from autorag_parsers import chunk_document as _chunk_doc

    result = state["_conversion_result"]
    chunk_size = state.get("chunk_size", 512)
    chunk_overlap = state.get("chunk_overlap", 64)

    cfg = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = _chunk_doc(result, config=cfg)

    return {"chunks": [asdict(c) for c in chunks]}
