"""Chunking with provenance metadata inheritance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from autorag_parsers._protocol import ConversionResult, PageResult
from autorag_parsers.provenance import ChunkProvenance


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    separator: str = "\n\n"


def _compute_bbox_for_chunk(
    chunk_text: str, bbox_data: list[dict] | None
) -> tuple[float, float, float, float] | None:
    """Find bounding box that covers the chunk text from word-level bbox data."""
    if not bbox_data:
        return None

    # Find words that appear in chunk
    chunk_words = set(re.findall(r"\S+", chunk_text.lower()))
    matching = [
        b for b in bbox_data if b.get("text", "").lower().strip() in chunk_words
    ]

    if not matching:
        return None

    x0 = min(b["x0"] for b in matching)
    y0 = min(b["y0"] for b in matching)
    x1 = max(b["x1"] for b in matching)
    y1 = max(b["y1"] for b in matching)
    return (x0, y0, x1, y1)


def _doc_id_from_path(source_path: str) -> str:
    return hashlib.sha256(source_path.encode()).hexdigest()[:12]


def chunk_page(
    page: PageResult,
    source_path: str,
    doc_id: str | None = None,
    config: ChunkConfig | None = None,
) -> list[ChunkProvenance]:
    """Split a single page into chunks with provenance metadata."""
    cfg = config or ChunkConfig()
    did = doc_id or _doc_id_from_path(source_path)

    text = page.markdown
    if not text.strip():
        return []

    # Split by separator, then merge into chunks
    paragraphs = text.split(cfg.separator)
    chunks: list[ChunkProvenance] = []
    current = ""
    chunk_idx = 0

    for para in paragraphs:
        if len(current) + len(para) + len(cfg.separator) > cfg.chunk_size and current:
            bbox = _compute_bbox_for_chunk(current, page.bbox_data)
            chunks.append(
                ChunkProvenance(
                    doc_id=did,
                    source_path=source_path,
                    page_number=page.page_num,
                    chunk_id=f"p{page.page_num}_c{chunk_idx}",
                    chunk_text=current.strip(),
                    bbox=bbox,
                    backend=page.backend,
                )
            )
            # Overlap: keep tail of current chunk
            if cfg.chunk_overlap > 0:
                current = current[-cfg.chunk_overlap :] + cfg.separator + para
            else:
                current = para
            chunk_idx += 1
        else:
            current = current + cfg.separator + para if current else para

    # Flush remaining
    if current.strip():
        bbox = _compute_bbox_for_chunk(current, page.bbox_data)
        chunks.append(
            ChunkProvenance(
                doc_id=did,
                source_path=source_path,
                page_number=page.page_num,
                chunk_id=f"p{page.page_num}_c{chunk_idx}",
                chunk_text=current.strip(),
                bbox=bbox,
                backend=page.backend,
            )
        )

    return chunks


def chunk_document(
    result: ConversionResult,
    config: ChunkConfig | None = None,
) -> list[ChunkProvenance]:
    """Chunk an entire document with provenance tracking."""
    doc_id = _doc_id_from_path(result.pdf_path)
    all_chunks: list[ChunkProvenance] = []

    for page in result.pages:
        all_chunks.extend(
            chunk_page(page, result.pdf_path, doc_id=doc_id, config=config)
        )

    return all_chunks
