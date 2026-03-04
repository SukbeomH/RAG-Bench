"""Citation provenance — metadata tracking through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkProvenance:
    """Tracks the origin of a text chunk back to its source PDF location."""

    doc_id: str
    source_path: str
    page_number: int
    chunk_id: str  # e.g. "p3_c2"
    chunk_text: str = ""
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1) normalized
    backend: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Citation:
    """A single citation reference in a RAG answer."""

    chunk_id: str
    source_path: str
    page_number: int
    text_snippet: str
    bbox: list[float] | None = None  # [x0, y0, x1, y1] normalized
    relevance_score: float = 0.0


@dataclass
class CitedAnswer:
    """RAG answer with citation references."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
