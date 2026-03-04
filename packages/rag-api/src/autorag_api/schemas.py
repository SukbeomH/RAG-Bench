"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Parse ─────────────────────────────────────────────────────────────────────


class ParseRequest(BaseModel):
    backend: str = "pymupdf"
    chunk: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 64


class PageResponse(BaseModel):
    page_num: int
    markdown: str
    backend: str
    has_bbox: bool = False


class ParseResponse(BaseModel):
    doc_id: str
    pdf_path: str
    pages: list[PageResponse]
    total_time_s: float
    chunk_count: int = 0


# ── Index ─────────────────────────────────────────────────────────────────────


class IndexRequest(BaseModel):
    doc_id: str
    dense_model: str = "intfloat/multilingual-e5-large"
    sparse_type: str = "korean_bm25"


class IndexResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    strategy: str


# ── Retrieve / Ask ────────────────────────────────────────────────────────────


class RetrieveRequest(BaseModel):
    query: str
    doc_id: str | None = None
    k: int = 5


class CitationItem(BaseModel):
    chunk_id: str
    source_path: str
    page_number: int
    text_snippet: str
    bbox: list[float] | None = None  # [x0, y0, x1, y1] normalized
    relevance_score: float = 0.0


class RetrieveResponse(BaseModel):
    query: str
    results: list[CitationItem]


class AskRequest(BaseModel):
    query: str
    doc_id: str | None = None
    k: int = 5


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationItem] = Field(default_factory=list)


# ── Documents ─────────────────────────────────────────────────────────────────


class DocumentInfo(BaseModel):
    doc_id: str
    source_path: str
    page_count: int
    chunk_count: int
    backend: str
