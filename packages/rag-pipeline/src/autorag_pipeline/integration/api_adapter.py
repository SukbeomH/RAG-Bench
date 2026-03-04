"""FastAPI v2 router using LangGraph pipeline."""

from __future__ import annotations


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v2", tags=["pipeline"])


class PipelineAskRequest(BaseModel):
    """Request body for the v2 /ask endpoint."""

    pdf_path: str
    query: str
    backend: str = "pymupdf"
    k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64


class CitationResponse(BaseModel):
    chunk_id: str = ""
    source_path: str = ""
    page_number: int = 0
    text_snippet: str = ""
    bbox: list[float] | None = None
    relevance_score: float = 0.0


class PipelineAskResponse(BaseModel):
    """Response from the v2 /ask endpoint."""

    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    doc_id: str = ""
    total_parse_time_s: float = 0.0


@router.post("/ask", response_model=PipelineAskResponse)
async def ask_v2(req: PipelineAskRequest) -> PipelineAskResponse:
    """Process a query through the full LangGraph RAG pipeline.

    Parses the PDF, chunks, indexes, retrieves, and generates an answer
    in a single request using the LangGraph state graph.
    """
    from autorag_pipeline.graphs.rag_pipeline import build_rag_pipeline

    graph = build_rag_pipeline()

    initial_state = {
        "pdf_path": req.pdf_path,
        "query": req.query,
        "backend": req.backend,
        "k": req.k,
        "chunk_size": req.chunk_size,
        "chunk_overlap": req.chunk_overlap,
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    citations = [CitationResponse(**c) for c in result.get("citations", [])]

    return PipelineAskResponse(
        answer=result.get("answer", ""),
        citations=citations,
        doc_id=result.get("doc_id", ""),
        total_parse_time_s=result.get("total_parse_time_s", 0.0),
    )
