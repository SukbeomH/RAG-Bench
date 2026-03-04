"""RAG retrieval API router with citation support."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from autorag_api.routers.parse import get_parsed_doc
from autorag_api.schemas import (
    AskRequest,
    AskResponse,
    CitationItem,
    RetrieveRequest,
    RetrieveResponse,
)

router = APIRouter(prefix="/api", tags=["retrieve"])


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Retrieve relevant chunks with citation metadata."""
    if not req.doc_id:
        raise HTTPException(400, "doc_id is required")

    doc = get_parsed_doc(req.doc_id)
    if not doc:
        raise HTTPException(404, f"Document {req.doc_id} not found. Parse first.")

    chunks = doc.get("chunks", [])
    if not chunks:
        raise HTTPException(400, "No chunks available. Re-parse with chunk=true.")

    # Simple keyword search for now (replace with vector search when indexed)
    query_lower = req.query.lower()
    scored = []
    for c in chunks:
        text_lower = c.chunk_text.lower()
        # Simple relevance: count query term occurrences
        score = sum(1 for term in query_lower.split() if term in text_lower)
        if score > 0:
            scored.append((c, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = scored[: req.k]

    results = [
        CitationItem(
            chunk_id=c.chunk_id,
            source_path=c.source_path,
            page_number=c.page_number,
            text_snippet=c.chunk_text[:200],
            bbox=list(c.bbox) if c.bbox else None,
            relevance_score=float(score),
        )
        for c, score in top_k
    ]

    return RetrieveResponse(query=req.query, results=results)


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Answer a question with citations from parsed documents."""
    # Retrieve relevant chunks
    retrieve_resp = await retrieve(
        RetrieveRequest(query=req.query, doc_id=req.doc_id, k=req.k)
    )

    if not retrieve_resp.results:
        return AskResponse(
            answer="관련 문서를 찾을 수 없습니다.",
            citations=[],
        )

    # Build context from chunks
    context_parts = []
    for i, r in enumerate(retrieve_resp.results):
        context_parts.append(f"[{i + 1}] (p.{r.page_number}) {r.text_snippet}")

    context = "\n\n".join(context_parts)

    # For now, return a formatted answer referencing chunks
    # In production, this would call an LLM with the context
    answer = (
        f"다음 {len(retrieve_resp.results)}개 출처에서 관련 내용을 찾았습니다:\n\n"
        + "\n".join(
            f"[{i + 1}] {r.source_path} 페이지 {r.page_number}"
            for i, r in enumerate(retrieve_resp.results)
        )
    )

    return AskResponse(answer=answer, citations=retrieve_resp.results)
