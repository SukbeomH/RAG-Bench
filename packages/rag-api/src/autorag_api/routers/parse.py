"""PDF parsing API router."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from autorag_parsers import ChunkConfig, available_backends, chunk_document, get_parser

from autorag_api.schemas import PageResponse, ParseResponse

router = APIRouter(prefix="/api", tags=["parse"])

# In-memory store for parsed documents (replace with DB in production)
_PARSED_DOCS: dict[str, dict] = {}


@router.post("/parse", response_model=ParseResponse)
async def parse_pdf(
    file: UploadFile = File(...),
    backend: str = Form("pymupdf"),
    chunk: bool = Form(True),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(64),
) -> ParseResponse:
    """Upload and parse a PDF file."""
    # Save uploaded file to temp location
    suffix = Path(file.filename or "upload.pdf").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        try:
            parser = get_parser(backend)
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown backend '{backend}'. Available: {available_backends()}",
            )
        result = parser.convert(tmp_path)

        doc_id = hashlib.sha256(tmp_path.encode()).hexdigest()[:12]

        # Chunk if requested
        chunk_count = 0
        chunks = []
        if chunk:
            chunks = chunk_document(
                result, ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            )
            chunk_count = len(chunks)

        # Store for later indexing
        _PARSED_DOCS[doc_id] = {
            "result": result,
            "chunks": chunks,
            "source_path": file.filename or "upload.pdf",
        }

        pages = [
            PageResponse(
                page_num=p.page_num,
                markdown=p.markdown,
                backend=p.backend,
                has_bbox=p.bbox_data is not None,
            )
            for p in result.pages
        ]

        return ParseResponse(
            doc_id=doc_id,
            pdf_path=file.filename or "upload.pdf",
            pages=pages,
            total_time_s=result.total_time_s,
            chunk_count=chunk_count,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def get_parsed_doc(doc_id: str) -> dict | None:
    """Access parsed document store (used by other routers)."""
    return _PARSED_DOCS.get(doc_id)
