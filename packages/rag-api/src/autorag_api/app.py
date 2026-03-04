"""FastAPI application — AutoRAG API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autorag_api.routers import parse, retrieve

app = FastAPI(
    title="AutoRAG API",
    description="PDF parsing, RAG retrieval, and citation pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(parse.router)
app.include_router(retrieve.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
