"""Unit tests for rag-api: schemas, parse error handling."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from autorag_api.routers.parse import router
from autorag_api.schemas import (
    AskRequest,
    IndexRequest,
    PageResponse,
    ParseRequest,
    ParseResponse,
    RetrieveRequest,
)

# ── Schema defaults ──────────────────────────────────────────────────────────


class TestSchemas:
    def test_parse_request_defaults(self):
        r = ParseRequest()
        assert r.backend == "pymupdf"
        assert r.chunk is True
        assert r.chunk_size == 512
        assert r.chunk_overlap == 64

    def test_page_response_fields(self):
        p = PageResponse(page_num=1, markdown="# Title", backend="pymupdf")
        assert p.has_bbox is False

    def test_parse_response_chunk_count_default(self):
        r = ParseResponse(doc_id="abc", pdf_path="x.pdf", pages=[], total_time_s=1.0)
        assert r.chunk_count == 0

    def test_index_request_defaults(self):
        r = IndexRequest(doc_id="abc")
        assert r.dense_model == "intfloat/multilingual-e5-large"
        assert r.sparse_type == "korean_bm25"

    def test_retrieve_request_defaults(self):
        r = RetrieveRequest(query="test")
        assert r.k == 5
        assert r.doc_id is None

    def test_ask_request_defaults(self):
        r = AskRequest(query="what?")
        assert r.k == 5


# ── Parse endpoint error handling ────────────────────────────────────────────


class TestParseEndpoint:
    @pytest.fixture()
    def client(self):
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_invalid_backend_returns_400(self, client):
        resp = client.post(
            "/api/parse",
            data={"backend": "nonexistent_backend"},
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "nonexistent_backend" in resp.json()["detail"]
        assert "Available" in resp.json()["detail"]
