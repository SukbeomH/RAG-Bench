"""E2E tests for autorag_api — FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from autorag_api.app import app
from autorag_api.schemas import (
    AskResponse,
    CitationItem,
    ParseResponse,
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestParseEndpoint:
    def test_parse_pymupdf(self, client: TestClient, sample_pdf: Path) -> None:
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/parse",
                files={"file": ("text_only.pdf", f, "application/pdf")},
                data={"backend": "pymupdf"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "doc_id" in data
        assert len(data["pages"]) > 0

    def test_parse_invalid_backend(self, client: TestClient, sample_pdf: Path) -> None:
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/parse",
                files={"file": ("text_only.pdf", f, "application/pdf")},
                data={"backend": "nonexistent_backend_xyz"},
            )
        assert resp.status_code == 400
        assert "nonexistent_backend_xyz" in resp.json()["detail"]


class TestSchemaValidation:
    def test_parse_response_roundtrip(self) -> None:
        data = {
            "doc_id": "abc123",
            "pdf_path": "test.pdf",
            "pages": [
                {
                    "page_num": 0,
                    "markdown": "# Hello",
                    "backend": "pymupdf",
                    "has_bbox": False,
                }
            ],
            "total_time_s": 1.23,
            "chunk_count": 5,
        }
        model = ParseResponse(**data)
        dumped = model.model_dump()
        assert dumped["doc_id"] == "abc123"
        assert len(dumped["pages"]) == 1

    def test_citation_item_optional_bbox(self) -> None:
        item = CitationItem(
            chunk_id="p0_c0",
            source_path="test.pdf",
            page_number=0,
            text_snippet="hello",
        )
        assert item.bbox is None
        assert item.relevance_score == 0.0

    def test_ask_response_empty_citations(self) -> None:
        resp = AskResponse(answer="test answer")
        assert resp.citations == []
