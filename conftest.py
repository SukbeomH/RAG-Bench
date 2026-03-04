"""Root conftest — shared fixtures for all e2e tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_PDF_DIR = _PROJECT_ROOT / "data" / "benchmark_pdfs"
GT_DIR = BENCHMARK_PDF_DIR / "gt"


@pytest.fixture()
def benchmark_pdf_dir() -> Path:
    assert BENCHMARK_PDF_DIR.exists(), f"benchmark_pdfs not found: {BENCHMARK_PDF_DIR}"
    return BENCHMARK_PDF_DIR


@pytest.fixture()
def gt_dir() -> Path:
    assert GT_DIR.exists(), f"gt dir not found: {GT_DIR}"
    return GT_DIR


@pytest.fixture()
def sample_pdf(benchmark_pdf_dir: Path) -> Path:
    pdf = benchmark_pdf_dir / "text_only.pdf"
    assert pdf.exists(), f"text_only.pdf not found: {pdf}"
    return pdf


@pytest.fixture()
def table_pdf(benchmark_pdf_dir: Path) -> Path:
    pdf = benchmark_pdf_dir / "table_native.pdf"
    assert pdf.exists(), f"table_native.pdf not found: {pdf}"
    return pdf
