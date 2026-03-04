"""Tests for report generation module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autorag_pdf_eval.report import (
    WEIGHTS,
    classify_doc_type,
    classify_dpi,
    compute_backend_averages,
    compute_weighted_scores,
    generate_report,
    load_results,
    render_report,
)


# ── 분류 함수 ────────────────────────────────────────────────────────────────


class TestClassifyDocType:
    def test_text(self):
        assert classify_doc_type("text_only.pdf") == "text"

    def test_table_native(self):
        assert classify_doc_type("table_native.pdf") == "table"

    def test_table_image_dpi(self):
        assert classify_doc_type("table_image_150dpi.pdf") == "table"

    def test_graph(self):
        assert classify_doc_type("graph_rich.pdf") == "graph"

    def test_graph_image_dpi(self):
        assert classify_doc_type("graph_rich_image_72dpi.pdf") == "graph"

    def test_unknown(self):
        assert classify_doc_type("random.pdf") == "unknown"


class TestClassifyDpi:
    def test_native(self):
        assert classify_dpi("text_only.pdf") == "native"
        assert classify_dpi("table_native.pdf") == "native"

    def test_image(self):
        assert classify_dpi("table_image.pdf") == "image"

    def test_72dpi(self):
        assert classify_dpi("table_image_72dpi.pdf") == "72dpi"

    def test_150dpi(self):
        assert classify_dpi("table_image_150dpi.pdf") == "150dpi"

    def test_200dpi(self):
        assert classify_dpi("graph_rich_image_200dpi.pdf") == "200dpi"


# ── 테스트 데이터 fixtures ───────────────────────────────────────────────────


def _make_metrics(
    backend: str,
    pdf_name: str,
    ned: float,
    teds_html: float = -1.0,
    speed: float = 1.0,
    timestamp: str = "2026-03-04 12:00:00",
) -> dict:
    """테스트용 metrics dict 생성."""
    return {
        "backend": backend,
        "pdf_name": pdf_name,
        "mode": "direct",
        "error": None,
        "pages": [
            {
                "page": 1,
                "speed_s": speed,
                "word_count": 100,
                "has_headers": True,
                "has_tables": "|" in pdf_name or "table" in pdf_name,
                "has_formulas": False,
                "omnidoc": {
                    "edit_dist": ned,
                    "bleu": ned * 100,
                    "meteor": ned * 100,
                    "teds_html": teds_html,
                },
            }
        ],
        "summary": {
            "avg_edit_dist": ned,
            "avg_bleu": ned * 100,
            "avg_meteor": ned * 100,
            "avg_teds_html": teds_html if teds_html >= 0 else None,
            "avg_speed_s": speed,
            "total_time_s": speed,
            "total_words": 100,
        },
        "timestamp": timestamp,
    }


@pytest.fixture
def sample_metrics() -> list[dict]:
    """2개 백엔드 × 3 PDF 샘플 데이터."""
    return [
        _make_metrics("paddleocr-vl", "text_only.pdf", 0.82),
        _make_metrics("paddleocr-vl", "table_native.pdf", 0.77, teds_html=0.56),
        _make_metrics("paddleocr-vl", "graph_rich.pdf", 0.73),
        _make_metrics("upstage", "text_only.pdf", 0.85),
        _make_metrics("upstage", "table_native.pdf", 0.81, teds_html=0.63),
        _make_metrics("upstage", "graph_rich.pdf", 0.55),
    ]


@pytest.fixture
def metrics_dir(tmp_path: Path, sample_metrics: list[dict]) -> Path:
    """metrics.json 파일이 포함된 디렉토리 생성."""
    for m in sample_metrics:
        d = tmp_path / f"{m['backend']}-{m['pdf_name'].replace('.pdf', '')}"
        d.mkdir()
        (d / "metrics.json").write_text(
            json.dumps(m, ensure_ascii=False), encoding="utf-8"
        )
    return tmp_path


# ── load_results ─────────────────────────────────────────────────────────────


class TestLoadResults:
    def test_loads_all(self, metrics_dir: Path):
        results = load_results([metrics_dir])
        assert len(results) == 6

    def test_skips_errors(self, tmp_path: Path):
        d = tmp_path / "err-test"
        d.mkdir()
        (d / "metrics.json").write_text(
            json.dumps({"backend": "x", "pdf_name": "y", "error": "fail"}),
            encoding="utf-8",
        )
        results = load_results([tmp_path])
        assert len(results) == 0

    def test_dedup_newer_wins(self, tmp_path: Path):
        old_dir = tmp_path / "old"
        old_dir.mkdir()
        (old_dir / "metrics.json").write_text(
            json.dumps(
                _make_metrics("x", "y.pdf", 0.5, timestamp="2026-01-01 00:00:00")
            ),
            encoding="utf-8",
        )

        new_dir = tmp_path / "new"
        new_dir.mkdir()
        (new_dir / "metrics.json").write_text(
            json.dumps(
                _make_metrics("x", "y.pdf", 0.9, timestamp="2026-03-04 00:00:00")
            ),
            encoding="utf-8",
        )

        results = load_results([tmp_path])
        assert len(results) == 1
        assert results[0]["summary"]["avg_edit_dist"] == 0.9

    def test_empty_dir(self, tmp_path: Path):
        results = load_results([tmp_path])
        assert results == []


# ── compute_backend_averages ─────────────────────────────────────────────────


class TestComputeBackendAverages:
    def test_basic(self, sample_metrics):
        avgs = compute_backend_averages(sample_metrics)
        assert "paddleocr-vl" in avgs
        assert "upstage" in avgs

        paddle = avgs["paddleocr-vl"]
        assert paddle["text_ned"] == pytest.approx(0.82, abs=0.01)
        assert paddle["table_ned"] == pytest.approx(0.77, abs=0.01)
        assert paddle["graph_ned"] == pytest.approx(0.73, abs=0.01)
        assert paddle["overall_ned"] is not None
        assert paddle["pdf_count"] == 3

    def test_teds(self, sample_metrics):
        avgs = compute_backend_averages(sample_metrics)
        assert avgs["upstage"]["table_teds"] == pytest.approx(0.63, abs=0.01)
        # paddleocr-vl has teds_html only for table
        assert avgs["paddleocr-vl"]["table_teds"] == pytest.approx(0.56, abs=0.01)


# ── compute_weighted_scores ──────────────────────────────────────────────────


class TestComputeWeightedScores:
    def test_total_is_sum_of_weighted(self, sample_metrics):
        avgs = compute_backend_averages(sample_metrics)
        scores = compute_weighted_scores(avgs)

        for backend, ws in scores.items():
            expected_total = sum(ws["weighted"].values())
            assert ws["total"] == pytest.approx(expected_total, abs=0.01)

    def test_weights_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_criteria_present(self, sample_metrics):
        avgs = compute_backend_averages(sample_metrics)
        scores = compute_weighted_scores(avgs)
        for backend, ws in scores.items():
            for criterion in WEIGHTS:
                assert criterion in ws["scores"]
                assert criterion in ws["weighted"]


# ── render_report ────────────────────────────────────────────────────────────


class TestRenderReport:
    def test_contains_all_sections(self, sample_metrics):
        avgs = compute_backend_averages(sample_metrics)
        ws = compute_weighted_scores(avgs)
        md = render_report(sample_metrics, avgs, ws)

        assert "# PDF 파서 솔루션 비교·선정 보고서" in md
        assert "## 1. Executive Summary" in md
        assert "## 2. 배경 및 범위" in md
        assert "## 3. 평가 방법론" in md
        assert "## 4. 평가 기준 및 가중치" in md
        assert "## 5. 후보 솔루션 개요" in md
        assert "## 6. 평가 결과" in md
        assert "### 6-1." in md
        assert "### 6-2." in md
        assert "### 6-3." in md
        assert "## 7. 비용 분석" in md
        assert "## 8. 리스크 평가" in md
        assert "## 9. 추천" in md
        assert "## 10. 다음 단계" in md
        assert "## 부록 A." in md
        assert "## 부록 B." in md
        assert "## 부록 C." in md
        assert "## 부록 D." in md
        assert "## 부록 E." in md

    def test_all_backends_in_report(self, sample_metrics):
        avgs = compute_backend_averages(sample_metrics)
        ws = compute_weighted_scores(avgs)
        md = render_report(sample_metrics, avgs, ws)

        assert "paddleocr-vl" in md
        assert "upstage" in md

    def test_ranking_order(self, sample_metrics):
        avgs = compute_backend_averages(sample_metrics)
        ws = compute_weighted_scores(avgs)
        md = render_report(sample_metrics, avgs, ws)

        # paddleocr-vl should rank higher (local, free, good NED)
        assert "1위" in md


# ── generate_report (E2E) ────────────────────────────────────────────────────


class TestGenerateReport:
    def test_generates_file(self, metrics_dir: Path, tmp_path: Path):
        out = tmp_path / "output" / "report.md"
        result = generate_report([metrics_dir], output_path=out)
        assert result == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "# PDF 파서 솔루션 비교·선정 보고서" in content

    def test_default_output_path(self, metrics_dir: Path):
        result = generate_report([metrics_dir])
        assert result.exists()
        assert result.name == "report.md"

    def test_empty_dir_raises(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="metrics.json"):
            generate_report([empty])


# ── 구형 metrics 호환 ────────────────────────────────────────────────────────


class TestLegacyCompat:
    def test_old_format_avg_text_ned(self, tmp_path: Path):
        """구형 metrics.json (avg_text_ned) 로드 호환."""
        d = tmp_path / "old-result"
        d.mkdir()
        old_data = {
            "backend": "pymupdf",
            "pdf_name": "text_only.pdf",
            "mode": "direct",
            "error": None,
            "pages": [],
            "summary": {
                "avg_text_ned": 0.65,
                "avg_table_teds": 0.49,
                "avg_speed_s": 2.0,
                "total_time_s": 2.0,
                "total_words": 500,
            },
            "omnidoc_summary": {
                "avg_bleu": 60.0,
                "avg_meteor": 65.0,
                "avg_teds_html": 0.49,
            },
            "timestamp": "2026-03-01 10:00:00",
        }
        (d / "metrics.json").write_text(json.dumps(old_data), encoding="utf-8")
        results = load_results([tmp_path])
        assert len(results) == 1

        avgs = compute_backend_averages(results)
        assert avgs["pymupdf"]["overall_ned"] == pytest.approx(0.65)
