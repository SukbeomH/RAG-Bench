"""E2E tests for autorag_pdf_eval — NED/TEDS evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from autorag_pdf_eval import (
    GT_MAP,
    PRESETS,
    BenchSpec,
    compute_table_teds,
    compute_text_ned,
    evaluate_document,
    get_preset,
)


class TestNED:
    def test_perfect_match(self) -> None:
        assert compute_text_ned("hello world", "hello world") == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        score = compute_text_ned("hello world", "hello worl")
        assert 0 < score < 1

    def test_empty_strings(self) -> None:
        score = compute_text_ned("", "")
        assert isinstance(score, float)

    def test_completely_different(self) -> None:
        score = compute_text_ned("abc", "xyz")
        assert score < 1.0


class TestTEDS:
    def test_identical_tables(self) -> None:
        table_md = "| A | B |\n|---|---|\n| 1 | 2 |"
        score = compute_table_teds(table_md, table_md)
        assert score == pytest.approx(1.0)

    def test_no_table_in_gt(self) -> None:
        score = compute_table_teds("| A | B |\n|---|---|\n| 1 | 2 |", "no table here")
        assert score == pytest.approx(-1.0)


class TestEvaluateDocument:
    def test_evaluate_document_pymupdf(self, sample_pdf: Path, gt_dir: Path) -> None:
        from autorag_parsers import get_parser

        parser = get_parser("pymupdf")
        result = parser.convert(str(sample_pdf))
        pred = result.full_markdown

        gt_path = gt_dir / "text_only.md"
        gt_text = gt_path.read_text(encoding="utf-8")

        bench = evaluate_document(
            pred_text=pred,
            gt_text=gt_text,
            speed_s=result.total_time_s,
            backend="pymupdf",
            pdf_name="text_only.pdf",
            mode="direct",
        )
        assert bench.avg_text_ned > 0.5, f"NED too low: {bench.avg_text_ned}"


class TestPresets:
    def test_preset_quick(self) -> None:
        specs = get_preset("quick")
        assert isinstance(specs, list)
        assert len(specs) > 0
        assert all(isinstance(s, BenchSpec) for s in specs)

    def test_all_presets_valid(self) -> None:
        for name in PRESETS:
            specs = get_preset(name)
            assert len(specs) > 0, f"preset '{name}' is empty"

    def test_invalid_preset_raises(self) -> None:
        with pytest.raises(ValueError):
            get_preset("nonexistent_preset")


class TestGTMap:
    def test_gt_map_files_exist(self, benchmark_pdf_dir: Path, gt_dir: Path) -> None:
        for pdf_name, gt_name in GT_MAP.items():
            pdf_path = benchmark_pdf_dir / pdf_name
            gt_path = gt_dir / gt_name
            assert pdf_path.exists(), f"PDF not found: {pdf_path}"
            assert gt_path.exists(), f"GT not found: {gt_path}"
