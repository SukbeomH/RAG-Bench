"""Tests for OmniDocBench-compatible metrics + evaluator integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from autorag_pdf_eval.omnidoc_metrics import (
    OmniDocScore,
    _levenshtein,
    _md_table_to_html,
    compute_bleu,
    compute_meteor,
    compute_omnidoc_scores,
    compute_teds_html,
    compute_text_ned,
)
from autorag_pdf_eval.evaluator import (
    BenchResult,
    PageScore,
    compute_structure,
    evaluate_document,
    evaluate_page,
)
from autorag_pdf_eval.spec import GT_MAP, BenchSpec, get_preset


# ── Levenshtein ──────────────────────────────────────────────────────────────


class TestLevenshtein:
    def test_identical_strings(self):
        assert _levenshtein("abc", "abc") == 0

    def test_empty_strings(self):
        assert _levenshtein("", "") == 0

    def test_one_empty(self):
        assert _levenshtein("abc", "") == 3
        assert _levenshtein("", "xy") == 2

    def test_single_edit(self):
        assert _levenshtein("cat", "hat") == 1

    def test_swap_order_same_result(self):
        assert _levenshtein("short", "longer") == _levenshtein("longer", "short")


# ── compute_text_ned ─────────────────────────────────────────────────────────


class TestTextNED:
    def test_identical(self):
        assert compute_text_ned("hello world", "hello world") == 1.0

    def test_both_empty(self):
        assert compute_text_ned("", "") == 1.0

    def test_completely_different(self):
        ned = compute_text_ned("aaa", "bbb")
        assert 0.0 <= ned <= 1.0

    def test_whitespace_normalized(self):
        ned = compute_text_ned("hello   world", "hello world")
        assert ned == 1.0

    def test_partial_match(self):
        ned = compute_text_ned("hello world", "hello earth")
        assert 0.0 < ned < 1.0


# ── BLEU ─────────────────────────────────────────────────────────────────────


class TestBLEU:
    def test_perfect_match(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        score = compute_bleu(text, text)
        assert score is not None
        assert score > 90

    def test_partial_match(self) -> None:
        pred = "The quick brown fox"
        gt = "The quick brown fox jumps over the lazy dog"
        score = compute_bleu(pred, gt)
        assert score is not None
        assert 0 < score < 100

    def test_empty_pred(self) -> None:
        score = compute_bleu("", "hello world test sentence")
        assert score == 0.0

    def test_empty_gt(self) -> None:
        score = compute_bleu("hello world", "")
        assert score == 0.0

    def test_korean_text(self) -> None:
        text = "안녕하세요 세계 이것은 한국어 테스트 문장입니다"
        score = compute_bleu(text, text)
        assert score is not None
        assert score > 90


# ── METEOR ───────────────────────────────────────────────────────────────────


class TestMETEOR:
    def test_perfect_match(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        score = compute_meteor(text, text)
        assert score is not None
        assert score > 95

    def test_partial_match(self) -> None:
        pred = "The quick brown fox"
        gt = "The quick brown fox jumps over the lazy dog"
        score = compute_meteor(pred, gt)
        assert score is not None
        assert 0 < score < 100

    def test_empty_pred(self) -> None:
        score = compute_meteor("", "hello world")
        assert score == 0.0

    def test_empty_gt(self) -> None:
        score = compute_meteor("hello world", "")
        assert score == 0.0


# ── _md_table_to_html ────────────────────────────────────────────────────────


class TestMdTableToHtml:
    def test_simple_table(self) -> None:
        rows = [["A", "B"], ["1", "2"]]
        html = _md_table_to_html(rows)
        assert "<table>" in html
        assert "</table>" in html
        assert "<th>A</th>" in html
        assert "<td>1</td>" in html

    def test_empty_rows(self) -> None:
        assert _md_table_to_html([]) == "<table></table>"

    def test_special_chars_escaped(self) -> None:
        rows = [["<script>", "a&b"]]
        html = _md_table_to_html(rows)
        assert "&lt;script&gt;" in html
        assert "a&amp;b" in html


# ── TEDS-HTML ────────────────────────────────────────────────────────────────


class TestTEDSHTML:
    def test_identical_tables(self) -> None:
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        score = compute_teds_html(md, md)
        assert score is not None
        assert score == pytest.approx(1.0)

    def test_no_gt_table(self) -> None:
        score = compute_teds_html("| A | B |\n|---|---|\n| 1 | 2 |", "no table here")
        assert score == -1.0

    def test_no_pred_table(self) -> None:
        score = compute_teds_html("no table here", "| A | B |\n|---|---|\n| 1 | 2 |")
        assert score is not None
        assert score == 0.0

    def test_different_tables(self) -> None:
        pred = "| A | B |\n|---|---|\n| 1 | 2 |"
        gt = "| X | Y | Z |\n|---|---|---|\n| 3 | 4 | 5 |"
        score = compute_teds_html(pred, gt)
        assert score is not None
        assert 0 < score < 1


# ── OmniDocScore 통합 ────────────────────────────────────────────────────────


class TestOmniDocScores:
    def test_combined_metrics(self) -> None:
        pred = "The quick brown fox jumps over the lazy dog"
        gt = "The quick brown fox jumps over the lazy dog"
        scores = compute_omnidoc_scores(pred, gt)
        assert isinstance(scores, OmniDocScore)
        assert scores.edit_dist is not None and scores.edit_dist > 0.9
        assert scores.bleu is not None and scores.bleu > 90
        assert scores.meteor is not None and scores.meteor > 95
        assert scores.teds_html == -1.0

    def test_with_tables(self) -> None:
        md = "Some text.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nMore text."
        scores = compute_omnidoc_scores(md, md)
        assert scores.teds_html is not None and scores.teds_html > 0.9

    def test_empty_input(self) -> None:
        scores = compute_omnidoc_scores("", "")
        assert scores.edit_dist is not None
        assert scores.bleu == 0.0
        assert scores.meteor == 0.0


# ── compute_structure ────────────────────────────────────────────────────────


class TestComputeStructure:
    def test_headers_detected(self):
        s = compute_structure("# Title\nsome text")
        assert s["has_headers"] is True

    def test_no_headers(self):
        s = compute_structure("plain text only")
        assert s["has_headers"] is False

    def test_word_count(self):
        s = compute_structure("one two three")
        assert s["word_count"] == 3


# ── evaluate_page / evaluate_document ────────────────────────────────────────


class TestEvaluatePage:
    def test_with_gt(self):
        score = evaluate_page("hello world", "hello world", page_num=1, speed_s=0.5)
        assert score.omnidoc is not None
        assert score.omnidoc.edit_dist == pytest.approx(1.0)
        assert score.word_count == 2

    def test_without_gt(self):
        score = evaluate_page("hello world", None, page_num=1, speed_s=0.5)
        assert score.omnidoc is None
        assert score.word_count == 2


class TestEvaluateDocument:
    def test_basic(self):
        bench = evaluate_document(
            pred_text="hello world",
            gt_text="hello world",
            speed_s=1.0,
            backend="test",
            pdf_name="test.pdf",
            mode="direct",
        )
        assert len(bench.pages) == 1
        assert bench.avg_edit_dist is not None
        assert bench.avg_edit_dist > 0.9


# ── BenchResult properties ───────────────────────────────────────────────────


class TestBenchResult:
    def test_avg_edit_dist_no_pages(self):
        r = BenchResult(backend="x", pdf_name="y", mode="direct")
        assert r.avg_edit_dist is None

    def test_avg_edit_dist_with_omnidoc(self):
        r = BenchResult(
            backend="x",
            pdf_name="y",
            mode="direct",
            pages=[
                PageScore(
                    page=1,
                    speed_s=1.0,
                    word_count=10,
                    omnidoc=OmniDocScore(edit_dist=0.9, bleu=80.0, meteor=85.0),
                )
            ],
        )
        assert r.avg_edit_dist == pytest.approx(0.9)
        assert r.avg_bleu == pytest.approx(80.0)
        assert r.avg_meteor == pytest.approx(85.0)
        assert r.avg_teds_html is None

    def test_total_words(self):
        r = BenchResult(
            backend="x",
            pdf_name="y",
            mode="direct",
            pages=[
                PageScore(page=1, speed_s=1.0, word_count=100),
                PageScore(page=2, speed_s=2.0, word_count=200),
            ],
        )
        assert r.total_words == 300
        assert r.total_time_s == 3.0


# ── BenchSpec ────────────────────────────────────────────────────────────────


class TestBenchSpec:
    def test_label_format(self):
        spec = BenchSpec(backend="pymupdf", pdf_name="text_only.pdf")
        assert spec.label == "pymupdf-text-only-direct"

    def test_label_truncated_to_63(self):
        spec = BenchSpec(backend="pymupdf", pdf_name="a" * 80 + ".pdf")
        assert len(spec.label) <= 63

    def test_default_mode(self):
        spec = BenchSpec(backend="pymupdf", pdf_name="x.pdf")
        assert spec.mode == "direct"


# ── Presets / GT_MAP ─────────────────────────────────────────────────────────


class TestPresetsAndGTMap:
    def test_get_preset_quick(self):
        specs = get_preset("quick")
        assert len(specs) >= 1
        assert all(isinstance(s, BenchSpec) for s in specs)

    def test_get_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="알 수 없는 프리셋"):
            get_preset("nonexistent")

    def test_gt_map_all_values_are_md(self):
        for pdf, gt in GT_MAP.items():
            assert pdf.endswith(".pdf")
            assert gt.endswith(".md")

    def test_gt_map_has_text_only(self):
        assert "text_only.pdf" in GT_MAP


# ── E2E: evaluate_document with real parser ──────────────────────────────────


class TestE2EEvaluateDocument:
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
        assert bench.avg_edit_dist is not None
        assert bench.avg_edit_dist > 0.5, f"NED too low: {bench.avg_edit_dist}"


class TestGTMap:
    def test_gt_map_files_exist(self, benchmark_pdf_dir: Path, gt_dir: Path) -> None:
        for pdf_name, gt_name in GT_MAP.items():
            pdf_path = benchmark_pdf_dir / pdf_name
            gt_path = gt_dir / gt_name
            assert pdf_path.exists(), f"PDF not found: {pdf_path}"
            assert gt_path.exists(), f"GT not found: {gt_path}"
