"""Unit tests for pdf-eval: BenchSpec, evaluator, omnidoc_metrics, GT_MAP."""

from __future__ import annotations

import pytest

from autorag_pdf_eval.evaluator import (
    BenchResult,
    PageScore,
    compute_structure,
)
from autorag_pdf_eval.omnidoc_metrics import (
    OmniDocScore,
    _extract_md_tables,
    _md_table_to_html,
    compute_text_ned,
)
from autorag_pdf_eval.spec import GT_MAP, BenchSpec, get_preset


# ── Levenshtein ──────────────────────────────────────────────────────────────


class TestLevenshtein:
    def test_identical_strings(self):
        from rapidfuzz.distance import Levenshtein

        assert Levenshtein.distance("abc", "abc") == 0

    def test_empty_strings(self):
        from rapidfuzz.distance import Levenshtein

        assert Levenshtein.distance("", "") == 0

    def test_one_empty(self):
        from rapidfuzz.distance import Levenshtein

        assert Levenshtein.distance("abc", "") == 3
        assert Levenshtein.distance("", "xy") == 2

    def test_single_edit(self):
        from rapidfuzz.distance import Levenshtein

        assert Levenshtein.distance("cat", "hat") == 1

    def test_swap_order_same_result(self):
        from rapidfuzz.distance import Levenshtein

        assert Levenshtein.distance("short", "longer") == Levenshtein.distance(
            "longer", "short"
        )


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


# ── _extract_md_tables ───────────────────────────────────────────────────────


class TestExtractMdTables:
    def test_no_tables(self):
        assert _extract_md_tables("plain text") == []

    def test_single_table(self):
        md = "| h1 | h2 |\n|---|---|\n| a | b |"
        tables = _extract_md_tables(md)
        assert len(tables) == 1
        assert tables[0] == [["h1", "h2"], ["a", "b"]]

    def test_separator_rows_excluded(self):
        md = "| h1 |\n|---|\n| v1 |"
        tables = _extract_md_tables(md)
        assert len(tables) == 1
        assert len(tables[0]) == 2


# ── _md_table_to_html ────────────────────────────────────────────────────────


class TestMdTableToHtml:
    def test_empty_rows(self):
        assert _md_table_to_html([]) == "<html><body><table></table></body></html>"

    def test_header_uses_th(self):
        result = _md_table_to_html([["h1", "h2"], ["a", "b"]])
        assert "<th>h1</th>" in result
        assert "<td>a</td>" in result


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


# ── BenchResult properties ───────────────────────────────────────────────────


class TestBenchResult:
    def _make_page(self, omnidoc: OmniDocScore | None = None, **kwargs) -> PageScore:
        defaults = {"page": 1, "speed_s": 1.0, "word_count": 100}
        defaults.update(kwargs)
        return PageScore(**defaults, omnidoc=omnidoc)

    def test_no_pages_defaults(self):
        r = BenchResult(backend="x", pdf_name="y", mode="direct")
        assert r.avg_edit_dist is None
        assert r.avg_bleu is None
        assert r.total_words == 0

    def test_total_words(self):
        r = BenchResult(
            backend="x",
            pdf_name="y",
            mode="direct",
            pages=[
                self._make_page(word_count=100),
                self._make_page(page=2, speed_s=2.0, word_count=200),
            ],
        )
        assert r.total_words == 300
        assert r.total_time_s == 3.0

    def test_avg_edit_dist_with_omnidoc(self):
        r = BenchResult(
            backend="x",
            pdf_name="y",
            mode="direct",
            pages=[
                self._make_page(omnidoc=OmniDocScore(edit_dist=0.9)),
                self._make_page(page=2, omnidoc=OmniDocScore(edit_dist=0.7)),
            ],
        )
        assert r.avg_edit_dist == pytest.approx(0.8, abs=0.001)

    def test_avg_teds_html_excludes_negative(self):
        r = BenchResult(
            backend="x",
            pdf_name="y",
            mode="direct",
            pages=[
                self._make_page(omnidoc=OmniDocScore(teds_html=-1.0)),
            ],
        )
        assert r.avg_teds_html is None


# ── reeval_spec / reeval_dir ────────────────────────────────────────────────


class TestReeval:
    def test_reeval_spec_basic(self, tmp_path):
        """기존 output.md + metrics.json으로 재평가가 동작하는지 검증."""
        import json

        from autorag_pdf_eval.runner import reeval_spec

        result_dir = tmp_path / "pymupdf-text-only-direct"
        result_dir.mkdir()

        # 가짜 output.md
        (result_dir / "output.md").write_text("# Title\n\nSome text content.")

        # 가짜 metrics.json (기존 파싱 결과)
        metrics = {
            "backend": "pymupdf",
            "pdf_name": "text_only.pdf",
            "mode": "direct",
            "error": None,
            "summary": {"avg_speed_s": 1.0},
            "timestamp": "2026-01-01 00:00:00",
        }
        (result_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

        result = reeval_spec(result_dir, verbose=False)
        assert result is not None
        assert result.backend == "pymupdf"
        assert result.pdf_name == "text_only.pdf"

        # metrics.json이 갱신되었는지 확인
        updated = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
        assert "raw_summary" in updated
        assert "normalization" in updated

    def test_reeval_spec_missing_output_md(self, tmp_path):
        """output.md가 없으면 None을 반환."""
        import json

        from autorag_pdf_eval.runner import reeval_spec

        result_dir = tmp_path / "test-result"
        result_dir.mkdir()
        (result_dir / "metrics.json").write_text(
            json.dumps({"backend": "x", "pdf_name": "y.pdf"}), encoding="utf-8"
        )

        result = reeval_spec(result_dir, verbose=False)
        assert result is None

    def test_reeval_dir_processes_all(self, tmp_path):
        """reeval_dir이 모든 하위 결과를 처리하는지 검증."""
        import json

        from autorag_pdf_eval.runner import reeval_dir

        for label in ["pymupdf-text-only-direct", "docling-text-only-direct"]:
            d = tmp_path / label
            d.mkdir()
            (d / "output.md").write_text(f"# {label}\n\nContent here.")
            metrics = {
                "backend": label.split("-")[0],
                "pdf_name": "text_only.pdf",
                "mode": "direct",
                "error": None,
                "summary": {"avg_speed_s": 0.5},
            }
            (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

        results = reeval_dir(tmp_path, verbose=False)
        assert len(results) == 2
        backends = {r.backend for r in results}
        assert "pymupdf" in backends
        assert "docling" in backends
