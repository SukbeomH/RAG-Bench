"""RAG 벤치마크 보고서 모듈 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autorag_rag_eval.constants import RAGAS_WEIGHTS
from autorag_rag_eval.display import (
    DENSE_DISPLAY,
    RERANKER_DISPLAY,
    SPARSE_DISPLAY,
    short_name,
)
from autorag_rag_eval.report import (
    generate_report,
    load_latency,
    load_results,
    rank_combos,
    render_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_run_dir(tmp_path: Path) -> Path:
    """최소한의 K8s 결과 디렉토리 fixture 생성."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    strategies = [
        "e5+korean_bm25+colbert+contextual",
        "bge-m3+splade+colbert+contextual",
        "snowflake+korean_bm25+colbert+contextual",
    ]

    for cat in ["general", "legal"]:
        for i, strategy in enumerate(strategies):
            combo_dir = run_dir / cat / f"combo-{i}"
            combo_dir.mkdir(parents=True)

            result = {
                "category": cat,
                "combo": strategy,
                "n_qa": 20,
                "timestamp": f"2026-03-01 10:0{i}:00",
                "ragas": [
                    {
                        "faithfulness": 0.5 + i * 0.1,
                        "answer_relevancy": 0.6 + i * 0.05,
                        "context_precision": 0.7 + i * 0.05,
                        "context_recall": 0.8 - i * 0.05,
                        "strategy": strategy,
                    }
                ],
            }
            (combo_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False), encoding="utf-8"
            )

    return run_dir


# ---------------------------------------------------------------------------
# TestDisplayConstants
# ---------------------------------------------------------------------------


class TestDisplayConstants:
    def test_dense_display_has_required_keys(self):
        for key, meta in DENSE_DISPLAY.items():
            assert "short" in meta, f"{key} missing 'short'"
            assert "params" in meta, f"{key} missing 'params'"
            assert "type" in meta, f"{key} missing 'type'"
            assert meta["type"] in ("local", "api"), f"{key} bad type: {meta['type']}"

    def test_sparse_display_not_empty(self):
        assert len(SPARSE_DISPLAY) >= 2

    def test_reranker_display_not_empty(self):
        assert len(RERANKER_DISPLAY) >= 2

    def test_short_name_converts_dense(self):
        result = short_name("e5_korean_bm25_colbert")
        assert "E5-multilingual" in result
        assert "BM25" in result
        assert "ColBERT" in result

    def test_short_name_unknown_passthrough(self):
        assert short_name("unknown_strategy_xyz") == "unknown_strategy_xyz"

    def test_short_name_partial_match(self):
        result = short_name("bge-m3+splade")
        assert "BGE-M3" in result
        assert "SPLADE" in result


# ---------------------------------------------------------------------------
# TestLoadResults
# ---------------------------------------------------------------------------


class TestLoadResults:
    def test_loads_categories(self, fake_run_dir: Path):
        results = load_results(fake_run_dir)
        assert "general" in results
        assert "legal" in results

    def test_ragas_data_present(self, fake_run_dir: Path):
        results = load_results(fake_run_dir)
        for cat, data in results.items():
            assert len(data["ragas"]) > 0
            assert data["n_qa"] == 20

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        results = load_results(empty)
        assert results == {}


# ---------------------------------------------------------------------------
# TestRankCombos
# ---------------------------------------------------------------------------


class TestRankCombos:
    def test_ragas_weights_sum_to_one(self):
        assert abs(sum(RAGAS_WEIGHTS.values()) - 1.0) < 1e-6

    def test_composite_score_calculated(self, fake_run_dir: Path):
        results = load_results(fake_run_dir)
        latency = load_latency(fake_run_dir)
        ranked = rank_combos(results, latency)

        for cat, df in ranked.items():
            assert "composite" in df.columns
            assert "rank" in df.columns
            assert df["rank"].iloc[0] == 1
            # composite 내림차순 정렬 확인
            composites = df["composite"].tolist()
            assert composites == sorted(composites, reverse=True)

    def test_latency_columns_always_present(self, fake_run_dir: Path):
        results = load_results(fake_run_dir)
        ranked = rank_combos(results, {})

        for cat, df in ranked.items():
            assert "avg_latency_ms" in df.columns
            assert "median_latency_ms" in df.columns


# ---------------------------------------------------------------------------
# TestRenderReport
# ---------------------------------------------------------------------------


class TestRenderReport:
    def test_required_sections_present(self, fake_run_dir: Path):
        results = load_results(fake_run_dir)
        ranked = rank_combos(results, {})
        md = render_report(fake_run_dir, results, ranked)

        assert "Executive Summary" in md
        assert "## 1." in md
        assert "## 2." in md
        assert "## 3." in md
        assert "## 4." in md
        assert "## 6." in md
        assert "## 7." in md
        assert "부록" in md

    def test_report_contains_category_labels(self, fake_run_dir: Path):
        results = load_results(fake_run_dir)
        ranked = rank_combos(results, {})
        md = render_report(fake_run_dir, results, ranked)

        assert "GENERAL" in md
        assert "LEGAL" in md

    def test_report_contains_strategy_names(self, fake_run_dir: Path):
        results = load_results(fake_run_dir)
        ranked = rank_combos(results, {})
        md = render_report(fake_run_dir, results, ranked)

        # short_name 변환 결과가 포함되어야 함
        assert "E5-multilingual" in md or "BGE-M3" in md or "Snowflake" in md


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_e2e_file_created(self, fake_run_dir: Path):
        output = fake_run_dir / "rag_benchmark_report.md"
        result_path = generate_report(fake_run_dir, output)

        assert result_path == output
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert len(content) > 100
        assert "Executive Summary" in content

    def test_default_output_path(self, fake_run_dir: Path):
        result_path = generate_report(fake_run_dir)
        assert result_path == fake_run_dir / "rag_benchmark_report.md"
        assert result_path.exists()

    def test_custom_output_path(self, fake_run_dir: Path, tmp_path: Path):
        custom = tmp_path / "reports" / "my_report.md"
        result_path = generate_report(fake_run_dir, custom)
        assert result_path == custom
        assert custom.exists()

    def test_empty_dir_raises(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="result.json"):
            generate_report(empty)
