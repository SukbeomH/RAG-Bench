"""E2E tests for autorag_rag_eval — RAGAS constants + metric presets."""

from __future__ import annotations

import pytest

from autorag_rag_eval.constants import RAGAS_COLS, RAGAS_WEIGHTS
from autorag_rag_eval.metrics import METRIC_REGISTRY, MetricPreset, MetricTier


class TestRagasWeights:
    def test_weights_sum_to_one(self) -> None:
        total = sum(RAGAS_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_cols_match_weights(self) -> None:
        assert set(RAGAS_COLS) == set(RAGAS_WEIGHTS.keys())

    def test_all_weights_positive(self) -> None:
        for key, val in RAGAS_WEIGHTS.items():
            assert val > 0, f"{key} weight should be positive"


class TestMetricPreset:
    def test_preset_values_exist(self) -> None:
        expected = {"core_only", "full", "reference_free", "comprehensive"}
        actual = {p.value for p in MetricPreset}
        assert expected == actual


class TestMetricRegistry:
    def test_registry_not_empty(self) -> None:
        assert len(METRIC_REGISTRY) > 0

    def test_required_fields(self) -> None:
        required = {"tier", "requires_llm", "requires_reference"}
        for name, info in METRIC_REGISTRY.items():
            for field in required:
                assert field in info, f"metric '{name}' missing field '{field}'"

    def test_tier_types(self) -> None:
        for name, info in METRIC_REGISTRY.items():
            assert isinstance(info["tier"], MetricTier), (
                f"metric '{name}' tier should be MetricTier"
            )

    def test_core_metrics_count(self) -> None:
        core = [k for k, v in METRIC_REGISTRY.items() if v["tier"] == MetricTier.CORE]
        assert len(core) == 4, "should have exactly 4 core metrics"

    def test_lightweight_no_llm(self) -> None:
        for name, info in METRIC_REGISTRY.items():
            if info["tier"] == MetricTier.LIGHTWEIGHT:
                assert not info["requires_llm"], (
                    f"lightweight metric '{name}' should not require LLM"
                )
