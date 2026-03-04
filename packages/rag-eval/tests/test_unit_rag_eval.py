"""Unit tests for rag-eval: RAGAS weights, MetricRegistry, presets."""

from __future__ import annotations


from autorag_rag_eval.constants import RAGAS_COLS, RAGAS_WEIGHTS
from autorag_rag_eval.metrics import (
    METRIC_REGISTRY,
    MetricPreset,
    MetricTier,
    _get_metrics_for_preset,
)


# ── RAGAS weights ────────────────────────────────────────────────────────────


class TestRagasWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(RAGAS_WEIGHTS.values()) - 1.0) < 1e-9

    def test_cols_match_weight_keys(self):
        assert set(RAGAS_COLS) == set(RAGAS_WEIGHTS.keys())

    def test_all_weights_positive(self):
        for k, v in RAGAS_WEIGHTS.items():
            assert v > 0, f"Weight for {k} should be positive"


# ── MetricRegistry ───────────────────────────────────────────────────────────


class TestMetricRegistry:
    def test_registry_not_empty(self):
        assert len(METRIC_REGISTRY) > 0

    def test_all_entries_have_required_keys(self):
        required = {"cls", "tier", "requires_reference", "requires_llm"}
        for name, info in METRIC_REGISTRY.items():
            assert required.issubset(info.keys()), f"Metric '{name}' missing keys"

    def test_core_metrics_count(self):
        core = [k for k, v in METRIC_REGISTRY.items() if v["tier"] == MetricTier.CORE]
        assert len(core) == 4

    def test_tiers_are_valid(self):
        for name, info in METRIC_REGISTRY.items():
            assert isinstance(info["tier"], MetricTier), f"'{name}' tier invalid"


# ── Presets ───────────────────────────────────────────────────────────────────


class TestMetricPresets:
    def test_core_only_returns_core(self):
        keys = _get_metrics_for_preset(MetricPreset.CORE_ONLY)
        assert len(keys) == 4
        for k in keys:
            assert METRIC_REGISTRY[k]["tier"] == MetricTier.CORE

    def test_full_returns_all(self):
        keys = _get_metrics_for_preset(MetricPreset.FULL)
        assert len(keys) == len(METRIC_REGISTRY)

    def test_reference_free_excludes_reference(self):
        keys = _get_metrics_for_preset(MetricPreset.REFERENCE_FREE)
        for k in keys:
            assert METRIC_REGISTRY[k]["requires_reference"] is False

    def test_comprehensive_includes_core(self):
        keys = _get_metrics_for_preset(MetricPreset.COMPREHENSIVE)
        core = {k for k, v in METRIC_REGISTRY.items() if v["tier"] == MetricTier.CORE}
        assert core.issubset(set(keys))

    def test_enum_values(self):
        assert MetricPreset.CORE_ONLY.value == "core_only"
        assert MetricPreset.FULL.value == "full"
