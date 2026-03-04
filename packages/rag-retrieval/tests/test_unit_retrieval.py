"""Unit tests for rag-retrieval: ComboSpec, presets, validation."""

from __future__ import annotations

import pytest

from autorag_retrieval.combo.spec import (
    PRESETS,
    ComboSpec,
    generate_valid_combinations,
)
from autorag_retrieval.strategies.dense_sparse import DENSE_MODELS, SPARSE_TYPES


# ── ComboSpec validation ─────────────────────────────────────────────────────


class TestComboSpec:
    def test_valid_creation(self):
        c = ComboSpec(
            dense="bge-m3",
            sparse="korean_bm25",
            reranker="colbert",
            llm_support="contextual",
        )
        assert c.dense == "bge-m3"

    def test_missing_dense_raises(self):
        with pytest.raises(ValueError, match="dense는 필수"):
            ComboSpec(
                dense="",
                sparse="korean_bm25",
                reranker="colbert",
                llm_support="contextual",
            )

    def test_missing_sparse_raises(self):
        with pytest.raises(ValueError, match="sparse는 필수"):
            ComboSpec(
                dense="bge-m3", sparse="", reranker="colbert", llm_support="contextual"
            )

    def test_missing_reranker_raises(self):
        with pytest.raises(ValueError, match="reranker는 필수"):
            ComboSpec(
                dense="bge-m3",
                sparse="korean_bm25",
                reranker=None,
                llm_support="contextual",
            )

    def test_missing_llm_support_raises(self):
        with pytest.raises(ValueError, match="llm_support는 필수"):
            ComboSpec(
                dense="bge-m3",
                sparse="korean_bm25",
                reranker="colbert",
                llm_support=None,
            )


# ── ComboSpec properties ─────────────────────────────────────────────────────


class TestComboSpecProperties:
    def test_label(self):
        c = ComboSpec(
            dense="e5", sparse="splade", reranker="flashrank", llm_support="contextual"
        )
        assert c.label == "e5+splade+flashrank+contextual"

    def test_index_key(self):
        c = ComboSpec(
            dense="e5", sparse="splade", reranker="colbert", llm_support="contextual"
        )
        assert c.index_key == "e5:splade"

    def test_retrieval_mode(self):
        c = ComboSpec(
            dense="e5", sparse="splade", reranker="colbert", llm_support="contextual"
        )
        assert "hybrid" in c.retrieval_mode
        assert "colbert_rerank" in c.retrieval_mode
        assert "llm_support" in c.retrieval_mode


# ── Presets ───────────────────────────────────────────────────────────────────


class TestPresets:
    def test_quick_preset_exists(self):
        assert "quick" in PRESETS

    def test_all_presets_have_required_keys(self):
        required = {"dense_models", "sparse_models", "rerankers", "llm_support"}
        for name, config in PRESETS.items():
            assert required.issubset(config.keys()), f"Preset '{name}' missing keys"

    def test_generate_quick(self):
        combos = generate_valid_combinations(PRESETS["quick"])
        assert len(combos) == 1

    def test_generate_service(self):
        combos = generate_valid_combinations(PRESETS["service"])
        assert len(combos) == 6  # 3 dense × 2 sparse


# ── generate_valid_combinations validation ───────────────────────────────────


class TestGenerateValidCombinations:
    def test_invalid_dense_raises(self):
        config = {
            "dense_models": ["nonexistent_model"],
            "sparse_models": ["korean_bm25"],
            "rerankers": ["colbert"],
            "llm_support": ["contextual"],
        }
        with pytest.raises(ValueError, match="유효하지 않은 dense_models"):
            generate_valid_combinations(config)

    def test_invalid_sparse_raises(self):
        config = {
            "dense_models": ["bge-m3"],
            "sparse_models": ["invalid_sparse"],
            "rerankers": ["colbert"],
            "llm_support": ["contextual"],
        }
        with pytest.raises(ValueError, match="유효하지 않은 sparse_models"):
            generate_valid_combinations(config)


# ── Model registries ─────────────────────────────────────────────────────────


class TestModelRegistries:
    def test_dense_models_has_bge_m3(self):
        assert "bge-m3" in DENSE_MODELS

    def test_sparse_types(self):
        assert "korean_bm25" in SPARSE_TYPES
        assert "splade" in SPARSE_TYPES
