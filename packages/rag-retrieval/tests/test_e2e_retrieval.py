"""E2E tests for autorag_retrieval — ComboSpec + DocType."""

from __future__ import annotations

import pytest

from autorag_retrieval.combo.spec import (
    PRESETS,
    ComboSpec,
    generate_valid_combinations,
)
from autorag_retrieval.document_types.types import (
    DOC_TYPE_METADATA,
    DocType,
    get_sampling_ratio,
    list_doc_types,
)
from autorag_retrieval.strategies.dense_sparse import DENSE_MODELS, SPARSE_TYPES


class TestComboSpec:
    def test_label_format(self) -> None:
        spec = ComboSpec(
            dense="bge-m3",
            sparse="korean_bm25",
            reranker="colbert",
            llm_support="contextual",
        )
        label = spec.label
        assert "bge-m3" in label
        assert "korean_bm25" in label
        assert "+" in label

    def test_index_key_deterministic(self) -> None:
        a = ComboSpec(
            dense="e5", sparse="splade", reranker="colbert", llm_support="contextual"
        )
        b = ComboSpec(
            dense="e5", sparse="splade", reranker="flashrank", llm_support="contextual"
        )
        assert a.index_key == b.index_key, "same (dense, sparse) → same index_key"

    def test_different_dense_different_index_key(self) -> None:
        a = ComboSpec(
            dense="e5", sparse="splade", reranker="colbert", llm_support="contextual"
        )
        b = ComboSpec(
            dense="bge-m3",
            sparse="splade",
            reranker="colbert",
            llm_support="contextual",
        )
        assert a.index_key != b.index_key

    def test_missing_dense_raises(self) -> None:
        with pytest.raises(ValueError):
            ComboSpec(
                dense="", sparse="splade", reranker="colbert", llm_support="contextual"
            )

    def test_missing_reranker_raises(self) -> None:
        with pytest.raises(ValueError):
            ComboSpec(
                dense="e5", sparse="splade", reranker=None, llm_support="contextual"
            )


class TestGenerateCombinations:
    EXPECTED_COUNTS = {
        "quick": 1,
        "standard": 10,
        "full": 20,
        "service": 6,
    }

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_preset_count(self, preset_name: str) -> None:
        combos = generate_valid_combinations(PRESETS[preset_name])
        expected = self.EXPECTED_COUNTS[preset_name]
        assert len(combos) == expected, (
            f"{preset_name}: expected {expected}, got {len(combos)}"
        )

    def test_all_combos_are_combospec(self) -> None:
        for name, config in PRESETS.items():
            combos = generate_valid_combinations(config)
            assert all(isinstance(c, ComboSpec) for c in combos), f"preset '{name}'"


class TestDocType:
    def test_enum_values(self) -> None:
        values = list_doc_types()
        assert len(values) == 5
        for expected in ["technical", "legal", "business", "medical", "general"]:
            assert expected in values

    def test_metadata_fields(self) -> None:
        for dt in DocType:
            meta = DOC_TYPE_METADATA[dt]
            assert "sampling_ratio" in meta
            assert isinstance(meta["sampling_ratio"], float)
            assert "hf_dataset" in meta

    def test_sampling_ratio_range(self) -> None:
        for dt in DocType:
            ratio = get_sampling_ratio(dt)
            assert 0 < ratio <= 1.0, f"{dt.value}: ratio={ratio}"


class TestModelRegistry:
    def test_dense_models_keys(self) -> None:
        expected = {"kosimcse", "e5", "bge-m3", "openai-large", "upstage"}
        assert expected <= set(DENSE_MODELS.keys())

    def test_sparse_types(self) -> None:
        assert "korean_bm25" in SPARSE_TYPES
        assert "splade" in SPARSE_TYPES
        assert len(SPARSE_TYPES) == 2
