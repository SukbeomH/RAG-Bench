"""
ComboSpec + PRESETS + generate_valid_combinations.

4-Layer 조합 명세 및 프리셋 정의.

  Layer 1: Dense Model   — kosimcse, e5, bge-m3, snowflake-ko (로컬) / openai-large, upstage (API)
  Layer 2: Sparse Model  — korean_bm25, splade
  Layer 3: Reranker      — none, colbert, flashrank
  Layer 4: Contextual    — none, contextual (인덱싱 시 LLM 문맥 부착)

총 유효 조합 (full): 5 × 2 × 3 × 2 = 60개
service 프리셋 (서비스 모델 선정용): 4 × 2 × 1 × 1 = 8개
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from rag_bench.strategies.dense_sparse import DENSE_MODELS, SPARSE_TYPES

# HuggingFace 로컬 모델 목록 (snowflake-ko 추가: 한국어 Retrieval SOTA).
_HF_DENSE_MODELS = ["kosimcse", "e5", "bge-m3", "snowflake-ko"]

# 유료 API 모델 포함 전체 Dense 모델 목록.
_ALL_DENSE_MODELS = _HF_DENSE_MODELS + ["openai-large", "upstage"]


@dataclass
class ComboSpec:
    """4-Layer 조합 명세."""

    dense: str = ""                   # Layer 1: DENSE_MODELS 키 (예: "kosimcse")
    sparse: str = ""                  # Layer 2: SPARSE_TYPES 값 (예: "splade")
    reranker: Optional[str] = None    # Layer 3: None | "colbert" | "flashrank"
    llm_support: Optional[str] = None # Layer 4: None | "contextual"

    @property
    def label(self) -> str:
        parts = [self.dense, self.sparse]
        if self.reranker:
            parts.append(self.reranker)
        if self.llm_support:
            parts.append(self.llm_support)
        return "+".join(parts)

    @property
    def retrieval_mode(self) -> str:
        mode = "hybrid"
        suffixes = []
        if self.reranker:
            suffixes.append(self.reranker + "_rerank")
        if self.llm_support:
            suffixes.append("llm_support")
        if suffixes:
            mode += "_with_" + "_and_".join(suffixes)
        return mode

    @property
    def index_key(self) -> str:
        """인덱스 캐싱 키. (dense, sparse) 쌍으로 결정."""
        return f"{self.dense}:{self.sparse}"


# ---------------------------------------------------------------------------
# 프리셋
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, list]] = {
    "quick": {
        "dense_models": ["bge-m3"],
        "sparse_models": ["korean_bm25"],
        "rerankers": [None, "flashrank"],
        "llm_support": [None],
    },
    "standard": {
        "dense_models": _ALL_DENSE_MODELS,
        "sparse_models": list(SPARSE_TYPES),
        "rerankers": [None, "flashrank"],
        "llm_support": [None],
    },
    "full": {
        "dense_models": _ALL_DENSE_MODELS,
        "sparse_models": list(SPARSE_TYPES),
        "rerankers": [None, "colbert", "flashrank"],
        "llm_support": [None, "contextual"],
    },
    # 서비스 모델 선정 전용 프리셋:
    # ColBERT 리랭킹 + Contextual Retrieval 고정, HF 4종 × Sparse 2종 = 8개 조합
    "service": {
        "dense_models": _HF_DENSE_MODELS,        # kosimcse, e5, bge-m3, snowflake-ko
        "sparse_models": list(SPARSE_TYPES),      # korean_bm25, splade
        "rerankers": ["colbert"],                 # ColBERT 고정
        "llm_support": ["contextual"],            # Contextual 고정
    },
}


def generate_valid_combinations(config: Dict[str, list]) -> List[ComboSpec]:
    """4-Layer 카테시안 곱으로 유효 조합 생성.

    Args:
        config: PRESETS 딕셔너리 항목.

    Raises:
        ValueError: config의 dense_models 또는 sparse_models에 유효하지 않은
            키가 포함된 경우.
    """
    # 유효성 검증
    valid_dense = set(DENSE_MODELS.keys())
    invalid_dense = [d for d in config.get("dense_models", []) if d not in valid_dense]
    if invalid_dense:
        raise ValueError(
            f"유효하지 않은 dense_models 키: {invalid_dense}. "
            f"허용 값: {sorted(valid_dense)}"
        )

    valid_sparse = set(SPARSE_TYPES)
    invalid_sparse = [s for s in config.get("sparse_models", []) if s not in valid_sparse]
    if invalid_sparse:
        raise ValueError(
            f"유효하지 않은 sparse_models 값: {invalid_sparse}. "
            f"허용 값: {sorted(valid_sparse)}"
        )

    combos = []
    for d in config["dense_models"]:
        for s in config["sparse_models"]:
            for r in config["rerankers"]:
                for llm_sup in config["llm_support"]:
                    combos.append(ComboSpec(dense=d, sparse=s, reranker=r, llm_support=llm_sup))
    return combos
