"""
ComboSpec + PRESETS + generate_valid_combinations.

3-Layer 조합 명세 및 프리셋 정의.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from rag_bench.strategies.dense_sparse import DENSE_MODELS, SPARSE_TYPES


@dataclass
class ComboSpec:
    """3-Layer 조합 명세."""

    dense: str = ""                   # DENSE_MODELS 키 (예: "kosimcse")
    sparse: str = ""                  # SPARSE_TYPES 값 (예: "splade")
    reranker: Optional[str] = None    # None | "colbert" | "flashrank"
    llm_support: Optional[str] = None # None | "contextual"

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
        "dense_models": ["bge-m3", "minilm"],
        "sparse_models": ["fastembed_bm25"],
        "rerankers": [None, "flashrank"],
        "llm_support": [None],
    },
    "standard": {
        "dense_models": list(DENSE_MODELS.keys()),
        "sparse_models": list(SPARSE_TYPES),
        "rerankers": [None, "flashrank"],
        "llm_support": [None],
    },
    "full": {
        "dense_models": list(DENSE_MODELS.keys()),
        "sparse_models": list(SPARSE_TYPES),
        "rerankers": [None, "colbert", "flashrank"],
        "llm_support": [None, "contextual"],
    },
}


def generate_valid_combinations(config: Dict[str, list]) -> List[ComboSpec]:
    """3-Layer 카테시안 곱으로 유효 조합 생성.

    Args:
        config: PRESETS 딕셔너리 항목.
    """
    combos = []
    for d in config["dense_models"]:
        for s in config["sparse_models"]:
            for r in config["rerankers"]:
                for llm_sup in config["llm_support"]:
                    combos.append(ComboSpec(dense=d, sparse=s, reranker=r, llm_support=llm_sup))
    return combos
