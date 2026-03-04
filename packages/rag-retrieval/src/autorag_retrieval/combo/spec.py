"""
ComboSpec + PRESETS + generate_valid_combinations.

4-Layer 조합 명세 및 프리셋 정의.

  Layer 1: Dense Model   — kosimcse, e5, bge-m3 (로컬) / openai-large, upstage (API)
  Layer 2: Sparse Model  — korean_bm25, splade
  Layer 3: Reranker      — none, colbert, flashrank
  Layer 4: Contextual    — none, contextual (인덱싱 시 LLM 문맥 부착)

총 유효 조합 (full):     5 × 2 × 2 × 1 = 20개  (HF 3 + API 2) × sparse 2 × (colbert/flashrank) × contextual
total (standard):       5 × 2 × 1 × 1 = 10개  (HF 3 + API 2) × sparse 2 × flashrank × contextual
total (quick):          1 × 1 × 1 × 1 =  1개  bge-m3 × korean_bm25 × flashrank × contextual
service 프리셋 (서비스 모델 선정용): 3 × 2 × 1 × 1 = 6개  HF 3 × sparse 2 × colbert × contextual

[공통 제약 조건 — 모든 프리셋]
  - Dense + Sparse 하이브리드 필수 (단일 모델 전용 불가)
  - Reranker 항시 적용 (None 없음)
  - Contextual Retrieval 항시 적용 (None 없음)
  → ComboSpec.__post_init__에서 위반 시 ValueError

[service 프리셋]
  - 비교 변수: Dense 3종 × Sparse 2종 = 6개 조합
  - 통제 변수: ColBERT 리랭커 고정, Contextual Retrieval 고정
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from autorag_retrieval.strategies.dense_sparse import DENSE_MODELS, SPARSE_TYPES

# HuggingFace 로컬 모델 목록.
_HF_DENSE_MODELS = ["kosimcse", "e5", "bge-m3"]

# 유료 API 모델 포함 전체 Dense 모델 목록.
_ALL_DENSE_MODELS = _HF_DENSE_MODELS + ["openai-large", "upstage"]


@dataclass
class ComboSpec:
    """4-Layer 조합 명세.

    4개 필드 모두 필수. None 또는 빈 문자열은 허용하지 않는다.
      - dense:       Dense 임베딩 모델 (예: "bge-m3")
      - sparse:      Sparse 임베딩 모델 (예: "splade")
      - reranker:    리랭커 (예: "colbert", "flashrank") — 항시 적용
      - llm_support: LLM 지원 기법 (예: "contextual") — 항시 적용
    """

    dense: str = ""                   # Layer 1: DENSE_MODELS 키
    sparse: str = ""                  # Layer 2: SPARSE_TYPES 값
    reranker: Optional[str] = None    # Layer 3: "colbert" | "flashrank"
    llm_support: Optional[str] = None # Layer 4: "contextual"

    def __post_init__(self) -> None:
        if not self.dense:
            raise ValueError("ComboSpec.dense는 필수입니다. 단일 Sparse 전용 조합은 허용하지 않습니다.")
        if not self.sparse:
            raise ValueError("ComboSpec.sparse는 필수입니다. 단일 Dense 전용 조합은 허용하지 않습니다.")
        if not self.reranker:
            raise ValueError("ComboSpec.reranker는 필수입니다. 리랭커 없는 조합은 허용하지 않습니다.")
        if not self.llm_support:
            raise ValueError("ComboSpec.llm_support는 필수입니다. Contextual Retrieval 없는 조합은 허용하지 않습니다.")

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
    # 빠른 동작 확인용 (bge-m3 × korean_bm25 단일 조합)
    "quick": {
        "dense_models": ["bge-m3"],
        "sparse_models": ["korean_bm25"],
        "rerankers": ["flashrank"],
        "llm_support": ["contextual"],
    },
    # 전체 모델(HF + API) × flashrank 기준 표준 비교
    "standard": {
        "dense_models": _ALL_DENSE_MODELS,
        "sparse_models": list(SPARSE_TYPES),
        "rerankers": ["flashrank"],
        "llm_support": ["contextual"],
    },
    # 전체 모델 × 리랭커 2종 × Contextual 전체 탐색
    "full": {
        "dense_models": _ALL_DENSE_MODELS,
        "sparse_models": list(SPARSE_TYPES),
        "rerankers": ["colbert", "flashrank"],
        "llm_support": ["contextual"],
    },
    # 서비스 모델 선정 전용 프리셋:
    # ColBERT 리랭킹 + Contextual Retrieval 고정, HF 3종 × Sparse 2종 = 6개 조합
    "service": {
        "dense_models": _HF_DENSE_MODELS,        # kosimcse, e5, bge-m3
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
        ValueError: 유효하지 않은 모델 키이거나 ComboSpec 제약 조건 위반 시.
            (dense/sparse 필수, reranker/llm_support 필수 — None 불가)
    """
    # 유효성 검증 — dense/sparse
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
