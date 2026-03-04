"""
RAG 벤치마크 표시명 매핑 상수 및 유틸리티.

보고서, orchestrator, 콘솔 요약 등에서 재사용.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Dense 임베딩 모델 표시명
# ---------------------------------------------------------------------------

DENSE_DISPLAY: dict[str, dict[str, str]] = {
    "bge-m3": {
        "short": "BGE-M3",
        "params": "570M",
        "note": "100+ 언어, MIRACL 한국어 SOTA",
        "type": "local",
    },
    "e5": {
        "short": "E5-multilingual",
        "params": "560M",
        "note": "다국어 E5, 명령어 prefix 방식",
        "type": "local",
    },
    "kosimcse": {
        "short": "KoSimCSE",
        "params": "110M",
        "note": "한국어 SimCSE 대조 학습",
        "type": "local",
    },
    "snowflake": {
        "short": "Snowflake-KO",
        "params": "600M",
        "note": "한국어 실무 문서 SOTA",
        "type": "local",
    },
    "text-embedding-3-large": {
        "short": "OpenAI (API)",
        "params": "—",
        "note": "text-embedding-3-large, 3072차원",
        "type": "api",
    },
    "embedding-query": {
        "short": "Upstage Solar (API)",
        "params": "—",
        "note": "Upstage Solar Embedding, 4096차원",
        "type": "api",
    },
}

# ---------------------------------------------------------------------------
# Sparse / Reranker 표시명
# ---------------------------------------------------------------------------

SPARSE_DISPLAY: dict[str, str] = {
    "korean_bm25": "BM25",
    "splade": "SPLADE",
}

RERANKER_DISPLAY: dict[str, str] = {
    "colbert": "ColBERT",
    "flashrank": "FlashRank",
}

# ---------------------------------------------------------------------------
# 데이터셋 정보
# ---------------------------------------------------------------------------

DATASET_INFO: dict[str, dict[str, str]] = {
    "general": {
        "source": "MIRACL(ko) + Ko-StrategyQA + Belebele + MrTiDy",
        "note": "위키피디아 기반 범용 질의응답",
    },
    "legal": {"source": "법률 QA 데이터셋", "note": "법률 문서 질의응답"},
    "business": {"source": "비즈니스 QA 데이터셋", "note": "비즈니스 문서 질의응답"},
    "medical": {"source": "의료 QA 데이터셋", "note": "의료 문서 질의응답"},
    "technical": {"source": "기술 QA 데이터셋", "note": "기술 문서 질의응답"},
}


# ---------------------------------------------------------------------------
# 표시명 변환 유틸리티
# ---------------------------------------------------------------------------


def short_name(strategy: str) -> str:
    """기술적 strategy 문자열을 사람 친화 표시명으로 변환.

    예: "e5_korean_bm25_colbert" → "E5-multilingual + BM25 + ColBERT"
    """
    dense = None
    for key, meta in DENSE_DISPLAY.items():
        if key.lower() in strategy.lower():
            dense = meta["short"]
            break

    sparse = None
    for key, label in SPARSE_DISPLAY.items():
        if key in strategy.lower():
            sparse = label
            break

    reranker = None
    strategy_lower = strategy.lower()
    for key, label in RERANKER_DISPLAY.items():
        if key in strategy_lower:
            reranker = label
            break

    parts = []
    if dense:
        parts.append(dense)
    if sparse:
        parts.append(sparse)
    if reranker:
        parts.append(reranker)

    if parts:
        return " + ".join(parts)
    return strategy
