"""
build_strategy_from_spec — ComboSpec → 전략 인스턴스 빌드.
"""

from pathlib import Path
from typing import Optional

from rag_bench.combo.spec import ComboSpec
from rag_bench.combo.cache import IndexCacheManager


def build_strategy_from_spec(
    spec: ComboSpec,
    index_cache: IndexCacheManager,
    child_chunks,
    parent_pairs,
    reindex: bool = False,
    pre_enriched=None,
    qdrant_base_dir: Optional[Path] = None,
):
    """ComboSpec에서 전략 인스턴스 생성.

    Args:
        spec: 인덱스 스펙 (dense, sparse 모델 정보).
        index_cache: 인덱스 캐시 매니저.
        child_chunks: 인덱싱에 사용할 child 청크 목록.
        parent_pairs: parent 문서 쌍 목록.
        reindex: True이면 기존 인덱스를 무시하고 재인덱싱한다.
        pre_enriched: 사전 생성된 enriched 청크 목록.
        qdrant_base_dir: Qdrant DB 경로의 기준 디렉토리.
                         None이면 BENCH_DATA_DIR을 사용한다.
    """
    # 1. Base: DenseSparse (인덱스 캐시 활용)
    base = index_cache.get_or_build(
        spec, child_chunks, reindex=reindex, qdrant_base_dir=qdrant_base_dir
    )

    # 2. LLM Support 적용 (contextual 캐시 활용)
    if spec.llm_support == "contextual":
        base = index_cache.get_or_build_contextual(
            spec, child_chunks, parent_pairs, reindex,
            pre_enriched=pre_enriched, qdrant_base_dir=qdrant_base_dir,
        )

    # 3. Reranker 적용 (Decorator)
    cfg = index_cache.config
    if spec.reranker == "colbert":
        from rag_bench.strategies.colbert_rerank import ColBERTRerankStrategy

        shared = index_cache.get_colbert_model()
        shared_lock = index_cache.get_colbert_lock()
        return ColBERTRerankStrategy(
            base_strategy=base,
            model_name=cfg.colbert_model,
            rerank_n=cfg.rerank_n,
            device=cfg.colbert_device,
            shared_model=shared,
            shared_lock=shared_lock,
        )
    elif spec.reranker == "flashrank":
        from rag_bench.strategies.flashrank_rerank import FlashRankRerankStrategy

        shared = index_cache.get_flashrank_ranker()
        return FlashRankRerankStrategy(
            base_strategy=base,
            model_name=cfg.flashrank_model,
            rerank_n=cfg.rerank_n,
            max_length=cfg.flashrank_max_length,
            shared_ranker=shared,
        )

    return base
