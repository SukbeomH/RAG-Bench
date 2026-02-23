"""
build_strategy_from_spec — ComboSpec → 전략 인스턴스 빌드.
"""

from rag_bench.combo.spec import ComboSpec
from rag_bench.combo.cache import IndexCacheManager


def build_strategy_from_spec(
    spec: ComboSpec,
    index_cache: IndexCacheManager,
    child_chunks,
    parent_pairs,
    reindex: bool = False,
    pre_enriched=None,
):
    """ComboSpec에서 전략 인스턴스 생성."""
    # 1. Base: DenseSparse (인덱스 캐시 활용)
    base = index_cache.get_or_build(spec, child_chunks, reindex=reindex)

    # 2. LLM Support 적용 (contextual 캐시 활용)
    if spec.llm_support == "contextual":
        base = index_cache.get_or_build_contextual(
            spec, child_chunks, parent_pairs, reindex, pre_enriched=pre_enriched
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
