"""
CacheConfig + IndexCacheManager.

인덱스 캐시 관리 및 설정.
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_bench.config import (
    BENCH_DATA_DIR,
    DEFAULT_COLBERT_MODEL,
    DEFAULT_CONTEXTUAL_LLM,
    DEFAULT_FLASHRANK_MODEL,
    MODELS_DIR,
    QDRANT_DB_PREFIX,
)
from rag_bench.combo.spec import ComboSpec


@dataclass
class CacheConfig:
    """IndexCacheManager에서 사용하는 모델/리소스 설정."""

    colbert_model: str = DEFAULT_COLBERT_MODEL
    colbert_device: str = "cpu"
    dense_device: Optional[str] = None   # None이면 DenseSparseStrategy 내부에서 detect_device() 자동 사용
    flashrank_model: str = DEFAULT_FLASHRANK_MODEL
    flashrank_max_length: int = 512
    contextual_llm: str = DEFAULT_CONTEXTUAL_LLM
    rerank_n: int = 20


class IndexCacheManager:
    """동일 (dense, sparse) 쌍은 같은 Qdrant 인덱스를 재사용."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.cache: Dict[str, Tuple[Any, str]] = {}
        self.ctx_cache: Dict[str, Any] = {}
        self._colbert_model: Any = None
        self._colbert_lock: threading.Lock = threading.Lock()
        self._flashrank_ranker: Any = None

    def get_colbert_model(self):
        """ColBERT 모델을 1회만 로드하고 이후 공유."""
        if self._colbert_model is not None:
            return self._colbert_model
        from pylate import models
        print(f"[ColBERT 캐시] 모델 최초 로드 중 (device={self.config.colbert_device})...")
        self._colbert_model = models.ColBERT(
            model_name_or_path=self.config.colbert_model,
            device=self.config.colbert_device,
            trust_remote_code=True,
        )
        print("[ColBERT 캐시] 모델 로드 완료.")
        return self._colbert_model

    def get_colbert_lock(self) -> threading.Lock:
        """공유 ColBERT 모델에 대한 전용 Lock을 반환한다."""
        return self._colbert_lock

    def get_flashrank_ranker(self):
        """FlashRank Ranker를 1회만 로드하고 이후 공유."""
        if self._flashrank_ranker is not None:
            return self._flashrank_ranker
        from flashrank import Ranker
        flashrank_cache_dir = MODELS_DIR / "flashrank"
        flashrank_cache_dir.mkdir(parents=True, exist_ok=True)
        print("[FlashRank 캐시] Ranker 최초 로드 중 (이후 공유)...")
        self._flashrank_ranker = Ranker(
            model_name=self.config.flashrank_model,
            max_length=self.config.flashrank_max_length,
            cache_dir=str(flashrank_cache_dir),
        )
        print("[FlashRank 캐시] Ranker 로드 완료.")
        return self._flashrank_ranker

    @staticmethod
    def _restore_bm25_vocab(strategy, vocab_path: str, fit_docs) -> None:
        """BM25 어휘를 디스크에서 복원하거나 fit_docs로 새로 fit한다.

        Args:
            strategy: _sparse_embeddings를 보유한 DenseSparseStrategy 인스턴스.
            vocab_path: 어휘 JSON 파일 경로.
            fit_docs: vocab_path가 없을 때 BM25 fit에 사용할 문서 목록.
        """
        from rag_bench.strategies.dense_sparse import KoreanBM25Encoder

        if not isinstance(strategy._sparse_embeddings, KoreanBM25Encoder):
            return
        if Path(vocab_path).exists():
            strategy._sparse_embeddings = KoreanBM25Encoder.load(vocab_path)
        else:
            texts = [doc.page_content for doc in fit_docs]
            strategy._sparse_embeddings.fit(texts)

    def get_or_build(self, spec: ComboSpec, child_chunks, reindex=False):
        """base DenseSparseStrategy를 캐시에서 가져오거나 새로 빌드.

        디스크에 기존 Qdrant 인덱스가 있고 reindex=False면 재인덱싱 없이 연결만 한다.
        BM25 어휘는 {qdrant_path}_bm25_vocab.json에 영속화하여 재시작 시 재fit을 생략한다.
        """
        from rag_bench.strategies.dense_sparse import DenseSparseStrategy

        key = spec.index_key
        qdrant_path = str(BENCH_DATA_DIR / f"{QDRANT_DB_PREFIX}{spec.dense}_{spec.sparse}")
        vocab_path = qdrant_path + "_bm25_vocab.json"

        if key in self.cache and not reindex:
            cached_strategy, _ = self.cache[key]
            return cached_strategy

        strategy = DenseSparseStrategy(
            dense_model=spec.dense,
            sparse_type=spec.sparse,
            qdrant_path=qdrant_path,
            device=self.config.dense_device,   # None이면 detect_device() 자동 사용
        )

        qdrant_dir = Path(qdrant_path)
        index_exists = qdrant_dir.exists() and any(qdrant_dir.iterdir())

        if index_exists and not reindex:
            print(f"  [기존 인덱스 재사용] {spec.dense}+{spec.sparse} — {qdrant_path}")
            strategy._ensure_initialized()
            self._restore_bm25_vocab(strategy, vocab_path, child_chunks)
            strategy._is_ready = True
        else:
            strategy.index(child_chunks)
            from rag_bench.strategies.dense_sparse import KoreanBM25Encoder
            if isinstance(strategy._sparse_embeddings, KoreanBM25Encoder):
                strategy._sparse_embeddings.save(vocab_path)

        self.cache[key] = (strategy, qdrant_path)
        return strategy

    def get_or_build_contextual(
        self,
        spec: ComboSpec,
        child_chunks,
        parent_pairs,
        reindex=False,
        pre_enriched: Optional[List] = None,
    ):
        """contextual 전략을 캐시에서 가져오거나 새로 빌드.

        디스크에 기존 Contextual Qdrant 인덱스가 있고 reindex=False면
        LLM 문맥 생성 및 재인덱싱 없이 연결만 한다.
        BM25 어휘는 {ctx_qdrant_path}_bm25_vocab.json에 영속화하여 재시작 시 재fit을 생략한다.

        Args:
            pre_enriched: 사전 생성된 enriched 청크 목록. 제공 시 LLM 호출 없이
                          pre_enriched를 직접 ctx_base에 인덱싱한다.
        """
        from rag_bench.strategies.contextual_retrieval import ContextualRetrievalStrategy
        from rag_bench.strategies.dense_sparse import DenseSparseStrategy

        key = f"ctx:{spec.index_key}"

        if key in self.ctx_cache and not reindex:
            return self.ctx_cache[key]

        ctx_qdrant_path = str(
            BENCH_DATA_DIR / f"{QDRANT_DB_PREFIX}ctx_{spec.dense}_{spec.sparse}"
        )
        ctx_vocab_path = ctx_qdrant_path + "_bm25_vocab.json"

        ctx_base = DenseSparseStrategy(
            dense_model=spec.dense,
            sparse_type=spec.sparse,
            qdrant_path=ctx_qdrant_path,
            device=self.config.dense_device,   # None이면 detect_device() 자동 사용
        )

        # 이미 캐시된 base 전략에서 Dense/Sparse 모델 객체 공유 (재로드 방지)
        base_key = spec.index_key
        if base_key in self.cache:
            cached_base, _ = self.cache[base_key]
            if cached_base._dense_embeddings is not None:
                ctx_base.share_embeddings(
                    dense_embeddings=cached_base._dense_embeddings,
                    sparse_embeddings=cached_base._sparse_embeddings,
                    embedding_dim=cached_base._embedding_dim,
                    use_langchain_sparse=cached_base._use_langchain_sparse,
                )

        strategy = ContextualRetrievalStrategy(
            base_strategy=ctx_base,
            parent_pairs=parent_pairs,
            llm_model=self.config.contextual_llm,
        )

        ctx_qdrant_dir = Path(ctx_qdrant_path)
        ctx_index_exists = ctx_qdrant_dir.exists() and any(ctx_qdrant_dir.iterdir())

        if ctx_index_exists and not reindex:
            print(f"  [기존 Contextual 인덱스 재사용] {spec.dense}+{spec.sparse} — {ctx_qdrant_path}")
            ctx_base._ensure_initialized()
            # pre_enriched 있으면 enriched 텍스트 기준으로 BM25 fit (인덱스와 어휘 일치)
            fit_docs = pre_enriched if pre_enriched is not None else child_chunks
            self._restore_bm25_vocab(ctx_base, ctx_vocab_path, fit_docs)
            ctx_base._is_ready = True
            strategy._is_ready = True
        else:
            from rag_bench.strategies.dense_sparse import KoreanBM25Encoder
            if pre_enriched is not None:
                ctx_base._ensure_initialized()
                if isinstance(ctx_base._sparse_embeddings, KoreanBM25Encoder):
                    texts = [d.page_content for d in pre_enriched]
                    ctx_base._sparse_embeddings.fit(texts)
                ctx_base.index(pre_enriched)
                if isinstance(ctx_base._sparse_embeddings, KoreanBM25Encoder):
                    ctx_base._sparse_embeddings.save(ctx_vocab_path)
                strategy._is_ready = True
            else:
                strategy.index(child_chunks)
                if isinstance(ctx_base._sparse_embeddings, KoreanBM25Encoder):
                    ctx_base._sparse_embeddings.save(ctx_vocab_path)

        self.ctx_cache[key] = strategy
        return strategy
