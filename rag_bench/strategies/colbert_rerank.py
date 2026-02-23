"""
ColBERTRerankStrategy — 기존 검색 결과를 ColBERT MaxSim으로 재정렬하는 2단계 전략

[Query] → base_strategy.retrieve(k=rerank_n) → 후보 N개
                                                    ↓
                                        ColBERT encode (query + N docs)
                                                    ↓
                                        rank.rerank() → MaxSim 스코어링
                                                    ↓
                                        상위 k개 반환

ColBERT를 검색 자체에 쓰는 ColBERTStrategy와 달리,
이 전략은 임의의 1차 검색 전략 위에 ColBERT 리랭킹만 얹는다.
전체 코퍼스 인코딩 없이 후보 N개만 인코딩하므로 효율적이다.
"""

import threading
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_bench.base import BaseRAGStrategy, StrategyRetriever
from rag_bench.utils.device import detect_device

# ColBERT 모델은 스레드 비안전(PyTorch 배치 텐서 공유) — 전역 Lock으로 직렬화
_COLBERT_INFERENCE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# ColBERTRerankStrategy — 메인 클래스
# ---------------------------------------------------------------------------


class ColBERTRerankStrategy(BaseRAGStrategy):
    """
    기존 검색 전략의 결과를 ColBERT MaxSim으로 재정렬하는 리랭킹 전략.

    1단계: base_strategy로 rerank_n개 후보를 검색
    2단계: ColBERT로 쿼리+후보를 인코딩하여 MaxSim 재정렬
    3단계: 상위 k개 반환

    ColBERT는 후보 N개만 인코딩하므로 전체 코퍼스 인코딩이 불필요하다.
    """

    def __init__(
        self,
        base_strategy: BaseRAGStrategy,
        model_name: str = "jinaai/jina-colbert-v2",
        rerank_n: int = 20,
        device: Optional[str] = None,
        batch_size: int = 32,
        shared_model: Any = None,
    ):
        self._base_strategy = base_strategy
        self._model_name = model_name
        self._rerank_n = rerank_n
        self._device = device
        self._batch_size = batch_size

        self._model: Any = shared_model
        self._is_shared_model = shared_model is not None
        self._is_ready = shared_model is not None

    @property
    def name(self) -> str:
        return f"ColBERT Rerank ({self._base_strategy.name})"

    @property
    def description(self) -> str:
        return (
            f"ColBERT MaxSim 리랭킹 (top-{self._rerank_n} → top-k). "
            f"1차 검색: {self._base_strategy.name}. "
            f"모델: {self._model_name}"
        )

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _ensure_initialized(self) -> None:
        """ColBERT 모델만 lazy 로드 (문서 인코딩 없음)."""
        if self._model is not None:
            return

        from pylate import models

        if self._device is None:
            self._device = detect_device()

        print(f"\n[{self.name}] ColBERT 리랭커 초기화 중...")
        print(f"  모델: {self._model_name}")
        print(f"  디바이스: {self._device}")

        self._model = models.ColBERT(
            model_name_or_path=self._model_name,
            device=self._device,
            trust_remote_code=True,
        )

        print("  모델 로드 완료.")

    def index(self, documents: List[Document]) -> None:
        """base_strategy에 인덱싱을 위임하고, ColBERT 모델을 로드한다."""
        self._base_strategy.index(documents)
        self._ensure_initialized()
        self._is_ready = True
        print(f"[{self.name}] 인덱싱 완료 (리랭커 준비됨).")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        2단계 검색: base_strategy로 후보 추출 → ColBERT MaxSim 리랭킹.

        Args:
            query: 검색 쿼리.
            k: 최종 반환할 문서 수.

        Returns:
            ColBERT MaxSim으로 재정렬된 상위 k개 Document 목록.
        """
        if not self._is_ready:
            raise RuntimeError("index()를 먼저 호출하세요.")

        from pylate import rank

        # 1단계: base_strategy로 rerank_n개 후보 검색
        candidates = self._base_strategy.retrieve(query, k=self._rerank_n)

        if not candidates:
            return []

        # k를 후보 수 이하로 제한
        k = min(k, len(candidates))

        # 2단계: ColBERT 인코딩 + 3단계: MaxSim 리랭킹
        # 공유 모델은 스레드 비안전 — Lock으로 직렬화
        doc_texts = [doc.page_content for doc in candidates]
        doc_ids = list(range(len(candidates)))
        with _COLBERT_INFERENCE_LOCK:
            query_embedding = self._model.encode(
                sentences=[query],
                batch_size=1,
                is_query=True,
            )
            doc_embeddings = self._model.encode(
                sentences=doc_texts,
                batch_size=self._batch_size,
                is_query=False,
            )
            reranked = rank.rerank(
                documents_ids=[doc_ids],
                queries_embeddings=query_embedding,
                documents_embeddings=[doc_embeddings],
            )

        # reranked: [[{"id": 0, "score": 13.8}, ...]]
        ranked = reranked[0]

        return [
            Document(
                page_content=candidates[r["id"]].page_content,
                metadata=candidates[r["id"]].metadata.copy(),
            )
            for r in ranked[:k]
        ]

    def get_retriever(self, k: int = 5) -> BaseRetriever:
        """LangChain 호환 Retriever 객체를 반환한다."""
        return StrategyRetriever(strategy=self, k=k)

    def cleanup(self) -> None:
        """base_strategy 리소스 정리 (공유 모델은 유지)."""
        self._base_strategy.cleanup()
        if not self._is_shared_model:
            self._model = None
        self._is_ready = False
