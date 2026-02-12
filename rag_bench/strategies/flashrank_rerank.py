"""
FlashRankRerankStrategy — FlashRank 경량 리랭커를 활용한 2단계 검색 전략

[Query] → base_strategy.retrieve(k=rerank_n) → 후보 N개
                                                    ↓
                                        FlashRank rerank (CPU, ~4MB)
                                                    ↓
                                        상위 k개 반환

ColBERTRerank 대비 장점:
  - Torch/Transformers 불필요 (ONNX 기반)
  - CPU 전용, 초경량 (~4MB 기본 모델)
  - 100개+ 후보 리랭킹에도 밀리초 단위 응답

모델 옵션:
  - ms-marco-TinyBERT-L-2-v2  (~4MB, 기본)
  - ms-marco-MiniLM-L-12-v2   (~34MB, 최고 성능)
  - ms-marco-MultiBERT-L-12   (~150MB, 100+ 언어, 한국어 지원)

Usage:
    base = DenseSparseStrategy(combo_id=1)
    strategy = FlashRankRerankStrategy(base_strategy=base, model_name="ms-marco-MultiBERT-L-12")
    strategy.index(child_chunks)
    results = strategy.retrieve("질문", k=3)
"""

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag_bench.base import BaseRAGStrategy


# ---------------------------------------------------------------------------
# FlashRankRerankRetriever — LangChain BaseRetriever 래퍼
# ---------------------------------------------------------------------------


class FlashRankRerankRetriever(BaseRetriever):
    """FlashRankRerankStrategy를 LangChain Retriever 인터페이스로 래핑."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: Any
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.strategy.retrieve(query, k=self.k)


# ---------------------------------------------------------------------------
# FlashRankRerankStrategy — 메인 클래스
# ---------------------------------------------------------------------------


class FlashRankRerankStrategy(BaseRAGStrategy):
    """
    FlashRank 경량 리랭커를 활용한 2단계 검색 전략.

    1단계: base_strategy로 rerank_n개 후보를 검색
    2단계: FlashRank로 쿼리-문서 관련성 스코어링 후 재정렬
    3단계: 상위 k개 반환

    FlashRank는 ONNX 기반으로 Torch 없이 CPU에서 동작한다.
    ColBERTRerank 대비 10~100배 빠른 리랭킹이 가능하다.
    """

    def __init__(
        self,
        base_strategy: BaseRAGStrategy,
        model_name: str = "ms-marco-MultiBERT-L-12",
        rerank_n: int = 20,
        max_length: int = 512,
    ):
        self._base_strategy = base_strategy
        self._model_name = model_name
        self._rerank_n = rerank_n
        self._max_length = max_length

        self._ranker: Any = None
        self._is_ready = False

    @property
    def name(self) -> str:
        return f"FlashRank Rerank ({self._base_strategy.name})"

    @property
    def description(self) -> str:
        return (
            f"FlashRank 경량 리랭킹 (top-{self._rerank_n} → top-k). "
            f"1차 검색: {self._base_strategy.name}. "
            f"모델: {self._model_name}"
        )

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _ensure_initialized(self) -> None:
        """FlashRank Ranker lazy 로드."""
        if self._ranker is not None:
            return

        from flashrank import Ranker

        print(f"\n[{self.name}] FlashRank 리랭커 초기화 중...")
        print(f"  모델: {self._model_name}")

        self._ranker = Ranker(
            model_name=self._model_name,
            max_length=self._max_length,
        )

        print("  모델 로드 완료 (ONNX, CPU).")

    def index(self, documents: List[Document]) -> None:
        """base_strategy에 인덱싱을 위임하고, FlashRank 모델을 로드한다."""
        self._base_strategy.index(documents)
        self._ensure_initialized()
        self._is_ready = True
        print(f"[{self.name}] 인덱싱 완료 (리랭커 준비됨).")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        2단계 검색: base_strategy로 후보 추출 → FlashRank 리랭킹.

        Args:
            query: 검색 쿼리.
            k: 최종 반환할 문서 수.

        Returns:
            FlashRank로 재정렬된 상위 k개 Document 목록.
        """
        if not self._is_ready:
            raise RuntimeError("index()를 먼저 호출하세요.")

        from flashrank import RerankRequest

        # 1단계: base_strategy로 rerank_n개 후보 검색
        candidates = self._base_strategy.retrieve(query, k=self._rerank_n)

        if not candidates:
            return []

        k = min(k, len(candidates))

        # 2단계: FlashRank 리랭킹
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(candidates)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        reranked = self._ranker.rerank(rerank_request)

        # 3단계: 상위 k개 반환
        return [
            Document(
                page_content=r["text"],
                metadata=r.get("meta", {}),
            )
            for r in reranked[:k]
        ]

    def get_retriever(self, k: int = 5) -> BaseRetriever:
        """LangChain 호환 Retriever 객체를 반환한다."""
        return FlashRankRerankRetriever(strategy=self, k=k)

    def cleanup(self) -> None:
        """base_strategy 및 FlashRank 리소스 정리."""
        self._base_strategy.cleanup()
        self._ranker = None
        self._is_ready = False
