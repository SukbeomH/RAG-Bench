"""
GraphRAGStrategy — 그래프 기반 RAG 전략 (스텁)

향후 NodeRAG/LightRAG 패키지 통합 시 구현 예정.
현재는 인터페이스만 정의하여 벤치마크 프레임워크에 등록 가능하도록 한다.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_bench.base import BaseRAGStrategy


class GraphRAGStrategy(BaseRAGStrategy):
    """
    그래프 기반 RAG 전략 (NodeRAG 개념 기반).

    NodeRAG는 7가지 이질적 노드 타입으로 구성된 그래프를 구축하고,
    Dual Search + Shallow PPR로 다중 홉 추론 경로를 탐색한다.

    TODO:
        - NodeRAG 또는 LightRAG 패키지 통합
        - 한국어 NER/관계 추출 성능 검증
        - 인덱싱 비용 분석 (LLM API 호출)
        - 하이브리드 아키텍처 PoC (질문 라우터 + 이중 파이프라인)
    """

    def __init__(self, backend: str = "noderag"):
        self._backend = backend
        self._is_ready = False

    @property
    def name(self) -> str:
        return f"GraphRAG ({self._backend})"

    @property
    def description(self) -> str:
        return (
            "이질적 그래프 기반 RAG. "
            "7가지 노드 타입 + Dual Search + Shallow PPR로 다중 홉 추론 지원. "
            f"백엔드: {self._backend}"
        )

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def index(self, documents: List[Document]) -> None:
        """문서 인덱싱 (미구현 — 스텁)."""
        raise NotImplementedError(
            f"GraphRAGStrategy({self._backend})는 아직 구현되지 않았습니다. "
            "NodeRAG/LightRAG 패키지 통합 후 사용 가능합니다."
        )

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """검색 수행 (미구현 — 스텁)."""
        raise NotImplementedError(
            "GraphRAGStrategy.retrieve()는 아직 구현되지 않았습니다."
        )

    def get_retriever(self, k: int = 5) -> BaseRetriever:
        """LangChain Retriever 반환 (미구현 — 스텁)."""
        raise NotImplementedError(
            "GraphRAGStrategy.get_retriever()는 아직 구현되지 않았습니다."
        )
