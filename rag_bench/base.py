"""
BaseRAGStrategy — 모든 RAG 전략의 추상 기본 클래스

Strategy Pattern의 핵심 인터페이스를 정의한다.
새로운 RAG 방식을 추가하려면 이 클래스를 상속하여 구현하면 된다.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class BaseRAGStrategy(ABC):
    """
    RAG 전략 기본 인터페이스.

    모든 RAG 방식 (Dense+Sparse, ColBERT, GraphRAG 등)은
    이 클래스를 상속하여 통일된 인터페이스로 벤치마크에 참여한다.

    구현 필수 메서드:
        - name: 전략 이름 (예: "Dense+Sparse Combo 1")
        - description: 전략 설명
        - index(): 문서 인덱싱
        - retrieve(): 쿼리 검색
        - get_retriever(): LangChain Retriever 반환
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """전략 이름 (벤치마크 결과 표시용)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """전략의 간략한 설명."""
        ...

    @abstractmethod
    def index(self, documents: List[Document]) -> None:
        """
        문서를 인덱싱한다.

        Args:
            documents: 인덱싱할 LangChain Document 목록.
        """
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        쿼리에 대해 관련 문서를 검색한다.

        Args:
            query: 검색 쿼리.
            k: 반환할 최대 문서 수.

        Returns:
            관련 Document 목록.
        """
        ...

    @abstractmethod
    def get_retriever(self, k: int = 5) -> BaseRetriever:
        """
        LangChain 호환 Retriever 객체를 반환한다.
        LangGraph 에이전트에서 도구로 바인딩할 때 사용.

        Args:
            k: 검색 시 반환할 문서 수.

        Returns:
            BaseRetriever 구현체.
        """
        ...

    @property
    def is_ready(self) -> bool:
        """인덱싱이 완료되어 검색 가능한 상태인지 반환."""
        return False

    def cleanup(self) -> None:
        """인덱스 및 리소스 정리 (선택적 구현)."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
