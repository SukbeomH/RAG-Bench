"""
UpstageEmbedStrategy — Upstage Solar Embeddings 기반 Dense 검색.

Upstage는 문서용(passage)과 쿼리용(query) 모델을 분리 운용:
  - 인덱싱: embedding-passage
  - 쿼리:   embedding-query

환경변수: UPSTAGE_API_KEY
의존성: langchain-upstage, langchain-qdrant
"""

import shutil
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_bench.base import BaseRAGStrategy, StrategyRetriever


class UpstageEmbedStrategy(BaseRAGStrategy):
    """Upstage Solar Embeddings 기반 Dense 검색."""

    def __init__(
        self,
        model: str = "embedding-passage",
        query_model: str = "embedding-query",
        qdrant_path: Optional[str] = None,
        collection_name: str = "upstage_embed",
        k: int = 3,
    ):
        self.model = model
        self.query_model = query_model
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.k = k
        self._vectorstore = None
        self._query_embeddings = None
        self._is_ready = False

    @property
    def name(self) -> str:
        # 예: "Upstage(large)"
        return f"Upstage({self.model.split('-')[-1]})"

    @property
    def description(self) -> str:
        return f"Upstage {self.model} 기반 Dense 검색"

    def index(self, documents: List[Document]) -> None:
        """Upstage passage 임베딩으로 Qdrant 인덱싱."""
        from langchain_upstage import UpstageEmbeddings
        from langchain_qdrant import QdrantVectorStore

        passage_embeddings = UpstageEmbeddings(model=self.model)

        if self.qdrant_path:
            qdrant_dir = Path(self.qdrant_path)
            index_exists = qdrant_dir.exists() and any(qdrant_dir.iterdir())

            if index_exists:
                print(f"[{self.name}] 기존 인덱스 재사용: {self.qdrant_path}")
                self._vectorstore = QdrantVectorStore.from_existing_collection(
                    embedding=passage_embeddings,
                    path=self.qdrant_path,
                    collection_name=self.collection_name,
                )
            else:
                print(f"[{self.name}] 인덱싱 시작: {len(documents)}개 문서")
                self._vectorstore = QdrantVectorStore.from_documents(
                    documents=documents,
                    embedding=passage_embeddings,
                    path=self.qdrant_path,
                    collection_name=self.collection_name,
                )
                print(f"[{self.name}] 인덱싱 완료")
        else:
            self._vectorstore = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=passage_embeddings,
                location=":memory:",
                collection_name=self.collection_name,
            )

        # 쿼리용 모델 별도 초기화
        self._query_embeddings = UpstageEmbeddings(model=self.query_model)
        self._is_ready = True

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """쿼리 모델로 Dense 유사도 검색."""
        if not self._is_ready or self._vectorstore is None:
            raise RuntimeError(f"{self.name}: 인덱싱이 필요합니다.")

        # 쿼리 임베딩 생성 (query 전용 모델 사용)
        query_vector = self._query_embeddings.embed_query(query)
        return self._vectorstore.similarity_search_by_vector(query_vector, k=k)

    def get_retriever(self, k: int = 3) -> BaseRetriever:
        if not self._is_ready or self._vectorstore is None:
            raise RuntimeError(f"{self.name}: 인덱싱이 필요합니다.")
        return StrategyRetriever(strategy=self, k=k)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def cleanup(self) -> None:
        self._vectorstore = None
        self._query_embeddings = None
        self._is_ready = False
        if self.qdrant_path:
            p = Path(self.qdrant_path)
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
