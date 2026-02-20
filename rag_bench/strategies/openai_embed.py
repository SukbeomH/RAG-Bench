"""
OpenAIEmbedStrategy — OpenAI text-embedding-3-small/large 기반 Dense 검색.

환경변수: OPENAI_API_KEY
의존성: langchain-openai, langchain-qdrant
"""

import shutil
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_bench.base import BaseRAGStrategy


class OpenAIEmbedStrategy(BaseRAGStrategy):
    """OpenAI text-embedding-3-small/large 기반 순수 Dense 검색."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        qdrant_path: Optional[str] = None,
        collection_name: str = "openai_embed",
        k: int = 3,
    ):
        self.model = model
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.k = k
        self._vectorstore = None
        self._is_ready = False

    @property
    def name(self) -> str:
        return f"OpenAI({self.model})"

    @property
    def description(self) -> str:
        return f"OpenAI {self.model} 기반 Dense 검색"

    def index(self, documents: List[Document]) -> None:
        """OpenAI 임베딩으로 Qdrant 인덱싱."""
        from langchain_openai import OpenAIEmbeddings
        from langchain_qdrant import QdrantVectorStore

        print(f"[{self.name}] 인덱싱 시작: {len(documents)}개 문서")
        embeddings = OpenAIEmbeddings(model=self.model)

        if self.qdrant_path:
            self._vectorstore = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                path=self.qdrant_path,
                collection_name=self.collection_name,
                force_recreate=True,
            )
        else:
            self._vectorstore = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                location=":memory:",
                collection_name=self.collection_name,
            )

        self._is_ready = True
        print(f"[{self.name}] 인덱싱 완료")

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Dense 유사도 검색."""
        if not self._is_ready or self._vectorstore is None:
            raise RuntimeError(f"{self.name}: 인덱싱이 필요합니다.")
        return self._vectorstore.similarity_search(query, k=k)

    def get_retriever(self, k: int = 3) -> BaseRetriever:
        if not self._is_ready or self._vectorstore is None:
            raise RuntimeError(f"{self.name}: 인덱싱이 필요합니다.")
        return self._vectorstore.as_retriever(search_kwargs={"k": k})

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def cleanup(self) -> None:
        self._vectorstore = None
        self._is_ready = False
        if self.qdrant_path:
            p = Path(self.qdrant_path)
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
