"""
UpstageEmbedStrategy — Upstage Solar Embeddings 기반 Dense 검색.

Upstage는 문서용(passage)과 쿼리용(query) 모델을 분리 운용:
  - 인덱싱: embedding-passage
  - 쿼리:   embedding-query

환경변수: UPSTAGE_API_KEY
의존성: langchain-upstage, langchain-qdrant
"""

import shutil
import threading
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from autorag_retrieval.base import BaseRAGStrategy, StrategyRetriever
from autorag_retrieval.run_tracker import TokenUsage


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
        # 토큰 추적: indexing / query 별도 집계
        self._token_indexing = TokenUsage()
        self._token_query = TokenUsage()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        # 예: "Upstage(large)"
        return f"Upstage({self.model.split('-')[-1]})"

    @property
    def description(self) -> str:
        return f"Upstage {self.model} 기반 Dense 검색"

    def _track_embed_tokens(self, target: TokenUsage, texts: List[str], is_query: bool = False) -> List:
        """Upstage API를 직접 호출하고 토큰 사용량을 집계한다."""
        import os
        import httpx

        api_key = os.environ.get("UPSTAGE_API_KEY", "")
        model = self.query_model if is_query else self.model
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        vectors = []
        for text in texts:
            try:
                resp = httpx.post(
                    "https://api.upstage.ai/v1/solar/embeddings",
                    headers=headers,
                    json={"model": model, "input": text},
                    timeout=30,
                )
                data = resp.json()
                usage = data.get("usage", {})
                with self._lock:
                    target.prompt_tokens += usage.get("prompt_tokens", 0)
                    target.total_tokens += usage.get("total_tokens", 0)
                    target.num_calls += 1
                vectors.append(data["data"][0]["embedding"])
            except Exception:
                vectors.append(None)
        return vectors

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
                # 토큰 추적을 위해 직접 임베딩 후 Qdrant 저장
                texts = [d.page_content for d in documents]
                self._track_embed_tokens(self._token_indexing, texts, is_query=False)
                self._vectorstore = QdrantVectorStore.from_documents(
                    documents=documents,
                    embedding=passage_embeddings,
                    path=self.qdrant_path,
                    collection_name=self.collection_name,
                )
                print(f"[{self.name}] 인덱싱 완료 (Upstage tokens: {self._token_indexing.total_tokens:,})")
        else:
            texts = [d.page_content for d in documents]
            self._track_embed_tokens(self._token_indexing, texts, is_query=False)
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

        # 쿼리 임베딩 생성 + 토큰 추적
        self._track_embed_tokens(self._token_query, [query], is_query=True)
        query_vector = self._query_embeddings.embed_query(query)
        return self._vectorstore.similarity_search_by_vector(query_vector, k=k)

    def get_raw_token_usages(self):
        """인덱싱/쿼리 TokenUsage 객체를 그대로 반환한다.

        RunTracker.add_tokens_breakdown() 등 TokenUsage 객체를 직접 필요로 하는
        호출자를 위해 제공한다. 비공개 속성의 직접 접근을 방지한다.

        Returns:
            (indexing_usage, query_usage) 튜플
        """
        return self._token_indexing, self._token_query

    def get_token_usage(self) -> dict:
        """인덱싱/쿼리 토큰 사용량을 반환한다."""
        return {
            "indexing": {
                "prompt_tokens": self._token_indexing.prompt_tokens,
                "total_tokens": self._token_indexing.total_tokens,
                "num_calls": self._token_indexing.num_calls,
            },
            "query": {
                "prompt_tokens": self._token_query.prompt_tokens,
                "total_tokens": self._token_query.total_tokens,
                "num_calls": self._token_query.num_calls,
            },
        }

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
