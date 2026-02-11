"""
GraphRAGStrategy — LightRAG 기반 그래프 RAG 전략

LightRAG는 엔티티-관계 지식 그래프를 구축하여 local/global/hybrid 모드로 검색하는
Graph RAG 프레임워크다.

아키텍처:
    [Documents] → LightRAG.ainsert() → 지식 그래프 구축 (LLM 기반 엔티티/관계 추출)
                                              ↓
                                        NetworkX 그래프 + NanoVectorDB 저장

    [Query] → LightRAG.aquery(only_need_context=True)
                       ↓
              mode="hybrid" (local + global 결합)
                       ↓
              엔티티/관계/청크 컨텍스트 반환 → List[Document] 변환
"""

import asyncio
import os
import shutil
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag_bench.base import BaseRAGStrategy


# ---------------------------------------------------------------------------
# GraphRAGRetriever — LangChain BaseRetriever 래퍼
# ---------------------------------------------------------------------------


class GraphRAGRetriever(BaseRetriever):
    """GraphRAGStrategy를 LangChain Retriever 인터페이스로 래핑."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: Any
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.strategy.retrieve(query, k=self.k)


# ---------------------------------------------------------------------------
# GraphRAGStrategy — 메인 클래스
# ---------------------------------------------------------------------------


class GraphRAGStrategy(BaseRAGStrategy):
    """
    LightRAG 기반 그래프 RAG 전략.

    LLM을 사용하여 문서에서 엔티티와 관계를 추출해 지식 그래프를 구축하고,
    local/global/hybrid 모드로 검색한다.

    주의: index() 시 LLM API 호출이 발생하여 비용이 소모된다.
    """

    def __init__(
        self,
        mode: str = "hybrid",
        working_dir: Optional[str] = None,
        llm_model: str = "gpt-4.1-nano",
        top_k: int = 60,
    ):
        self._mode = mode
        self._working_dir = working_dir or "lightrag_index"
        self._llm_model = llm_model
        self._top_k = top_k

        self._rag: Any = None
        self._documents: List[Document] = []
        self._is_ready = False
        self._loop: Any = None  # 영속 이벤트 루프 (LightRAG 워커 호환)

    @property
    def name(self) -> str:
        return f"GraphRAG (LightRAG, {self._mode})"

    @property
    def description(self) -> str:
        return (
            f"LightRAG 기반 그래프 RAG. "
            f"모드: {self._mode}, LLM: {self._llm_model}, top_k: {self._top_k}. "
            f"엔티티-관계 지식 그래프 구축 후 검색."
        )

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # -- async → sync 래핑 ------------------------------------------------

    def _run_async(self, coro):
        """async 코루틴을 sync 컨텍스트에서 실행한다.

        LightRAG 내부 워커(PriorityQueue, Lock 등)가 이벤트 루프에 바인딩되므로,
        인스턴스 수명 동안 동일한 루프를 재사용해야 한다.
        Jupyter 등 이미 이벤트 루프가 실행 중인 환경에서는
        nest_asyncio를 적용하여 중첩 루프를 허용한다.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running and running.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            self._loop = running
            return running.run_until_complete(coro)

        # 영속 루프 생성 또는 재사용
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    # -- lazy 초기화 -------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """LightRAG 인스턴스를 lazy 생성하고 스토리지를 초기화한다."""
        if self._rag is not None:
            return

        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.kg.shared_storage import initialize_pipeline_status

        os.makedirs(self._working_dir, exist_ok=True)

        print(f"\n[{self.name}] LightRAG 초기화 중...")
        print(f"  작업 디렉토리: {self._working_dir}")
        print(f"  LLM: {self._llm_model}")
        print(f"  모드: {self._mode}")

        # llm_model 파라미터에 따라 LLM 함수 생성
        llm_model_name = self._llm_model

        async def llm_model_func(
            prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
        ) -> str:
            return await openai_complete_if_cache(
                llm_model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        self._rag = LightRAG(
            working_dir=self._working_dir,
            embedding_func=openai_embed,
            llm_model_func=llm_model_func,
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="JsonDocStatusStorage",
        )

        async def _init():
            await self._rag.initialize_storages()
            await initialize_pipeline_status()

        self._run_async(_init())
        print("  초기화 완료.")

    # -- BaseRAGStrategy 구현 ----------------------------------------------

    def index(self, documents: List[Document]) -> None:
        """문서를 LightRAG에 삽입하여 지식 그래프를 구축한다.

        주의: LLM API를 호출하여 엔티티/관계를 추출하므로 비용이 발생한다.
        """
        self._ensure_initialized()

        texts = [doc.page_content for doc in documents]
        print(f"[{self.name}] {len(texts)}개 문서 삽입 중 (LLM 호출 발생)...")

        self._run_async(self._rag.ainsert(texts))

        self._documents = list(documents)
        self._is_ready = True
        print(f"[{self.name}] 인덱싱 완료.")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """쿼리에 대해 지식 그래프 기반 검색을 수행한다."""
        if not self._is_ready:
            raise RuntimeError("index()를 먼저 호출하세요.")

        from lightrag import QueryParam

        context = self._run_async(
            self._rag.aquery(
                query,
                param=QueryParam(
                    mode=self._mode,
                    only_need_context=True,
                    top_k=self._top_k,
                ),
            )
        )

        if not context or not context.strip():
            return []

        # 컨텍스트를 섹션별로 분리하여 Document로 변환
        sections = [s.strip() for s in context.split("\n\n") if s.strip()]
        sections = sections[:k]

        return [
            Document(
                page_content=section,
                metadata={"source": "lightrag", "mode": self._mode},
            )
            for section in sections
        ]

    def get_retriever(self, k: int = 5) -> BaseRetriever:
        """LangChain 호환 Retriever 객체를 반환한다."""
        return GraphRAGRetriever(strategy=self, k=k)

    def cleanup(self) -> None:
        """LightRAG 스토리지 종료 및 작업 디렉토리 삭제."""
        if self._rag is not None:
            try:
                self._run_async(self._rag.finalize_storages())
            except Exception:
                pass

        if os.path.exists(self._working_dir):
            shutil.rmtree(self._working_dir)

        self._rag = None
        self._is_ready = False

        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None
