"""
ContextualRetrievalStrategy — Anthropic 제안 Contextual Retrieval 기법 구현

인덱싱 시 LLM이 각 청크에 문맥 요약(Contextual Prefix)을 부착하여
검색 정확도를 높인다. Parent 청크를 문서 문맥으로 사용.

[Parent Context + Child Chunk] → LLM → Contextual Prefix
                                         ↓
                         Enriched Chunk = Prefix + Original Chunk
                                         ↓
                         base_strategy.index(enriched_chunks)

Anthropic 벤치마크: 검색 실패율 49% 감소, 리랭킹 결합 시 67% 감소.

Usage:
    base = DenseSparseStrategy(combo_id=3)
    strategy = ContextualRetrievalStrategy(
        base_strategy=base,
        parent_pairs=parent_pairs,
    )
    strategy.index(child_chunks)
    results = strategy.retrieve("질문")
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag_bench.base import BaseRAGStrategy
from rag_bench.config import BENCH_DATA_DIR

# Anthropic 공식 프롬프트 (원문 유지, 한국어 응답 유도)
_CONTEXT_PROMPT = """\
<document>
{parent_content}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>
Please give a short succinct context to situate this chunk within the \
overall document for the purposes of improving search retrieval of the chunk. \
Answer only with the succinct context and nothing else. \
Respond in the same language as the document."""


# ---------------------------------------------------------------------------
# LangChain Retriever 래퍼
# ---------------------------------------------------------------------------


class ContextualRetrievalRetriever(BaseRetriever):
    """ContextualRetrievalStrategy를 LangChain Retriever 인터페이스로 래핑."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: Any
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.strategy.retrieve(query, k=self.k)


# ---------------------------------------------------------------------------
# ContextualRetrievalStrategy
# ---------------------------------------------------------------------------


class ContextualRetrievalStrategy(BaseRAGStrategy):
    """
    Anthropic Contextual Retrieval 전략.

    인덱싱 시 각 청크에 LLM이 생성한 문맥 요약을 부착하여 검색 품질을 개선.
    임의의 base_strategy와 조합 가능 (DenseSparse, ColBERT 등).

    Args:
        base_strategy: 인덱싱·검색을 위임할 기반 전략.
        parent_pairs: (parent_id, Document) 쌍 목록. 문맥 생성에 사용.
        llm_model: 문맥 생성용 LLM 모델명 (기본: gpt-4o-mini).
        max_context_tokens: 문맥 프롬프트에 포함할 Parent 최대 문자 수.
        cache_dir: 문맥 캐시 저장 디렉토리 (기본: _benchdata).
    """

    def __init__(
        self,
        base_strategy: BaseRAGStrategy,
        parent_pairs: List[Tuple[str, Document]],
        llm_model: str = "gpt-4o-mini",
        max_context_tokens: int = 4000,
        cache_dir: Optional[str] = None,
    ):
        self._base_strategy = base_strategy
        self._parent_lookup: Dict[str, Document] = {
            pid: doc for pid, doc in parent_pairs
        }
        self._llm_model = llm_model
        self._max_context_tokens = max_context_tokens
        self._cache_dir = Path(cache_dir) if cache_dir else BENCH_DATA_DIR
        self._is_ready = False
        self._llm: Any = None
        self._stats = {"cached": 0, "generated": 0, "failed": 0}

    @property
    def name(self) -> str:
        return f"Contextual Retrieval ({self._base_strategy.name})"

    @property
    def description(self) -> str:
        return (
            f"Anthropic Contextual Retrieval. "
            f"LLM({self._llm_model})이 각 청크에 문맥 요약을 부착. "
            f"기반 전략: {self._base_strategy.name}"
        )

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _ensure_llm(self) -> None:
        """LLM lazy 초기화."""
        if self._llm is not None:
            return
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=self._llm_model,
            temperature=0,
            max_tokens=256,
        )

    @staticmethod
    def _chunk_hash(content: str) -> str:
        """청크 내용의 SHA-256 해시 (캐시 키)."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _load_cache(self) -> Dict[str, str]:
        """디스크에서 문맥 캐시를 로드한다."""
        cache_path = self._cache_dir / "contextual_cache.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self, cache: Dict[str, str]) -> None:
        """문맥 캐시를 디스크에 저장한다."""
        cache_path = self._cache_dir / "contextual_cache.json"
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _generate_prefix(self, parent_content: str, chunk_content: str) -> str:
        """LLM으로 하나의 청크에 대한 문맥 요약을 생성한다."""
        self._ensure_llm()

        # Parent 내용을 최대 길이로 잘라냄
        truncated_parent = parent_content[: self._max_context_tokens]

        prompt = _CONTEXT_PROMPT.format(
            parent_content=truncated_parent,
            chunk_content=chunk_content,
        )

        response = self._llm.invoke(prompt)
        return response.content.strip()

    def _enrich_chunks(
        self, child_chunks: List[Document]
    ) -> List[Document]:
        """
        모든 Child 청크에 문맥 요약을 부착한다.

        캐시를 활용하여 이미 생성된 문맥은 재사용한다.
        """
        cache = self._load_cache()
        enriched: List[Document] = []
        total = len(child_chunks)

        # 캐시 히트율 사전 추정: 모든 청크가 캐시에 있으면 간략 출력
        all_cached = all(
            self._chunk_hash(c.page_content) in cache for c in child_chunks
        )

        if all_cached:
            print(f"  [Contextual Retrieval] {total}개 청크 — 전체 캐시 히트 (LLM 호출 없음)")
        else:
            print(f"  [Contextual Retrieval] {total}개 청크에 문맥 요약 부착 중...")
            print(f"  LLM: {self._llm_model}")
            print(f"  기존 캐시: {len(cache)}개")

        for i, chunk in enumerate(child_chunks):
            chunk_key = self._chunk_hash(chunk.page_content)

            if chunk_key in cache:
                prefix = cache[chunk_key]
                self._stats["cached"] += 1
            else:
                parent_id = chunk.metadata.get("parent_id", "")
                parent_doc = self._parent_lookup.get(parent_id)

                if parent_doc:
                    try:
                        prefix = self._generate_prefix(
                            parent_doc.page_content, chunk.page_content
                        )
                        cache[chunk_key] = prefix
                        self._stats["generated"] += 1
                    except Exception as e:
                        print(f"    경고: 청크 {i} 문맥 생성 실패 ({e})")
                        prefix = ""
                        self._stats["failed"] += 1
                else:
                    prefix = ""
                    self._stats["failed"] += 1

            # 문맥 요약 + 원본 내용 결합
            if prefix:
                enriched_content = f"{prefix}\n\n{chunk.page_content}"
            else:
                enriched_content = chunk.page_content

            enriched_doc = Document(
                page_content=enriched_content,
                metadata={
                    **chunk.metadata,
                    "contextual_prefix": prefix,
                    "original_content": chunk.page_content,
                },
            )
            enriched.append(enriched_doc)

            # 진행 상황 출력 + 중간 캐시 저장 (LLM 생성 시에만)
            if not all_cached and ((i + 1) % 10 == 0 or i == total - 1):
                print(
                    f"    진행: {i + 1}/{total} "
                    f"(캐시: {self._stats['cached']}, "
                    f"생성: {self._stats['generated']}, "
                    f"실패: {self._stats['failed']})"
                )
                if self._stats["generated"] > 0:
                    self._save_cache(cache)

        # 최종 캐시 저장 (신규 생성분이 있을 때만)
        if self._stats["generated"] > 0:
            self._save_cache(cache)
        print(
            f"  [Contextual Retrieval] 완료 — "
            f"캐시 {self._stats['cached']}건, "
            f"신규 {self._stats['generated']}건, "
            f"실패 {self._stats['failed']}건"
        )

        return enriched

    # ------------------------------------------------------------------
    # BaseRAGStrategy 인터페이스
    # ------------------------------------------------------------------

    def index(self, documents: List[Document]) -> None:
        """
        문서 인덱싱: 문맥 요약 부착 → base_strategy에 위임.

        Args:
            documents: 인덱싱할 Child 청크 목록.
        """
        print(f"\n[{self.name}] 인덱싱 시작")

        # 1단계: 문맥 요약 부착
        enriched = self._enrich_chunks(documents)

        # 2단계: base_strategy에 인덱싱 위임
        print(f"  기반 전략({self._base_strategy.name})에 인덱싱 위임...")
        self._base_strategy.index(enriched)

        self._is_ready = True
        print(f"[{self.name}] 인덱싱 완료")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        검색: base_strategy에 위임.

        검색 결과의 page_content는 enriched 버전이지만,
        metadata에 original_content가 보존되어 있다.
        """
        if not self._is_ready:
            raise RuntimeError("index()를 먼저 호출하세요.")

        results = self._base_strategy.retrieve(query, k=k)

        # 검색 결과에서 original_content를 복원 (LLM 응답 생성 시 원본 사용)
        restored = []
        for doc in results:
            original = doc.metadata.get("original_content")
            if original:
                restored.append(
                    Document(
                        page_content=original,
                        metadata={
                            k: v
                            for k, v in doc.metadata.items()
                            if k != "original_content"
                        },
                    )
                )
            else:
                restored.append(doc)

        return restored

    def get_retriever(self, k: int = 5) -> BaseRetriever:
        """LangChain 호환 Retriever 객체를 반환한다."""
        return ContextualRetrievalRetriever(strategy=self, k=k)

    def cleanup(self) -> None:
        """base_strategy 리소스 정리."""
        self._base_strategy.cleanup()
        self._llm = None
        self._is_ready = False
