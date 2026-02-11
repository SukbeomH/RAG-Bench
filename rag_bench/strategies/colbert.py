"""
ColBERTStrategy — PyLate 기반 ColBERT Late Interaction 검색 전략

PyLate(Sentence Transformers 기반)를 백엔드로 사용하여
토큰 수준의 MaxSim 연산으로 높은 정확도의 검색을 수행한다.

권장 모델: jinaai/jina-colbert-v2 (89개 언어, 한국어 포함)
"""

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rag_bench.base import BaseRAGStrategy


# ---------------------------------------------------------------------------
# ColBERTRetriever — LangChain BaseRetriever 래퍼
# ---------------------------------------------------------------------------


class ColBERTRetriever(BaseRetriever):
    """ColBERTStrategy를 LangChain Retriever 인터페이스로 래핑."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: Any
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.strategy.retrieve(query, k=self.k)


# ---------------------------------------------------------------------------
# ColBERTStrategy — 메인 클래스
# ---------------------------------------------------------------------------


class ColBERTStrategy(BaseRAGStrategy):
    """
    ColBERT Late Interaction 검색 전략 (PyLate 기반).

    ColBERT는 토큰 수준의 MaxSim 연산으로 Bi-Encoder의 속도와
    Cross-Encoder의 정확도 사이에서 최적의 균형점을 제공한다.

    기본 모드는 brute-force(인덱스 없음)로 소규모 코퍼스에 적합하다.
    use_index=True 시 Voyager ANN 인덱스를 사용하여 대규모 검색을 지원한다.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-colbert-v2",
        use_index: bool = False,
        index_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self._model_name = model_name
        self._use_index = use_index
        self._index_path = index_path or "colbert_index"
        self._device = device
        self._batch_size = batch_size

        self._model: Any = None
        self._documents: List[Document] = []
        self._doc_embeddings: Any = None
        self._index: Any = None
        self._retriever: Any = None
        self._is_ready = False

    @property
    def name(self) -> str:
        short = self._model_name.split("/")[-1]
        mode = "Voyager" if self._use_index else "brute-force"
        return f"ColBERT ({short}, {mode})"

    @property
    def description(self) -> str:
        return (
            "ColBERT Late Interaction 검색. "
            "토큰별 MaxSim으로 Dense+Reranker를 단일 모델로 대체 가능. "
            f"모델: {self._model_name}"
        )

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _detect_device(self) -> str:
        """CUDA → MPS → CPU 자동 감지."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_initialized(self) -> None:
        """필요 시 ColBERT 모델 로드 (lazy)."""
        if self._model is not None:
            return

        from pylate import models

        if self._device is None:
            self._device = self._detect_device()

        print(f"\n[{self.name}] 초기화 중...")
        print(f"  모델: {self._model_name}")
        print(f"  디바이스: {self._device}")

        self._model = models.ColBERT(
            model_name_or_path=self._model_name,
            device=self._device,
            trust_remote_code=True,
        )

        print("  모델 로드 완료.")

    def index(self, documents: List[Document]) -> None:
        """문서 인덱싱: 임베딩 인코딩 + 저장."""
        self._ensure_initialized()

        self._documents = list(documents)
        doc_texts = [doc.page_content for doc in documents]
        doc_ids = [str(i) for i in range(len(documents))]

        print(f"  {len(documents)}개 문서 인코딩 중 (batch_size={self._batch_size})...")

        self._doc_embeddings = self._model.encode(
            sentences=doc_texts,
            batch_size=self._batch_size,
            is_query=False,
            show_progress_bar=True,
        )

        if self._use_index:
            from pylate import indexes, retrieve

            self._index = indexes.Voyager(
                index_folder=self._index_path,
                index_name="colbert",
                override=True,
                embedding_size=self._model.get_sentence_embedding_dimension(),
            )
            self._index.add_documents(
                documents_ids=doc_ids,
                documents_embeddings=self._doc_embeddings,
            )
            self._retriever = retrieve.ColBERT(index=self._index)
            print(f"  Voyager 인덱스 생성 완료: {self._index_path}")

        self._is_ready = True
        print("  인덱싱 완료.")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """쿼리에 대해 관련 문서를 검색한다."""
        if not self._is_ready:
            raise RuntimeError("index()를 먼저 호출하세요.")

        k = min(k, len(self._documents))

        query_embedding = self._model.encode(
            sentences=[query],
            batch_size=1,
            is_query=True,
        )

        if self._use_index:
            results = self._retriever.retrieve(
                queries_embeddings=query_embedding,
                k=k,
                device=self._device,
            )
            # results: [[{"id": "0", "score": 13.8}, ...]]
            ranked = results[0]
            return [
                Document(
                    page_content=self._documents[int(r["id"])].page_content,
                    metadata=self._documents[int(r["id"])].metadata.copy(),
                )
                for r in ranked[:k]
            ]
        else:
            # Brute-force: rank.rerank으로 MaxSim 스코어링
            from pylate import rank

            doc_ids = list(range(len(self._documents)))

            reranked = rank.rerank(
                documents_ids=[doc_ids],
                queries_embeddings=query_embedding,
                documents_embeddings=[self._doc_embeddings],
            )
            # reranked: [[{"id": 0, "score": 13.8}, ...]]
            ranked = reranked[0]
            return [
                Document(
                    page_content=self._documents[r["id"]].page_content,
                    metadata=self._documents[r["id"]].metadata.copy(),
                )
                for r in ranked[:k]
            ]

    def get_retriever(self, k: int = 5) -> BaseRetriever:
        """LangChain 호환 Retriever 객체를 반환한다."""
        return ColBERTRetriever(strategy=self, k=k)

    def cleanup(self) -> None:
        """메모리 및 인덱스 파일 정리."""
        self._doc_embeddings = None
        self._documents = []
        self._index = None
        self._retriever = None
        self._model = None
        self._is_ready = False

        if self._use_index:
            import shutil
            from pathlib import Path

            index_dir = Path(self._index_path)
            if index_dir.exists():
                shutil.rmtree(index_dir)
                print(f"  인덱스 삭제: {self._index_path}")
