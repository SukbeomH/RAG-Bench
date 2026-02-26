"""
DenseSparseStrategy — Dense + Sparse 하이브리드 검색 전략

Qdrant 벡터 DB에 Dense(벡터) + Sparse(BM25/SPLADE) 하이브리드 검색을 수행한다.
dense_model / sparse_type 독립 파라미터로 조합을 지정한다.
"""

import json
import math
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import SparseVector

from rag_bench.base import BaseRAGStrategy, StrategyRetriever
from rag_bench.config import QDRANT_COLLECTION_NAME, QDRANT_DB_PREFIX
from rag_bench.utils.device import detect_device


# ---------------------------------------------------------------------------
# Sparse Encoders
# ---------------------------------------------------------------------------


class KoreanBM25Encoder:
    """
    한글 BM25 Sparse Encoder (OKt 형태소 분석).

    조합 1, 6에서 사용. KoNLPy OKt로 한국어 형태소 분석 후 BM25 스코어링.
    langchain_qdrant 호환: SparseVector 객체 반환.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        from konlpy.tag import Okt

        self.okt = Okt()
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self._next_id = 0
        self._vocab_lock = threading.Lock()
        print(f"KoreanBM25Encoder initialized (OKt, k1={k1}, b={b})")

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        for word, pos in self.okt.pos(text, stem=True):
            if pos in ["Noun", "Verb", "Adjective", "Foreign", "Alpha"]:
                tokens.append(word.lower())
        return tokens

    def _get_or_create_id(self, token: str) -> int:
        with self._vocab_lock:
            if token not in self.vocab:
                self.vocab[token] = self._next_id
                self._next_id += 1
            return self.vocab[token]

    def fit(self, documents: List[str]):
        doc_lens = []
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_lens.append(len(tokens))
            for token in set(tokens):
                self._get_or_create_id(token)
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        self.doc_count = len(documents)
        self.avg_doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0
        print(f"BM25 fit complete: {self.doc_count} docs, {len(self.vocab)} tokens")

    def _compute_idf(self, token: str) -> float:
        df = self.doc_freqs.get(token, 0)
        if df == 0:
            return 0.0
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def embed_query(self, text: str) -> SparseVector:
        """langchain_qdrant 호환 sparse embedding (query)."""
        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[0], values=[0.0])
        tf_counter = Counter(tokens)
        doc_len = len(tokens)
        indices, values = [], []
        for token, tf in tf_counter.items():
            token_id = self._get_or_create_id(token)
            idf = self._compute_idf(token)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            )
            score = idf * tf_norm
            if score > 0:
                indices.append(token_id)
                values.append(score)
        if not indices:
            return SparseVector(indices=[0], values=[0.0])
        return SparseVector(indices=indices, values=values)

    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        """langchain_qdrant 호환 sparse embedding (documents)."""
        return [self.embed_query(text) for text in texts]

    def save(self, path: str) -> None:
        """어휘 상태를 JSON 파일에 저장한다."""
        state = {
            "vocab": self.vocab,
            "doc_freqs": self.doc_freqs,
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "_next_id": self._next_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        print(f"BM25 vocab saved: {path} ({len(self.vocab)} tokens)")

    @classmethod
    def load(cls, path: str, k1: float = 1.5, b: float = 0.75) -> "KoreanBM25Encoder":
        """JSON 파일에서 어휘 상태를 복원한다."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        instance = cls(k1=k1, b=b)
        instance.vocab = state["vocab"]
        instance.doc_freqs = state["doc_freqs"]
        instance.doc_count = state["doc_count"]
        instance.avg_doc_len = state["avg_doc_len"]
        instance._next_id = state["_next_id"]
        print(f"BM25 vocab loaded: {path} ({len(instance.vocab)} tokens)")
        return instance


class SpladeEncoder:
    """
    SPLADE Sparse Encoder.

    조합 2, 5에서 사용. naver/splade-cocondenser-ensembledistil 기반 Term Expansion.
    langchain_qdrant 호환: SparseVector 객체 반환.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print(f"SPLADE Encoder ready (device={self.device}, vocab={self.vocab_size})")

    def _compute_vector(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        splade = self.torch.log1p(self.torch.relu(logits))
        mask = inputs["attention_mask"].unsqueeze(-1)
        splade = splade * mask
        splade, _ = self.torch.max(splade, dim=1)
        return splade.squeeze(0)

    def _to_sparse_vector(self, text: str) -> SparseVector:
        vec = self._compute_vector(text)
        nonzero = vec > 0
        indices = self.torch.nonzero(nonzero).squeeze(-1).cpu().tolist()
        values = vec[nonzero].cpu().tolist()
        if not indices:
            return SparseVector(indices=[0], values=[0.0])
        return SparseVector(indices=indices, values=values)

    def _compute_vectors_batch(self, texts: List[str]) -> list:
        """배치 단위로 SPLADE 벡터를 계산한다."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        splade = self.torch.log1p(self.torch.relu(logits))
        mask = inputs["attention_mask"].unsqueeze(-1)
        splade = splade * mask
        splade, _ = self.torch.max(splade, dim=1)  # (batch, vocab)
        return splade

    def embed_query(self, text: str) -> SparseVector:
        """langchain_qdrant 호환 sparse embedding (query)."""
        return self._to_sparse_vector(text)

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[SparseVector]:
        """langchain_qdrant 호환 sparse embedding (documents) — 배치 처리."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_vecs = self._compute_vectors_batch(batch)
            for vec in batch_vecs:
                nonzero = vec > 0
                indices = self.torch.nonzero(nonzero).squeeze(-1).cpu().tolist()
                values = vec[nonzero].cpu().tolist()
                if not indices:
                    results.append(SparseVector(indices=[0], values=[0.0]))
                else:
                    results.append(SparseVector(indices=indices, values=values))
        return results


# ---------------------------------------------------------------------------
# 모델 레지스트리
# ---------------------------------------------------------------------------

DENSE_MODELS: Dict[str, str] = {
    # HuggingFace (로컬)
    "kosimcse": "BM-K/KoSimCSE-roberta-multitask",
    "e5": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
    "snowflake-ko": "dragonkue/snowflake-arctic-embed-l-v2.0-ko",  # 한국어 Retrieval SOTA (0.7404)
    # OpenAI API
    "openai-large": "text-embedding-3-large",
    # Upstage API
    "upstage": "embedding-query",
}

DENSE_DIMS: Dict[str, int] = {
    "BM-K/KoSimCSE-roberta-multitask": 768,
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-m3": 1024,
    "dragonkue/snowflake-arctic-embed-l-v2.0-ko": 1024,  # 256 압축 지원, 기본 1024
    "text-embedding-3-large": 3072,
    "embedding-query": 4096,
}

SPARSE_TYPES: List[str] = ["korean_bm25", "splade"]

# 표시용 메타데이터 레지스트리 — 보고서 생성 시 사용
DENSE_MODEL_DISPLAY: Dict[str, Dict[str, str]] = {
    "kosimcse": {
        "display": "KoSimCSE",
        "params": "110M",
        "note": "한국어 SimCSE 대조 학습",
    },
    "e5": {
        "display": "E5-multilingual",
        "params": "560M",
        "note": "다국어 E5, 명령어 prefix 방식",
    },
    "bge-m3": {
        "display": "BGE-M3",
        "params": "570M",
        "note": "100+ 언어, MIRACL 한국어 SOTA",
    },
    "snowflake-ko": {
        "display": "snowflake-ko",
        "params": "600M",
        "note": "한국어 실무 문서 SOTA (법률/금융/의료)",
    },
    "openai-large": {
        "display": "OpenAI (API)",
        "params": "—",
        "note": "text-embedding-3-large, 외부 API",
    },
    "upstage": {
        "display": "Upstage Solar (API)",
        "params": "—",
        "note": "solar-embedding-1-query, 외부 API",
    },
}


# ---------------------------------------------------------------------------
# DenseSparseStrategy
# ---------------------------------------------------------------------------


class DenseSparseStrategy(BaseRAGStrategy):
    """
    Dense + Sparse 하이브리드 검색 전략.

    항상 Hybrid 모드로 동작하는 base retriever.
    Reranker/LLM Support 변형은 Decorator 패턴(ColBERTRerank, FlashRank, ContextualRetrieval)이 처리.

    사용법:
        strategy = DenseSparseStrategy(dense_model="kosimcse", sparse_type="splade")
        strategy = DenseSparseStrategy(dense_model="BM-K/KoSimCSE-roberta-multitask", sparse_type="korean_bm25")

    """

    def __init__(
        self,
        dense_model: str,
        sparse_type: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        device: Optional[str] = None,   # None이면 _init_dense에서 detect_device() 자동 사용
        collection_name: str = QDRANT_COLLECTION_NAME,
    ):
        self._dense_model = DENSE_MODELS.get(dense_model, dense_model)
        self._sparse_type = sparse_type or "korean_bm25"

        if qdrant_path is not None:
            self._qdrant_path = qdrant_path
        else:
            self._qdrant_path = f"{QDRANT_DB_PREFIX}{self._dense_short}_{self._sparse_type}"

        self._collection_name = collection_name
        self._device: Optional[str] = device   # None이면 _init_dense에서 detect_device() 사용

        self._dense_embeddings: Optional[Embeddings] = None
        self._sparse_embeddings: Optional[Union[KoreanBM25Encoder, "SpladeEncoder", "FastEmbedSparse"]] = None
        self._vector_store: Optional["QdrantVectorStore"] = None
        self._client: Optional[QdrantClient] = None
        self._is_ready = False

    @property
    def _dense_short(self) -> str:
        """모델 경로에서 짧은 이름을 추출한다 (예: 'BAAI/bge-m3' → 'bge-m3')."""
        return self._dense_model.split("/")[-1]

    @property
    def name(self) -> str:
        return f"DS({self._dense_short}+{self._sparse_type})"

    @property
    def description(self) -> str:
        return f"Hybrid: {self._dense_short} (dense) + {self._sparse_type} (sparse)"

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def share_embeddings(
        self,
        dense_embeddings,
        sparse_embeddings,
        embedding_dim: int,
        use_langchain_sparse: bool,
    ) -> None:
        """외부에서 Dense/Sparse 임베딩 모델 객체를 주입한다.

        IndexCacheManager 등에서 이미 로드된 모델을 공유할 때 사용.
        주입 후에는 _ensure_initialized()가 Qdrant만 초기화한다.

        Args:
            dense_embeddings: Dense 임베딩 모델 객체.
            sparse_embeddings: Sparse 임베딩 모델 객체.
            embedding_dim: Dense 벡터 차원.
            use_langchain_sparse: langchain sparse 호환 여부.
        """
        self._dense_embeddings = dense_embeddings
        self._sparse_embeddings = sparse_embeddings
        self._embedding_dim = embedding_dim
        self._use_langchain_sparse = use_langchain_sparse

    def _init_dense(self):
        """Dense 임베딩 모델 초기화 (HuggingFace / OpenAI / Upstage 자동 분기)."""
        model_spec = self._dense_model

        if "text-embedding-3" in model_spec or "ada" in model_spec:
            # OpenAI Embeddings API — device 파라미터 불필요
            from langchain_openai import OpenAIEmbeddings
            self._dense_embeddings = OpenAIEmbeddings(model=model_spec)
        elif "embedding-query" in model_spec or "embedding-passage" in model_spec or "solar-embedding" in model_spec:
            # Upstage Solar Embeddings API — device 파라미터 불필요
            from langchain_upstage import UpstageEmbeddings
            self._dense_embeddings = UpstageEmbeddings(model=model_spec)
        else:
            # HuggingFace 로컬 모델
            from langchain_huggingface import HuggingFaceEmbeddings
            _device = self._device if self._device is not None else detect_device()
            self._dense_embeddings = HuggingFaceEmbeddings(
                model_name=model_spec,
                model_kwargs={"device": _device, "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True},
            )

        # 정적 테이블에서 차원 조회, 미등록 모델만 런타임 확인
        if model_spec in DENSE_DIMS:
            self._embedding_dim = DENSE_DIMS[model_spec]
        else:
            test_vec = self._dense_embeddings.embed_query("test")
            self._embedding_dim = len(test_vec)
        print(f"  Dense: {model_spec} ({self._embedding_dim}d)")

    def _init_sparse(self):
        """Sparse 임베딩 모델 초기화."""
        sparse_type = self._sparse_type

        if sparse_type == "korean_bm25":
            self._sparse_embeddings = KoreanBM25Encoder(k1=1.5, b=0.75)
            self._use_langchain_sparse = False
        elif sparse_type == "splade":
            self._sparse_embeddings = SpladeEncoder()
            self._use_langchain_sparse = False
        elif sparse_type == "fastembed_bm25":
            from langchain_qdrant.fastembed_sparse import FastEmbedSparse

            self._sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
            self._use_langchain_sparse = True
        else:
            raise ValueError(f"Unknown sparse type: {sparse_type}")

        print(f"  Sparse: {sparse_type}")

    def _init_qdrant(self):
        """Qdrant 클라이언트 및 벡터 스토어 초기화. ':memory:' 경로는 인메모리 모드로 자동 처리."""
        from langchain_qdrant import QdrantVectorStore
        from langchain_qdrant.qdrant import RetrievalMode

        if self._client is None:
            if self._qdrant_path == ":memory:":
                self._client = QdrantClient(location=":memory:")
            else:
                self._client = QdrantClient(path=self._qdrant_path)

        # 컬렉션 생성
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self._embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
                sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
            )
            print(
                f"  Created collection: {self._collection_name} (dim={self._embedding_dim})"
            )
        else:
            print(f"  Collection exists: {self._collection_name}")

        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self._collection_name,
            embedding=self._dense_embeddings,
            sparse_embedding=self._sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="sparse",
        )

    def _ensure_initialized(self):
        """필요 시 모델 및 DB 초기화."""
        if self._dense_embeddings is None:
            print(f"\n[{self.name}] 초기화 중...")
            self._init_dense()
            self._init_sparse()
            self._init_qdrant()
        elif self._client is None:
            # 모델은 외부 주입됨, Qdrant만 초기화
            print(f"\n[{self.name}] Qdrant 초기화 중 (모델 공유)...")
            self._init_qdrant()

    def index(self, documents: List[Document]) -> None:
        """문서 인덱싱."""
        self._ensure_initialized()

        # 기존 컬렉션 삭제 후 재생성
        assert self._client is not None
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)
        self._init_qdrant()

        # BM25 인코더인 경우 fit 수행
        if isinstance(self._sparse_embeddings, KoreanBM25Encoder):
            doc_texts = [doc.page_content for doc in documents]
            self._sparse_embeddings.fit(doc_texts)

        print(f"  Indexing {len(documents)} documents...")
        self._vector_store.add_documents(documents)
        self._is_ready = True
        print("  Indexing complete.")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """검색 수행."""
        self._ensure_initialized()
        return self._vector_store.similarity_search(query, k=k)

    def get_retriever(self, k: int = 5) -> BaseRetriever:
        """LangChain Retriever 반환."""
        self._ensure_initialized()
        return self._vector_store.as_retriever(search_kwargs={"k": k})

    def cleanup(self, delete_index: bool = False) -> None:
        """Qdrant 리소스 정리. 인덱스는 기본적으로 보존하여 재실행 시 재사용."""
        if delete_index and self._client and self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)
            print(f"  Deleted collection: {self._collection_name}")
        if self._client:
            self._client.close()
            self._client = None
            print(f"  Closed Qdrant client (collection preserved: {self._collection_name})")
