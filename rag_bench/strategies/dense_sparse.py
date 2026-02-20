"""
DenseSparseStrategy — Dense + Sparse 하이브리드 검색 전략

노트북의 6가지 임베딩 조합을 모듈화한 구현.
Qdrant 벡터 DB에 Dense(벡터) + Sparse(BM25/SPLADE) 하이브리드 검색을 수행한다.

v2: dense_model / sparse_type 독립 파라미터화.
    combo_id는 하위 호환용으로 유지.
"""

import math
import threading
from collections import Counter
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import SparseVector

from rag_bench.base import BaseRAGStrategy, StrategyRetriever
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
    # HuggingFace (로컬 GPU)
    "kosimcse": "BM-K/KoSimCSE-roberta-multitask",
    "e5": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    # OpenAI API
    "openai-small": "text-embedding-3-small",
    "openai-large": "text-embedding-3-large",
    # Upstage API
    "upstage": "solar-embedding-1-query",
}

DENSE_DIMS: Dict[str, int] = {
    "BM-K/KoSimCSE-roberta-multitask": 768,
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-m3": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "solar-embedding-1-query": 4096,
}

SPARSE_TYPES: List[str] = ["korean_bm25", "splade", "fastembed_bm25"]

# ---------------------------------------------------------------------------
# 임베딩 조합 정의 (하위 호환용)
# ---------------------------------------------------------------------------

COMBO_DEFINITIONS = {
    1: {
        "name": "한국어 최적 (KoSimCSE + BM25/OKt)",
        "dense_model": "BM-K/KoSimCSE-roberta-multitask",
        "sparse_type": "korean_bm25",
        "description": "한국어 문서에 최적화. KoSimCSE(의미검색) + OKt(형태소 BM25) 조합.",
    },
    2: {
        "name": "다국어 균형 (E5 + SPLADE)",
        "dense_model": "intfloat/multilingual-e5-large",
        "sparse_type": "splade",
        "description": "70개+ 언어 지원. E5(다국어 의미검색) + SPLADE(Term Expansion) 조합.",
    },
    3: {
        "name": "올인원 통합 (BGE-M3)",
        "dense_model": "BAAI/bge-m3",
        "sparse_type": "fastembed_bm25",
        "description": "단일 모델 BGE-M3로 Dense/Sparse 통합. 중국어/영어/일본어 우수.",
    },
    4: {
        "name": "경량/빠른 속도 (MiniLM + BM25)",
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "sparse_type": "fastembed_bm25",
        "description": "최소 리소스. MiniLM(384d) + FastEmbed BM25. 가볍고 빠름.",
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
        # 새로운 독립 파라미터 방식
        strategy = DenseSparseStrategy(dense_model="kosimcse", sparse_type="splade")
        strategy = DenseSparseStrategy(dense_model="BM-K/KoSimCSE-roberta-multitask", sparse_type="fastembed_bm25")

        # 하위 호환: combo_id
        strategy = DenseSparseStrategy(combo_id=1)
    """

    def __init__(
        self,
        dense_model: Optional[str] = None,
        sparse_type: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        combo_id: Optional[int] = None,
    ):
        if combo_id is not None:
            if combo_id not in COMBO_DEFINITIONS:
                raise ValueError(f"combo_id는 1~4이어야 합니다. 입력값: {combo_id}")
            combo = COMBO_DEFINITIONS[combo_id]
            self._dense_model = combo["dense_model"]
            self._sparse_type = combo["sparse_type"]
            self._combo_id: Optional[int] = combo_id
            self._combo: Optional[dict] = combo
        elif dense_model is not None:
            self._dense_model = DENSE_MODELS.get(dense_model, dense_model)
            self._sparse_type = sparse_type or "fastembed_bm25"
            self._combo_id = None
            self._combo = None
        else:
            raise ValueError("combo_id 또는 dense_model 중 하나는 필수")

        if qdrant_path is not None:
            self._qdrant_path = qdrant_path
        elif self._combo_id is not None:
            self._qdrant_path = f"qdrant_db_combo{self._combo_id}"
        else:
            dense_short = self._dense_model.split("/")[-1]
            self._qdrant_path = f"qdrant_db_{dense_short}_{self._sparse_type}"

        self._collection_name = "document_child_chunks"

        self._dense_embeddings: Any = None
        self._sparse_embeddings: Any = None
        self._vector_store: Any = None
        self._client: Optional[QdrantClient] = None
        self._is_ready = False

    @property
    def name(self) -> str:
        if self._combo_id and self._combo:
            return f"[{self._combo_id}] {self._combo['name']}"
        dense_short = self._dense_model.split("/")[-1]
        return f"DS({dense_short}+{self._sparse_type})"

    @property
    def description(self) -> str:
        if self._combo_id and self._combo:
            return self._combo["description"]
        dense_short = self._dense_model.split("/")[-1]
        return f"Hybrid: {dense_short} (dense) + {self._sparse_type} (sparse)"

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
            # OpenAI Embeddings API
            from langchain_openai import OpenAIEmbeddings
            self._dense_embeddings = OpenAIEmbeddings(model=model_spec)
        elif "solar-embedding" in model_spec:
            # Upstage Solar Embeddings API (embed_query → query모델, embed_documents → passage모델)
            from langchain_upstage import UpstageEmbeddings
            self._dense_embeddings = UpstageEmbeddings(model="solar-embedding-1-query")
        else:
            # HuggingFace 로컬 모델
            from langchain_huggingface import HuggingFaceEmbeddings
            self._dense_embeddings = HuggingFaceEmbeddings(
                model_name=model_spec,
                model_kwargs={"device": detect_device()},
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
        """Qdrant 클라이언트 및 벡터 스토어 초기화."""
        from langchain_qdrant import QdrantVectorStore
        from langchain_qdrant.qdrant import RetrievalMode

        if self._client is None:
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
