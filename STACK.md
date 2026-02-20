# STACK.md

## 기술 스택 개요

Python 3.12 기반, uv 패키지 매니저, LangChain/LangGraph 에코시스템을 핵심으로 하는 RAG 벤치마크 시스템.

---

## 언어 및 런타임

| 항목 | 버전/세부 정보 |
|------|--------------|
| Python | 3.12+ (`.python-version` 파일에 명시) |
| 패키지 매니저 | `uv` (pyproject.toml + uv.lock) |
| 빌드 백엔드 | `hatch` (hatch.build.targets.wheel) |

---

## 핵심 프레임워크

| 라이브러리 | 버전 제약 | 역할 |
|-----------|----------|------|
| `langchain` | >=1.0 | LLM 체인, Document 타입, Retriever 추상화 |
| `langchain-core` | >=0.3 | BaseRetriever, Document, 메시지 타입 |
| `langchain-community` | >=0.3 | OpenAI 콜백 (`get_openai_callback`) |
| `langchain-huggingface` | >=0.1 | `HuggingFaceEmbeddings` |
| `langchain-openai` | >=0.2 | `ChatOpenAI`, `OpenAIEmbeddings` |
| `langchain-upstage` | >=0.3 | `UpstageEmbeddings` |
| `langchain-qdrant` | >=0.2 | `QdrantVectorStore`, `FastEmbedSparse` |
| `langchain-text-splitters` | >=0.3 | `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter` |
| `langgraph` | >=0.2 | StateGraph, 에이전트 서브그래프, Send fan-out, InMemorySaver |
| `pydantic` | (langgraph 의존) | State 타입 정의, Structured Output 스키마 |

---

## 벡터 데이터베이스

| 라이브러리 | 버전 제약 | 사용 방식 |
|-----------|----------|---------|
| `qdrant-client` | >=1.11 | 로컬 파일 모드 (`QdrantClient(path=...)`) |
| `langchain-qdrant` | >=0.2 | `QdrantVectorStore` (Dense + Sparse Hybrid, `RetrievalMode.HYBRID`) |

Qdrant는 서버 없이 로컬 디렉토리에 파일로 영속화한다. 각 (dense, sparse) 조합마다 독립된 `qdrant_db_<name>` 디렉토리가 생성된다.

---

## 임베딩 모델

### 로컬 HuggingFace 모델

| 모델 ID | 차원 | 용도 |
|---------|-----|------|
| `BM-K/KoSimCSE-roberta-multitask` | 768 | 한국어 최적화 Dense |
| `intfloat/multilingual-e5-large` | 1024 | 다국어(70개+) Dense |
| `BAAI/bge-m3` | 1024 | 올인원 Dense (중/영/일 우수) |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 경량 Dense |
| `naver/splade-cocondenser-ensembledistil` | vocab크기 | SPLADE Sparse |
| `jinaai/jina-colbert-v2` | - | ColBERT Late Interaction (89개 언어) |

### API 기반 임베딩

| 서비스 | 모델 | 차원 |
|--------|------|-----|
| OpenAI | `text-embedding-3-small` | 1536 |
| OpenAI | `text-embedding-3-large` | 3072 |
| Upstage | `solar-embedding-1-large-passage/query` | 4096 |

### Sparse 인코더

| 인코더 | 구현 | 비고 |
|--------|------|------|
| `KoreanBM25Encoder` | KoNLPy OKt + BM25 직접 구현 | 한국어 형태소 분석, TF-IDF 기반 |
| `SpladeEncoder` | Transformers + SPLADE | Term Expansion, GPU/CPU |
| `FastEmbedSparse` | `langchain_qdrant.fastembed_sparse` | Qdrant/bm25 모델, CPU ONNX |

---

## RAG 전략 구현

| 전략 클래스 | Retrieval 방식 | 외부 의존성 |
|------------|--------------|------------|
| `DenseSparseStrategy` | Hybrid (Dense + Sparse, Qdrant) | qdrant-client, HF/OpenAI/Upstage |
| `ColBERTStrategy` | Late Interaction (brute-force 또는 Voyager ANN) | pylate |
| `ColBERTRerankStrategy` | 1차 검색 + ColBERT MaxSim 리랭킹 | pylate |
| `FlashRankRerankStrategy` | 1차 검색 + ONNX 리랭킹 | flashrank |
| `ContextualRetrievalStrategy` | Decorator: LLM 문맥 부착 → base 위임 | langchain-openai |
| `OpenAIEmbedStrategy` | 순수 Dense (OpenAI API) | langchain-openai, langchain-qdrant |
| `UpstageEmbedStrategy` | 순수 Dense (Upstage API, Query/Passage 분리) | langchain-upstage, langchain-qdrant |

---

## 문서 처리

| 라이브러리 | 버전 제약 | 용도 |
|-----------|----------|------|
| `pymupdf4llm` | >=0.0.17 | PDF → Markdown 변환 |
| `konlpy` | >=0.6.0 | 한국어 형태소 분석 (OKt) — KoreanBM25Encoder |

---

## ML 런타임

| 라이브러리 | 버전 제약 | 용도 |
|-----------|----------|------|
| `torch` | >=2.2 | SPLADE, HuggingFace 모델 추론 |
| `transformers` | >=4.40 | SPLADE AutoModelForMaskedLM, AutoTokenizer |
| `sentence-transformers` | >=3.1 | HuggingFace 임베딩 백엔드 |
| `pylate` | >=1.0 | ColBERT Late Interaction (Voyager ANN 포함) |
| `fastembed` | >=0.4 | FastEmbed BM25 ONNX 인코더 |
| `flashrank` | >=0.2 | ONNX 기반 경량 크로스인코더 리랭커 |
| `einops` | >=0.8.2 | 텐서 연산 유틸리티 (ColBERT 등) |

**디바이스 정책:** `detect_device()`는 CUDA → CPU 순으로 자동 선택한다. Apple MPS는 ColBERT/SPLADE OOM 위험으로 기본 제외하고, `config.setup_ssl_bypass()`에서 `torch.backends.mps.is_available = lambda: False`로 패치한다.

---

## 평가 프레임워크

| 라이브러리 | 버전 제약 | 용도 |
|-----------|----------|------|
| `ragas` | >=0.4.3 | RAGAS v0.4+ 메트릭 (`EvaluationDataset`, `SingleTurnSample`, `evaluate`) |
| `pandas` | >=2.2 | 결과 DataFrame, CSV 출력 |
| `matplotlib` | >=3.10.8 | 시각화 |

### RAGAS 메트릭 목록

**Core (LLM 필요, reference 선택적):**
- `Faithfulness`, `AnswerRelevancy` — reference 불필요
- `ContextPrecision`, `LLMContextRecall` — reference 필요

**Extended (LLM 필요):**
- `AnswerCorrectness`, `FactualCorrectness`, `NoiseSensitivity`, `ContextEntityRecall`, `ResponseRelevancy`

**Lightweight (LLM 불필요):**
- `StringPresence`, `ExactMatch`, `NonLLMStringSimilarity`, `SemanticSimilarity`, `BleuScore`, `RougeScore`

---

## 개발 도구

| 도구 | 용도 |
|------|------|
| `jupyter` >=1.0 | 노트북 실행 환경 |
| `ipython` >=8.27 | Jupyter 커널, `IPython.display.Markdown` |
| `python-dotenv` >=1.0 | `.env` 파일 로드 |
| `ruff` | 린터/포매터 (`.ruff_cache` 존재) |
| `mypy` | 타입 체커 (`.mypy_cache` 존재) |
| `uv` | 패키지 설치 및 가상환경 관리 |

---

## 환경 변수

| 변수명 | 필수 여부 | 용도 |
|--------|---------|------|
| `OPENAI_API_KEY` | 필수 | GPT 답변 생성, 임베딩, RAGAS 평가 |
| `UPSTAGE_API_KEY` | Upstage 전략 사용 시 | Solar 임베딩 |
| `RAG_BENCH_PARALLEL` | 선택 | 쿼리 병렬 수 (기본 0=비활성) |
| `RAG_BENCH_PARALLEL_STRATEGIES` | 선택 | 전략 병렬 수 (기본 0=비활성) |
| `HF_HOME` | 자동 설정 | HuggingFace 캐시 경로 (config.py에서 오버라이드) |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | 자동 설정 | MPS OOM 방지 (0.0으로 설정) |
| `HF_HUB_DISABLE_SSL_VERIFY` | 자동 설정 | 기업망 SSL 우회 |
| `TOKENIZERS_PARALLELISM` | 자동 설정 | Tokenizer 경고 억제 |

---

## 패턴 및 설계 원칙

### Strategy Pattern
`BaseRAGStrategy` ABC가 모든 RAG 구현체에 `index()`, `retrieve()`, `get_retriever()` 계약을 강제한다. `BenchmarkRunner`는 구체적 전략을 몰라도 실행/비교가 가능하다.

### Decorator Pattern
`ContextualRetrievalStrategy`, `ColBERTRerankStrategy`, `FlashRankRerankStrategy`는 생성자에서 `base_strategy`를 주입받아 기능을 추가하고 `index()`/`retrieve()`를 위임한다.

### Lazy Initialization
모든 전략의 모델 로드 및 LLM 클라이언트 생성은 최초 `index()` 또는 `retrieve()` 호출 시 수행된다. `_ensure_initialized()` 패턴으로 통일.

### 중앙화된 상수
LLM 모델명 상수(`DEFAULT_ANSWER_LLM`, `DEFAULT_EVAL_LLM`, `DEFAULT_CONTEXTUAL_LLM`)를 `config.py`에서 일원 관리한다.

### 캐싱 전략
- `ContextualRetrievalStrategy`: SHA-256 해시 기반 `contextual_cache.json`으로 LLM 호출 중복 방지
- `IndexCacheManager`: 동일 (dense, sparse) 쌍 Qdrant 인덱스 재사용
- `ensure_model_cache()`: 글로벌 HF 캐시에서 심링크로 로컬 캐시 재사용

---

## 기술 부채 목록 (2026-02-20 기준)

> 상태 범례: **해결됨** | **부분 해결** | **잔존**

---

### 높은 우선순위

#### 1. 테스트 없음 — **잔존**

`tests/` 디렉토리가 여전히 존재하지 않는다. 전략 클래스, BenchmarkRunner, 평가 모듈, combo 패키지 어디에도 단위/통합 테스트가 없다. pyproject.toml에 pytest 의존성도 없다. `combo/` 모듈 분리 등 리팩토링이 지속되면서 리그레션 위험이 계속 높다.

#### 2. SSL 전역 몽키패치 — **잔존**

`config.setup_ssl_bypass()`에서 `requests.Session.request`를 전역 패치하여 모든 HTTP 요청의 SSL 검증을 비활성화한다. `runner.py`(L63)와 `colab_runner.py`(L824)에서도 `httpx.Client(verify=False)`를 하드코딩으로 사용한다. 프로덕션 환경 부적합이며 보안 취약점이다.

```python
# config.py:70-74
def _patched_request(self, *args, **kwargs):
    kwargs["verify"] = False           # 모든 SSL 검증 비활성화
    return _original_request(self, *args, **kwargs)
requests.Session.request = _patched_request

# runner.py:63
self._generator = ChatOpenAI(model=DEFAULT_ANSWER_LLM, http_client=httpx.Client(verify=False))
```

#### 3. MPS 런타임 패치 — **잔존 (논리적 개선 있음)**

`torch.backends.mps.is_available = lambda: False` 패치는 여전히 존재한다. 단, 이전에 함께 있던 `torch.set_default_device("cpu")` 전역 훅 문제는 제거되었고, `os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"` 환경변수가 병행 설정된다. OOM 근본 원인을 해결하지 않은 임시 조치임은 변함없다.

```python
# config.py:87
torch.backends.mps.is_available = lambda: False  # torch 내부 상태 직접 변조
```

#### 4. `OpenAIEmbedStrategy.index()`의 `force_recreate=True` — **잔존**

`openai_embed.py:57`에서 여전히 `force_recreate=True`로 항상 재인덱싱한다. `DenseSparseStrategy`가 디스크 인덱스 재사용(`index_exists` 확인 후 폴백)을 구현한 것과 대비된다. 동일한 패턴이 `UpstageEmbedStrategy`에도 적용되어 있을 가능성이 있다.

```python
# openai_embed.py:57
self._vectorstore = QdrantVectorStore.from_documents(
    ...
    force_recreate=True,   # 디스크 인덱스 재사용 불가
)
```

---

### 중간 우선순위

#### 5. `run_all_combos.py` 단일 파일 비대화 — **부분 해결**

이전 52KB에서 `ComboSpec`, `CacheConfig`, `IndexCacheManager`, `build_strategy_from_spec`이 `rag_bench/combo/` 패키지로 분리되었다. 그러나 파일은 여전히 **1,051줄**로 방대하다. 레거시 빌드 함수(`_try_build_dense_sparse`, `_try_build_colbert`, `_try_build_contextual` 등), 레이어 분석, 리포트 생성, 레거시 모드(`_run_legacy_mode`, ~200줄)가 혼재한다. 레거시 모드 분리 또는 제거가 남아 있다.

#### 6. 하드코딩된 매직 넘버 — **잔존**

아래 위치에 상수로 추출되지 않은 매직 넘버가 남아 있다:

| 위치 | 값 | 의미 |
|------|-----|------|
| `chunker.py:21,47,65,86` | `min_size=2000`, `max_size=10000`, `child_chunk_size=500` | 청킹 파라미터 기본값 |
| `graph/nodes.py:109` | `< 4` | 대화 요약 최소 메시지 수 임계값 |
| `graph/nodes.py:120` | `[-6:]` | 대화 이력 window 크기 |
| `runner.py:250` | `max_workers=8` | LLM 답변 생성 병렬 워커 수 |
| `scripts/run_all_combos.py:108,139` | `rerank_n=20` | 레거시 리랭킹 후보 수 |
| `combo/cache.py:26` | `flashrank_max_length=512`, `rerank_n=20` | CacheConfig 기본값 (상수화되어 있으나 설명 없음) |

#### 7. `_benchdata/` 인덱스 누적 — **잔존 (규모 확대)**

현재 `qdrant_db_*` 디렉토리가 **33개**로 증가했다 (이전 28개). `qdrant_db_combo1~4` 레거시 디렉토리가 여전히 남아 있으며, 오래된 인덱스 정리 정책이 없다.

```
_benchdata/
├── qdrant_db_combo1   (레거시, 제거 가능)
├── qdrant_db_combo2   (레거시, 제거 가능)
├── qdrant_db_combo3   (레거시, 제거 가능)
├── qdrant_db_combo4   (레거시, 제거 가능)
├── qdrant_db_bge-m3_fastembed_bm25
... (33개 합계)
```

#### 8. `KoreanBM25Encoder.avg_doc_len` 0 나누기 위험 — **해결됨**

`dense_sparse.py:76`에서 `avg_doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0`으로 수정되어, 빈 문서 목록을 받아도 `1.0`으로 폴백한다. 0 나누기 위험이 제거되었다.

#### 9. `colab_runner.py`와 `runner.py` 로직 중복 — **부분 해결**

`colab_runner.py`가 대폭 개편되어 `BenchmarkRunner`를 내부에서 재사용(`runner.run()`, `runner.inject_results()`)하는 구조로 변경되었다. 그러나 `_generate_answers()` 메서드(L815-851)에서 `ThreadPoolExecutor`를 사용한 LLM 답변 생성 로직이 `runner.py`의 `evaluate()` 내부 로직과 거의 동일하게 중복된다.

```python
# colab_runner.py:838-845 (runner.py:250-258과 동일 패턴)
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(_invoke, p): i for i, p in enumerate(prompts)}
    for future in as_completed(futures):
        idx = futures[future]
        answers[idx] = future.result()
```

---

### 낮은 우선순위

#### 10. `_MultiPerspectiveLLM`의 이벤트 루프 처리 — **잔존**

`evaluator.py:149-160`에서 `asyncio.get_running_loop()` 예외 처리로 동기/비동기 환경을 분기한다. Jupyter 환경에서는 항상 실행 중인 루프가 있으므로 `ThreadPoolExecutor`를 경유하는 패턴이 사용된다. `nest_asyncio` 등 명확한 해결책으로 교체 가능하다.

```python
# evaluator.py:149-160
try:
    asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, self.agenerate_text(...)).result()
except RuntimeError:
    return asyncio.run(self.agenerate_text(...))
```

#### 11. `psutil` 선택적 의존 — **부분 해결**

`pyproject.toml` 의존성에 `psutil`이 명시되지 않았지만, `uv.lock`에는 간접 의존성으로 포함되어 있다. `run_tracker.py:44-60`의 폴백 로직(macOS: `sysctl`, Linux: 누락)은 그대로다. Linux에서 `psutil`이 없으면 RAM 정보가 수집되지 않는다.

#### 12. `config.py` 경로 상수 혼재 — **잔존**

`PROJECT_ROOT`(최상위 `/autorag`) 기준 경로(`DOCS_DIR`, `MARKDOWN_DIR`, `PARENT_STORE_PATH`, `QDRANT_BASE_PATH`)와 `PACKAGE_ROOT`(`rag_bench/`) 기준 경로(`BENCH_DOCS_DIR`, `BENCH_DATA_DIR`, `MODELS_DIR`)가 동일 파일에 혼재한다. `QDRANT_BASE_PATH`는 `PROJECT_ROOT`를 가리키나 실제 Qdrant 인덱스는 `BENCH_DATA_DIR` 하위에 생성되어 주석과 실제 동작이 불일치한다.

#### 13. 타입 힌트 불완전 — **잔존**

아래 위치에 `Any` 타입이 구체화되지 않았다:

| 위치 | 변수 | 실제 타입 |
|------|------|----------|
| `base.py:103` | `strategy: Any` | `BaseRAGStrategy` |
| `dense_sparse.py:321-323` | `_dense_embeddings: Any`, `_sparse_embeddings: Any`, `_vector_store: Any` | `HuggingFaceEmbeddings \| OpenAIEmbeddings \| ...` |
| `combo/cache.py:33-36` | `cache`, `ctx_cache`, `_colbert_model`, `_flashrank_ranker` 필드 일부 | 구체적 전략/모델 타입 |

#### 14. 노트북 경로 하드코딩 — **잔존**

`rag_bench/scripts/bench_visualize.ipynb` 내부 경로가 로컬 환경에 종속되어 팀 공유 시 수정이 필요하다. `rag_bench_colab/rag_benchmark.ipynb`는 Colab 경로(`/content/RAG-Bench`)를 사용하나 `colab_config.py`의 `COLAB_PROJECT_ROOT`에 의존하여 일관성이 있다.

---

### 신규 발견 기술 부채

#### 15. `_MultiPerspectiveLLM.agenerate_text()` 뮤터블 기본 인수 — **신규**

`evaluator.py:96`에서 `callbacks: List[Any] = []`로 뮤터블 기본 인수를 사용한다. Python에서 함수 정의 시점에 단 한 번 생성되어 호출 간에 공유되는 알려진 안티패턴이다. `generate_text()`(L144)에도 동일 문제가 있다.

```python
# evaluator.py:96
async def agenerate_text(self, ..., callbacks: List[Any] = []):  # 뮤터블 기본 인수
```

#### 16. `run_all_combos.py` 레거시 모드 코드 — **신규**

`_run_legacy_mode()`(L786-997, 약 200줄)가 `combo/` 패키지 분리 이후에도 유지된다. 이 모드는 `--combos 1,2,3` 방식으로 `combo_id` 기반 빌드 함수(`_try_build_dense_sparse`, `_try_build_colbert` 등)를 직접 호출하며, 내부적으로 `_load_qa_dataset()` 함수(파일 내 별도 정의)를 사용하는 등 최신 `utils/qa_loader.py`와 중복된다. 새 프리셋 모드가 안정화되면 레거시 모드 제거를 검토해야 한다.

#### 17. `combo/cache.py`에서 `_private` 필드 dataclass 노출 — **신규**

`IndexCacheManager`가 `@dataclass`로 정의되어 있어 `_colbert_model`, `_flashrank_ranker`(프라이빗 의도)가 `dataclasses.fields()`에 노출되고, `repr=False`로 처리했으나 `asdict()` 호출 시 포함될 수 있다. `@dataclass` 대신 일반 클래스 또는 `__post_init__` 내 초기화로 교체하는 것이 명확하다.

#### 18. `colab_runner.py`의 `_tracker._record`, `_tracker._timings` 직접 접근 — **신규**

`colab_runner.py:759-763`에서 `RunTracker`의 프라이빗 속성(`_record`, `_timings`, `_phases`, `_token_total`)에 직접 접근한다. `RunTracker`에 공개 API가 없어 캡슐화가 깨진다.

```python
# colab_runner.py:759-763
rec = self._tracker._record
rec.strategy_timings = [asdict(t) for t in self._tracker._timings]
rec.phase_times = [asdict(p) for p in self._tracker._phases]
rec.token_usage_total = asdict(self._tracker._token_total)
```

---

### 해결된 항목 요약

| # | 항목 | 해결 방식 |
|---|------|---------|
| 8 | `KoreanBM25Encoder.avg_doc_len` 0 나누기 | `if doc_lens else 1.0` 폴백 추가 |

