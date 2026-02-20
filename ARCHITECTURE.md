# ARCHITECTURE.md

## 프로젝트 개요

**rag-bench** — Strategy Pattern 기반 모듈화 RAG 벤치마크 시스템

다양한 RAG(Retrieval-Augmented Generation) 전략을 동일한 쿼리 세트로 실행하고 RAGAS 메트릭으로 정량 비교하는 실험 프레임워크. LangChain/LangGraph 통합, Qdrant 벡터 DB, HuggingFace 로컬 모델 및 OpenAI/Upstage API를 지원한다.

---

## 디렉토리 레이아웃

```
autorag/
├── pyproject.toml              # 프로젝트 메타데이터 + 의존성 (uv 기반)
├── uv.lock                     # 재현 가능한 의존성 잠금 파일
├── .env                        # API 키 등 환경 변수 (비공개)
├── README.md                   # 프로젝트 전체 문서
├── MEMORY.md                   # 개발 맥락 및 결정 기록
├── docker-compose.yml          # (현재 Qdrant는 로컬 파일 모드 사용, 미활성)
├── docs/                       # 참고 문서 (PDF, 분석 MD)
│   ├── memory_oom_analysis.md
│   ├── service_modularization_proposal.md
│   ├── setup_guide.md
│   └── research/               # 연구 노트
├── scripts/                    # 환경 검증 스크립트 (최상위)
│   ├── verify_env.py
│   ├── verify_graphrag.py
│   ├── verify_rag_bench.py
│   └── verify_ragas_eval.py
├── rag_bench/                  # 핵심 패키지 (설치 가능한 Python 패키지)
│   ├── __init__.py             # 공개 API: BaseRAGStrategy, BenchmarkRunner
│   ├── base.py                 # 추상 기본 클래스 + StrategyRetriever
│   ├── config.py               # 전역 경로 상수, LLM 모델 상수, SSL 우회
│   ├── runner.py               # BenchmarkRunner (전략 실행 + 집계)
│   ├── cli.py                  # RAGChat (Jupyter/스크립트 대화 인터페이스)
│   ├── run_tracker.py          # RunTracker (수행 이력, 토큰 사용량 추적)
│   ├── combo/                  # 조합 관련 로직 (신규 분리 모듈)
│   │   ├── __init__.py         # ComboSpec, PRESETS, CacheConfig, IndexCacheManager, build_strategy_from_spec 공개
│   │   ├── spec.py             # ComboSpec 데이터클래스 + PRESETS + generate_valid_combinations
│   │   ├── cache.py            # CacheConfig + IndexCacheManager (인덱스 재사용)
│   │   └── builder.py          # build_strategy_from_spec (ComboSpec → 전략 인스턴스)
│   ├── strategies/             # 구체적 RAG 전략 구현체
│   │   ├── dense_sparse.py     # DenseSparseStrategy (하이브리드 핵심)
│   │   ├── colbert.py          # ColBERTStrategy (Late Interaction)
│   │   ├── colbert_rerank.py   # ColBERTRerankStrategy (2단계 리랭킹)
│   │   ├── flashrank_rerank.py # FlashRankRerankStrategy (경량 ONNX 리랭킹)
│   │   ├── contextual_retrieval.py  # ContextualRetrievalStrategy (LLM 문맥 부착)
│   │   ├── openai_embed.py     # OpenAIEmbedStrategy (순수 Dense)
│   │   └── upstage_embed.py    # UpstageEmbedStrategy (순수 Dense)
│   ├── indexing/               # 문서 전처리 파이프라인
│   │   ├── pdf_converter.py    # PDF → Markdown (pymupdf4llm)
│   │   └── chunker.py          # Parent-Child 청킹
│   ├── graph/                  # LangGraph 에이전트
│   │   ├── builder.py          # 에이전트 그래프 조립/컴파일
│   │   ├── nodes.py            # 그래프 노드 함수 + 도구 팩토리
│   │   ├── state.py            # State/AgentState/QueryAnalysis 타입 정의
│   │   └── prompts.py          # 시스템 프롬프트 문자열
│   ├── evaluation/             # RAGAS 평가 모듈
│   │   ├── evaluator.py        # ExtendedRAGEvaluator + EvaluationReport
│   │   └── metrics.py          # MetricPreset + MetricRegistry
│   ├── utils/
│   │   ├── device.py           # detect_device() (CUDA → CPU)
│   │   ├── qa_loader.py        # load_qa_dataset() 유틸리티
│   │   └── report.py           # print_ragas_table() 출력 유틸리티
│   ├── scripts/                # 실행 스크립트
│   │   ├── run_all_combos.py   # 전체 조합 벤치마크 (3-Layer × 72조합)
│   │   ├── run_bench.py        # 단순 실행 스크립트
│   │   ├── generate_qa.py      # QA 데이터셋 자동 생성
│   │   ├── generate_html_report.py  # HTML 리포트 생성
│   │   ├── prefetch_models.py  # HF 모델 사전 다운로드
│   │   └── bench_visualize.ipynb    # 결과 시각화 노트북
│   ├── _benchdata/             # 벤치마크 산출물 (gitignore)
│   │   ├── qdrant_db_*/        # 전략별 Qdrant 인덱스 (현재 33개)
│   │   ├── parent_store/       # Parent 청크 JSON
│   │   ├── contextual_cache.json  # LLM 문맥 요약 캐시
│   │   ├── all_combos_results.csv
│   │   ├── all_combos_ragas.csv
│   │   ├── per_sample/         # 전략별 per-sample 평가 CSV
│   │   └── run_history/        # RunTracker JSON 이력
│   └── _models/                # HF 모델 캐시 (gitignore)
└── rag_bench_colab/            # Google Colab 이식 레이어
    ├── colab_config.py         # Colab 경로 및 환경 오버라이드
    ├── colab_runner.py         # Colab 전용 벤치마크 러너 (체크포인트 지원)
    ├── colab_visualizer.py     # Colab 시각화 유틸리티
    ├── rag_benchmark.ipynb     # Colab 실행 노트북
    └── requirements_colab.txt  # Colab pip 설치 목록
```

---

## 핵심 컴포넌트 및 책임

### 1. 추상 계층 (`base.py`)

```
BaseRAGStrategy (ABC)
├── name: str           @abstractproperty
├── description: str    @abstractproperty
├── index(documents)    @abstractmethod
├── retrieve(query, k)  @abstractmethod
├── get_retriever(k)    @abstractmethod
├── is_ready: bool      (기본 False)
└── cleanup()           (기본 pass)

StrategyRetriever (BaseRetriever)
└── LangChain Retriever 래퍼 — 모든 전략에서 공유
```

**역할:** 모든 RAG 전략의 통일된 계약을 강제한다. `StrategyRetriever` 하나로 LangGraph 도구 바인딩 중복을 제거한다.

### 2. 전략 계층 (`strategies/`)

| 클래스 | 검색 방식 | 핵심 의존성 |
|--------|----------|------------|
| `DenseSparseStrategy` | Dense + Sparse Hybrid | Qdrant, HF/OpenAI/Upstage 임베딩 |
| `ColBERTStrategy` | Late Interaction (전체 코퍼스) | PyLate, jina-colbert-v2 |
| `ColBERTRerankStrategy` | 1차 검색 + ColBERT 리랭킹 | PyLate (후보 N개만 인코딩) |
| `FlashRankRerankStrategy` | 1차 검색 + ONNX 리랭킹 | FlashRank (Torch 불필요) |
| `ContextualRetrievalStrategy` | Decorator: LLM 문맥 부착 + base 위임 | OpenAI GPT, SHA-256 캐시 |
| `OpenAIEmbedStrategy` | 순수 Dense (OpenAI) | OpenAI Embeddings API, Qdrant |
| `UpstageEmbedStrategy` | 순수 Dense (Upstage) | Upstage Solar API, Qdrant |

**Decorator 패턴 적용:** `ContextualRetrievalStrategy`, `ColBERTRerankStrategy`, `FlashRankRerankStrategy`는 임의의 `base_strategy`를 감싸 기능을 추가한다.

### 3. 조합 관리 계층 (`combo/`)

`run_all_combos.py`에서 분리된 독립 모듈이다.

```
combo/
├── spec.py    — ComboSpec (dataclass), PRESETS dict, generate_valid_combinations()
├── cache.py   — CacheConfig, IndexCacheManager (동일 인덱스 키 재사용)
└── builder.py — build_strategy_from_spec() (ComboSpec → 전략 인스턴스)
```

`IndexCacheManager`가 동일 `(dense, sparse)` 쌍의 Qdrant 인덱스를 `cache` dict에서 재사용하고, `ctx_cache`에서 contextual 전략을 재사용한다. ColBERT 모델과 FlashRank 리랭커는 싱글톤으로 공유된다.

### 4. 문서 처리 파이프라인 (`indexing/`)

```
PDF 원본
  ↓ pdf_converter.py (pymupdf4llm)
Markdown 파일
  ↓ chunker.py (MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter)
Parent 청크 (2,000~10,000자) → parents.json 저장
  ↓ 동일 청크에서 Child 청크 파생
Child 청크 (500자, 100자 overlap) → 전략별 인덱싱
```

**Parent-Child 구조:** Child는 검색에, Parent는 LangGraph 에이전트가 전체 맥락 조회 시 사용한다.

### 5. 실행 엔진 (`runner.py`)

```
BenchmarkRunner
├── run()                  — 순차/병렬(전략 또는 쿼리 단위) 실행
├── _run_single()          — 단일 전략 × 쿼리 실행, ms 레이턴시 측정
├── evaluate()             — RAGAS 평가 (LLM 답변 생성 + ExtendedRAGEvaluator)
├── compare()              — 전략 간 요약 출력
├── to_dataframe()         — pandas DataFrame 변환
└── inject_results()       — 외부 결과 주입 (재검색 방지)
```

병렬 실행은 `ThreadPoolExecutor`로 처리하며 `RAG_BENCH_PARALLEL`, `RAG_BENCH_PARALLEL_STRATEGIES` 환경 변수로 제어한다.

### 6. LangGraph 에이전트 (`graph/`)

```
메인 그래프 (State)
  START
    ↓ summarize          — 대화 이력 요약 (6메시지 window)
    ↓ analyze_rewrite    — 쿼리 분석/재작성 (Structured Output: QueryAnalysis)
    ↓ [Send 병렬 fan-out] — 재작성된 질문별 process_question 서브그래프
    ↓ aggregate          — 여러 답변 통합
  END

서브그래프 (AgentState)
  START
    ↓ agent              — ReAct 에이전트 (tools: search_child_chunks, retrieve_parent_chunks)
    ↓ tools              — ToolNode
    ↓ extract_answer
  END
```

`interrupt_before=["human_input"]`로 사용자 입력 대기 지점이 명시적으로 정의되어 있다.

### 7. 평가 모듈 (`evaluation/`)

```
ExtendedRAGEvaluator
├── MetricPreset: CORE_ONLY | FULL | REFERENCE_FREE | COMPREHENSIVE
├── LLM: LangchainLLMWrapper + _MultiPerspectiveLLM (n>1 역질문 구조화 생성)
├── EvaluationReport: per-sample DataFrame + aggregate_dict
└── rank_strategies(): SCORING_PROFILES(balanced/precision_critical/speed_critical/comprehensive) 가중 점수

SCORING_PROFILES 키: balanced, precision_critical, speed_critical, comprehensive
RAGAS 메트릭: faithfulness, answer_relevancy, context_precision, llm_context_recall,
             answer_correctness, factual_correctness, noise_sensitivity,
             context_entity_recall, response_relevancy + Lightweight 6종
```

### 8. 수행 이력 추적 (`run_tracker.py`)

`RunTracker`는 각 벤치마크 실행을 `run_<timestamp>.json`으로 기록하고, 플랫폼 정보(CPU/RAM/GPU/Apple Silicon), 단계별 소요 시간, 전략별 레이턴시 백분위수(p50/p95), LLM API 토큰 사용량 및 비용을 집계한다.

### 9. 전체 조합 실행 (`scripts/run_all_combos.py`)

3-Layer 카테시안 곱으로 최대 72개 조합을 생성한다:

- **Layer 1:** Dense Model (kosimcse, e5, bge-m3, minilm, openai-small/large, upstage)
- **Layer 2:** Sparse Type (korean_bm25, splade, fastembed_bm25)
- **Layer 3:** Retrieval Mode (hybrid × reranker[None/colbert/flashrank] × llm_support[None/contextual])

`combo/` 패키지의 `IndexCacheManager`가 동일 (dense, sparse) 쌍의 Qdrant 인덱스를 재사용하여 중복 인덱싱을 방지한다. 레거시 모드(`--combos`/`--skip_*`)와 새 프리셋 모드(`--preset quick|standard|full`)를 모두 지원한다.

---

## 데이터 흐름

```
[PDF 원본]
    |
    v (pymupdf4llm)
[Markdown]
    |
    v (MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter)
[Parent-Child 청크]
    |
    +---> [parents.json]  (LangGraph 에이전트 Parent 조회용)
    |
    v (ContextualRetrievalStrategy: LLM 문맥 요약 부착 + 캐싱)
[Enriched Child 청크]
    |
    v (DenseSparseStrategy / ColBERTStrategy)
[Qdrant 벡터 인덱스]
    |
    v (BenchmarkRunner.run())
[검색 결과 dict]
    |
    +---> [compare(): 레이턴시/결과수 요약]
    |
    v (BenchmarkRunner.evaluate())
[LLM 답변 생성]  →  [RAGAS 평가]  →  [EvaluationReport]
    |
    v (rank_strategies())
[순위 DataFrame]  →  [CSV/HTML 리포트]  →  [run_history JSON]
```

---

## 모듈 의존 관계

```
rag_bench/
  __init__       → config, base, runner
  runner         → base, config, evaluation.evaluator
  cli            → (외부에서 graph, strategies 주입)
  run_tracker    → (독립 — 표준 라이브러리만 사용)

  combo/         → config, strategies.dense_sparse (DENSE_MODELS, SPARSE_TYPES)
    spec         → strategies.dense_sparse
    cache        → config, combo.spec
    builder      → combo.spec, combo.cache, strategies.*

  strategies/    → base, config, utils.device
    dense_sparse → base, utils.device, qdrant_client, langchain_*
    colbert      → base, utils.device, pylate
    colbert_rerank → base, utils.device, pylate
    flashrank_rerank → base, flashrank
    contextual_retrieval → base, config, langchain_openai

  indexing/      → (독립 — langchain_text_splitters, pymupdf4llm)
  graph/         → config, base
    builder      → config, base, graph.nodes, graph.state
    nodes        → graph.state, graph.prompts
  evaluation/    → config
    evaluator    → config, evaluation.metrics, ragas
    metrics      → ragas
  utils/         → (독립 — torch 선택적, pandas)

  scripts/       → config, combo, indexing.chunker, run_tracker, runner,
                   strategies.*, utils.qa_loader, utils.report
```

---

## 외부 연동점

| 서비스 | 용도 | 인증 |
|--------|------|------|
| OpenAI API | 답변 생성, 임베딩, RAGAS 평가 LLM | `OPENAI_API_KEY` |
| Upstage Solar API | solar-embedding Dense 검색 | `UPSTAGE_API_KEY` |
| HuggingFace Hub | 로컬 모델 다운로드 (KoSimCSE, E5, BGE-M3, MiniLM, SPLADE, ColBERT) | HF_HOME 환경변수 |
| Qdrant | 벡터 DB (로컬 파일 모드) | 없음 (localhost) |
| Google Colab | Colab 이식 레이어 (`rag_bench_colab/`) | Google Drive 마운트 |

---

## 진입점 요약

| 진입점 | 경로 | 역할 |
|--------|------|------|
| 전체 조합 벤치마크 | `python -m rag_bench.scripts.run_all_combos` | 72조합 자동 실행 |
| 단순 벤치마크 | `python -m rag_bench.scripts.run_bench` | 지정 전략 실행 |
| QA 생성 | `python -m rag_bench.scripts.generate_qa` | LLM으로 QA 데이터셋 생성 |
| HTML 리포트 | `python -m rag_bench.scripts.generate_html_report` | 시각화 리포트 |
| 모델 사전 다운로드 | `python -m rag_bench.scripts.prefetch_models` | HF 모델 캐시 |
| 환경 검증 | `python scripts/verify_env.py` | 설치 확인 |
| Colab 노트북 | `rag_bench_colab/rag_benchmark.ipynb` | Colab 실행 |

