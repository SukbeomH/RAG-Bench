# rag_bench — 모듈화 RAG 벤치마크 시스템

Strategy Pattern 기반으로 다양한 RAG 방식을 통일된 인터페이스로 비교 벤치마크하는 LangChain/LangGraph 통합 패키지.

> 이 디렉토리 단독으로 공유 가능합니다. `pyproject.toml`, `uv.lock`이 포함되어 있습니다.

## 설계 의도

### 문제 인식

한국어 RAG 시스템에서 "어떤 임베딩을 쓸까?", "BM25와 SPLADE 중 뭐가 나을까?", "리랭커를 붙이면 얼마나 좋아질까?"와 같은 질문에 답하려면, 동일 조건에서 공정하게 비교할 수 있는 프레임워크가 필요합니다.

### 핵심 아이디어

```
┌─────────────────────────────────────────────────────┐
│           모든 전략이 동일한 인터페이스를 구현         │
│                                                     │
│  BaseRAGStrategy ABC                                │
│    ├── index(documents)     # 문서 인덱싱           │
│    ├── retrieve(query, k)   # 검색                  │
│    ├── get_retriever(k)     # LangChain 호환        │
│    └── cleanup()            # 리소스 정리            │
│                                                     │
│  → BenchmarkRunner가 모든 전략을 동일하게 실행       │
│  → RAGEvaluator가 RAGAS 메트릭으로 통일 평가        │
└─────────────────────────────────────────────────────┘
```

### 왜 3-Layer 교차 조합인가

기존 벤치마크는 "Dense+Sparse 조합 4가지 + ColBERT + Reranker" 같은 개별 전략 비교였습니다. 하지만 실제 프로덕션에서는 **Dense 모델, Sparse 모델, 후처리(리랭킹/문맥 부착)** 세 축을 독립적으로 선택합니다.

3-Layer 설계로 바꾸면:
- 각 레이어의 **독립적 기여도**를 분석할 수 있음 (예: "BGE-M3가 다른 Dense 모델보다 평균 15% 우수")
- **최적 조합**을 찾을 수 있음 (예: "e5+splade+flashrank가 의외로 1위")
- **비용 대비 효과**를 판단할 수 있음 (예: "contextual은 2배 느리지만 품질 차이는 5%")

### 왜 2-Pass 실행인가

72개 조합 전부에 RAGAS를 돌리면 OpenAI API 비용이 과도합니다.

```
Pass 1: 72개 전략 × 20 쿼리 = 1,440회 검색  → 레이턴시만 (API 비용 $0)
Pass 2: 상위 10개 전략 × 20 쿼리 = 200회 평가 → RAGAS (API 비용 ~$2-5)
```

API 호출을 86% 절감하면서도 의미 있는 상위권 전략의 품질을 평가할 수 있습니다.

## 아키텍처

### 전체 시스템 구조

```
┌────────────────────────────────────────────────────────────────┐
│                       BenchmarkRunner                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐     │
│  │  Dense   │ │  ColBERT │ │  Graph   │ │ColBERT Rerank │     │
│  │ +Sparse  │ │ (PyLate) │ │(LightRAG)│ │ (2-stage)     │     │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬────────┘     │
│       │       ┌────────────┐ ┌──────────────┐   │              │
│       │       │ FlashRank  │ │ Contextual   │   │              │
│       │       │  Rerank    │ │  Retrieval   │   │              │
│       │       └─────┬──────┘ └──────┬───────┘   │              │
│       └─────────────┴──────┬────────┴───────────┘              │
│                  ┌─────────▼─────────┐                         │
│                  │  BaseRAGStrategy  │  ← ABC                  │
│                  │  index()          │                          │
│                  │  retrieve()       │                          │
│                  │  get_retriever()  │                          │
│                  └───────────────────┘                          │
│                            │                                   │
│                  ┌─────────▼─────────┐                         │
│                  │   RAGEvaluator    │  ← RAGAS v0.4+          │
│                  └───────────────────┘                          │
└────────────────────────────────────────────────────────────────┘
```

### 문서 처리 흐름

```
┌───────────┐    ┌─────────────┐    ┌────────────────────────────────┐
│  PDF 문서  │    │  Markdown   │    │     Parent-Child 청킹          │
│  (원본)    │───→│   변환      │───→│                                │
│            │    │ pymupdf4llm │    │  Parent (2000-10000자)         │
└───────────┘    └─────────────┘    │  ┌──────────────────────────┐  │
                                    │  │ "AI DC의 전력 소비는...   │  │
                                    │  │ GPU 수요 증가에 따라...   │  │
                                    │  │ 2025년까지 총 전력..."    │  │
                                    │  └──────────┬───────────────┘  │
                                    │             │ 분할              │
                                    │  Child (300-500자)             │
                                    │  ┌──────┐ ┌──────┐ ┌──────┐   │
                                    │  │청크1 │ │청크2 │ │청크3 │   │
                                    │  │검색  │ │검색  │ │검색  │   │
                                    │  │단위  │ │단위  │ │단위  │   │
                                    │  └──────┘ └──────┘ └──────┘   │
                                    └────────────────────────────────┘
```

### Decorator Pattern 기반 전략 합성

```
┌─────────────────────────────────────────────────────────────────┐
│                    전략 합성 패턴 (Decorator)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  기본 전략 (Base)          리랭킹 데코레이터        LLM 데코레이터│
│  ┌─────────────────┐     ┌──────────────────┐                   │
│  │ DenseSparse     │     │ ColBERTRerank    │                   │
│  │ Strategy        │◄────│ Strategy         │                   │
│  │                 │     │                  │                   │
│  │ search(q, k=20) │     │ search(q, k=3)  │                   │
│  └─────────────────┘     │  ├ base(q, k=20) │                   │
│                          │  └ rerank → top-3 │                   │
│                          └──────────────────┘                   │
│                                                                 │
│  ┌─────────────────┐     ┌──────────────────┐                   │
│  │ DenseSparse     │     │ FlashRankRerank  │                   │
│  │ Strategy        │◄────│ Strategy         │                   │
│  │                 │     │ (ONNX, CPU 전용) │                   │
│  └─────────────────┘     └──────────────────┘                   │
│                                                                 │
│  ┌─────────────────┐     ┌──────────────────┐                   │
│  │ DenseSparse     │     │ Contextual       │                   │
│  │ Strategy        │◄────│ Retrieval        │                   │
│  │ (ctx 인덱스)    │     │ (인덱싱 시 LLM)  │                   │
│  └─────────────────┘     └──────────────────┘                   │
│                                                                 │
│  ◄──── : "base_strategy로 위임" (Decorator Pattern)             │
└─────────────────────────────────────────────────────────────────┘
```

### Dense+Sparse Hybrid 검색 상세

```
                        쿼리: "AI DC 전력 소비"
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
          ┌─────────────────┐ ┌─────────────────┐
          │  Dense 임베딩    │ │  Sparse 임베딩   │
          │  (벡터 유사도)   │ │  (키워드 매칭)   │
          │                 │ │                 │
          │  KoSimCSE 768d  │ │  BM25+OKt       │
          │  or E5 1024d    │ │  or SPLADE      │
          │  or BGE-M3      │ │  or FastEmbed   │
          │  or MiniLM 384d │ │                 │
          └────────┬────────┘ └────────┬────────┘
                   │                   │
                   ▼                   ▼
          ┌─────────────────────────────────────┐
          │      Reciprocal Rank Fusion (RRF)   │
          │                                     │
          │  Dense Top-K  ∪  Sparse Top-K       │
          │  → RRF 점수 = Σ 1/(k + rank_i)     │
          │  → 통합 순위 생성                     │
          └──────────────────┬──────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  검색 결과 반환  │
                    │  (top-k 문서)    │
                    └─────────────────┘
```

### LangGraph Agentic RAG 흐름

```
┌───────────────────────────────────────────────────────────────────┐
│                      LangGraph Agent 흐름                         │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   사용자 질문                                                      │
│       │                                                           │
│       ▼                                                           │
│   ┌──────────────┐                                                │
│   │  쿼리 분석    │  질문 명확성 판단 + 쿼리 재작성                  │
│   └──────┬───────┘                                                │
│          │                                                        │
│          ▼                                                        │
│   ┌──────────────┐    ┌────────────────────┐                      │
│   │  RAG Agent   │───→│  도구 호출          │                      │
│   │  (GPT-4o)    │    │                    │                      │
│   └──────────────┘    │  search_child      │                      │
│          ▲            │  _chunks(query, k) │                      │
│          │            │         │          │                      │
│          │            │         ▼          │                      │
│   ┌──────┴───────┐    │  retrieve_parent   │                      │
│   │  응답 집계    │    │  _chunks(id)       │                      │
│   │  + 정제       │    └────────────────────┘                      │
│   └──────────────┘                                                │
│          │                                                        │
│          ▼                                                        │
│   최종 답변 반환                                                    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## 패키지 구조

```
rag_bench/
├── __init__.py              # 패키지 진입점 (BaseRAGStrategy, BenchmarkRunner export)
├── config.py                # 전역 설정 (경로, LLM 모델, SSL 우회)
├── base.py                  # BaseRAGStrategy ABC — Strategy Pattern 핵심
├── runner.py                # BenchmarkRunner — 다중 전략 실행기 + 레이턴시 측정
├── cli.py                   # RAGChat — Agentic RAG 대화 인터페이스
│
├── strategies/              # RAG 전략 구현체 (6종)
│   ├── __init__.py          # 전략 클래스 일괄 export
│   ├── dense_sparse.py      # ★ Dense+Sparse Hybrid (4종 임베딩 + 3종 희소 검색)
│   ├── colbert.py           # ColBERT Late Interaction (PyLate, MaxSim)
│   ├── colbert_rerank.py    # ColBERT 2-stage Reranking (Decorator Pattern)
│   ├── flashrank_rerank.py  # FlashRank 경량 ONNX Reranking (Decorator)
│   ├── contextual_retrieval.py  # Contextual Retrieval (인덱싱 시 LLM 문맥 부착)
│   └── graph_rag.py         # GraphRAG (LightRAG — 엔터티/관계 그래프 검색)
│
├── indexing/                # 문서 처리 파이프라인
│   ├── __init__.py          # pdfs_to_markdowns, create_parent_child_chunks export
│   ├── pdf_converter.py     # PDF → Markdown 변환 (pymupdf4llm)
│   └── chunker.py           # Parent-Child 청킹 (Parent: 2000-10000자, Child: 300-500자)
│
├── evaluation/              # RAGAS 평가 모듈
│   ├── __init__.py          # RAGEvaluator export
│   ├── evaluator.py         # ExtendedRAGEvaluator — 샘플별 평가 + 비용 추적
│   ├── metrics.py           # MetricRegistry (Core 4종, Extended 3종, Lightweight 2종)
│   └── legacy.py            # 레거시 평가 코드
│
├── graph/                   # LangGraph Agentic RAG
│   ├── __init__.py          # build_agent_graph export
│   ├── state.py             # State, AgentState, QueryAnalysis 정의
│   ├── prompts.py           # 시스템 프롬프트 4종 (요약, 쿼리분석, RAG, 집계)
│   ├── nodes.py             # 그래프 노드 + 도구 바인딩 + 라우팅
│   └── builder.py           # build_agent_graph() — 컴파일된 LangGraph 생성
│
├── scripts/                 # 벤치마크 실행 스크립트
│   ├── __init__.py
│   ├── generate_qa.py       # QA 데이터셋 자동 생성 (GPT-4o-mini)
│   ├── run_bench.py         # 3종 전략 벤치마크 + RAGAS 평가
│   └── run_all_combos.py    # ★ 72개 3-Layer 교차 조합 벤치마크 (2-Pass)
│
├── docs/                    # 벤치마크 대상 Markdown 문서
│   ├── 20250910_AI 현황 보고서.md    # AI 현황 보고서 (502KB)
│   └── SPRi AI Brief...md           # AI 산업 동향 (87KB)
│
├── _benchdata/              # 벤치마크 산출물 (.gitignore)
│   ├── qa_dataset.json      # QA 데이터셋
│   ├── parent_store/        # Parent 청크 JSON 저장소
│   ├── all_combos_latency.csv   # 레이턴시 결과
│   ├── all_combos_ragas.csv     # RAGAS 평가 결과
│   └── e2e_report.md            # 종합 리포트
│
├── pyproject.toml           # 의존성 정의 (독립 실행 가능)
├── uv.lock                  # 의존성 잠금
└── .python-version          # Python 3.12
```

## 모듈 상세 설명

### 1. BaseRAGStrategy (base.py)

모든 RAG 전략의 추상 기본 클래스. Strategy Pattern의 핵심.

```python
class BaseRAGStrategy(ABC):
    @property
    def name(self) -> str: ...          # 전략 이름 (벤치마크 표시용)
    @property
    def description(self) -> str: ...   # 전략 설명
    def index(self, documents) -> None: ... # 문서 인덱싱
    def retrieve(self, query, k) -> List[Document]: ... # 검색
    def get_retriever(self, k) -> BaseRetriever: ...     # LangChain 호환
    def cleanup(self) -> None: ...      # 리소스 정리 (선택)
```

### 2. DenseSparseStrategy (strategies/dense_sparse.py)

Qdrant 벡터 DB 기반 Dense+Sparse Hybrid 검색. 프로젝트의 핵심 전략.

**Dense 모델 4종:**

| 키 | 모델 | 차원 | 한국어 | 속도 |
|----|------|------|:------:|:----:|
| `kosimcse` | KoSimCSE-roberta-multitask | 768 | ★★★ | ★★ |
| `e5` | multilingual-e5-large | 1024 | ★★ | ★ |
| `bge-m3` | BGE-M3 | 1024 | ★★ | ★ |
| `minilm` | all-MiniLM-L6-v2 | 384 | ★ | ★★★ |

**Sparse 모델 3종:**

| 키 | 방식 | 특징 |
|----|------|------|
| `korean_bm25` | BM25 + KoNLPy OKt | 한국어 형태소 분석 기반, 최고 한국어 성능 |
| `splade` | SPLADE (learned sparse) | 학습된 희소 표현, 다국어 지원 |
| `fastembed_bm25` | FastEmbed BM25 | Qdrant 네이티브, 설치 간편 |

**동작 흐름:**
1. Dense 임베딩 + Sparse 임베딩 동시 생성
2. Qdrant에 `named_vectors` + `named_sparse_vectors`로 저장
3. 검색 시 Dense 유사도 + Sparse BM25 점수를 RRF(Reciprocal Rank Fusion)로 병합

### 3. ColBERT / ColBERTRerank (strategies/colbert*.py)

**ColBERTStrategy**: PyLate 기반 Late Interaction. 토큰 수준 MaxSim 점수. 소규모 코퍼스에서 brute-force, 대규모에서 Voyager ANN 인덱스.

**ColBERTRerankStrategy**: Decorator Pattern. 임의의 base 전략 위에 ColBERT 리랭킹을 적용. base에서 top-N 후보를 검색한 뒤, ColBERT MaxSim으로 재정렬.

```
[검색 흐름]  base.retrieve(k=20) → ColBERT MaxSim 재점수화 → top-k 반환
```

### 4. FlashRankRerankStrategy (strategies/flashrank_rerank.py)

ONNX 기반 경량 리랭커. CPU 전용, ms 수준 레이턴시. ColBERT Rerank 대비 10-50배 빠름.

모델: `ms-marco-MultiBERT-L-12` (~150MB, 100+ 언어 지원)

```
[검색 흐름]  base.retrieve(k=20) → FlashRank ONNX 점수화 → top-k 반환
```

### 5. ContextualRetrievalStrategy (strategies/contextual_retrieval.py)

Anthropic이 제안한 Contextual Retrieval 기법. **인덱싱 시** LLM으로 각 청크에 상위 문서 문맥을 부착.

```
[인덱싱 시]  각 child 청크 → LLM이 parent 문맥 참조 → "이 청크는 AI 전력 문제를 다루는
             보고서의 일부로, DC의 전력 소비 추세를 설명합니다." 접두사 생성 → 인덱싱
```

보고된 효과: 검색 실패율 49-67% 감소 (Anthropic 논문 기준)

### 6. GraphRAGStrategy (strategies/graph_rag.py)

LightRAG 백엔드. LLM으로 엔터티/관계를 추출하여 지식 그래프 구축 후 그래프 탐색 검색.

3가지 모드: `local` (엔터티 중심), `global` (커뮤니티 요약), `hybrid` (둘 다)

### 7. 문서 처리 파이프라인 (indexing/)

```
PDF 문서 → pdf_converter.py → Markdown → chunker.py → Parent-Child 청크
                                          ├── Parent: 2000-10000자 (전체 문맥)
                                          └── Child: 300-500자 (검색 단위)
```

Parent-Child 전략: 검색은 작은 Child 청크로 정밀하게, 답변 생성은 큰 Parent 청크로 풍부하게.

### 8. RAGAS 평가 (evaluation/)

RAGAS v0.4+ 기반 4개 핵심 메트릭:

| 메트릭 | 분류 | 평가 대상 |
|--------|------|----------|
| Context Precision | Retrieval | 검색 결과의 정밀도 |
| Context Recall | Retrieval | 검색 결과의 재현율 |
| Faithfulness | Generation | 답변의 사실 충실도 |
| Answer Relevancy | Generation | 답변의 질문 적합도 |

`ExtendedRAGEvaluator`는 샘플별 평가 + API 토큰 사용량/비용 추적을 지원합니다.

### 9. LangGraph 에이전트 (graph/)

검색 전략을 LangGraph 기반 대화형 에이전트로 감싸는 모듈.

```
사용자 질문 → 쿼리 분석 → 도구 선택 → 검색 (search_child_chunks)
                                     → 부모 조회 (retrieve_parent_chunks)
                                     → 답변 생성 → 응답 집계
```

## 빠른 시작

### 0. 환경 설정

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
```

### 1. 72개 조합 벤치마크 (권장)

```bash
# Step 1: QA 데이터셋 생성
uv run python -m rag_bench.scripts.generate_qa --num_qa 20

# Step 2: dry-run으로 조합 확인
uv run python -m rag_bench.scripts.run_all_combos --preset full --dry-run --layers

# Step 3: 실행 (Pass 1: 레이턴시 → Pass 2: 상위 10개 RAGAS)
uv run python -m rag_bench.scripts.run_all_combos \
    --preset full \
    --top_n 10 \
    --k 3 \
    --layers
```

### 2. Python API로 전략 비교

```python
from rag_bench import BenchmarkRunner
from rag_bench.strategies import DenseSparseStrategy

strategies = [
    DenseSparseStrategy(combo_id=1),  # 한국어 최적 (KoSimCSE + BM25/OKt)
    DenseSparseStrategy(combo_id=4),  # 경량 (MiniLM + BM25)
]

queries = ["AI 산업 동향은?", "생성형 AI의 주요 활용 분야는?"]
runner = BenchmarkRunner(strategies, queries)
results = runner.run()
runner.compare()

df = runner.to_dataframe()
```

### 3. 문서 인덱싱 파이프라인

```python
from rag_bench.indexing import pdfs_to_markdowns, create_parent_child_chunks
from rag_bench.strategies import DenseSparseStrategy

# PDF → Markdown → Parent-Child 청킹
pdfs_to_markdowns(docs_dir="docs", output_dir="markdown")
parents, children = create_parent_child_chunks(
    markdown_dir="markdown",
    parent_store_path="parent_store",
)

strategy = DenseSparseStrategy(combo_id=1)
strategy.index(children)
```

### 4. Agentic RAG 대화

```python
from rag_bench.strategies import DenseSparseStrategy
from rag_bench.graph import build_agent_graph
from rag_bench.cli import RAGChat

strategy = DenseSparseStrategy(combo_id=1)
graph = build_agent_graph(strategy)
chat = RAGChat(graph, strategy)

chat.ask("제네시스 미션이 뭐야?")
chat.clear()  # 세션 초기화
```

## 3-Layer 조합 벤치마크 상세

### 프리셋

| 프리셋 | Dense | Sparse | Reranker | LLM Support | 조합 수 |
|--------|:-----:|:------:|:--------:|:-----------:|:-------:|
| `quick` | 2 (bge-m3, minilm) | 1 (fastembed_bm25) | 2 (none, flashrank) | 1 (none) | **4** |
| `standard` | 4 | 3 | 2 (none, flashrank) | 1 (none) | **24** |
| `full` | 4 | 3 | 3 (none, colbert, flashrank) | 2 (none, contextual) | **72** |

### 인덱스 캐싱 메커니즘

72개 조합이지만, 실제 인덱싱이 필요한 건 (Dense, Sparse) 쌍의 수 = 4×3 = 12개뿐.

```
kosimcse+korean_bm25   ← 인덱스 1회 생성
  ├── hybrid              (인덱스 재사용)
  ├── +colbert_rerank     (인덱스 재사용 + ColBERT 리랭킹)
  ├── +flashrank_rerank   (인덱스 재사용 + FlashRank 리랭킹)
  ├── +contextual         (별도 ctx 인덱스 생성)
  ├── +colbert+contextual (ctx 인덱스 + ColBERT)
  └── +flashrank+contextual (ctx 인덱스 + FlashRank)
```

### 2-Pass 실행 흐름

```
Step 1: QA 데이터셋 로드 (20쌍)
Step 2: 문서 청킹 (Parent-Child)
Step 3: 72개 전략 생성 + 인덱싱 (12개 고유 인덱스)
Step 4: Pass 1 — 72개 전략 × 20 쿼리 = 1,440회 검색 → 레이턴시
Step 5: Pass 2 — 상위 10개 전략 × 20 쿼리 = 200회 검색 → RAGAS 평가
Step 6: 리포트 생성 (e2e_report.md)
```

### 레이어별 기여도 분석 (`--layers`)

`--layers` 플래그를 사용하면 각 레이어의 독립적 기여도를 분석합니다:

```
Dense Model:
  kosimcse    → 0.234s (n=18)
  e5          → 0.456s (n=18)
  bge-m3      → 0.345s (n=18)
  minilm      → 0.123s (n=18)

Sparse Model:
  korean_bm25    → 0.312s (n=24)
  splade         → 0.289s (n=24)
  fastembed_bm25 → 0.198s (n=24)

Reranker:
  none       → 0.156s (n=24)
  colbert    → 0.523s (n=24)
  flashrank  → 0.178s (n=24)
```

## 벤치마크 CLI 옵션 전체

### run_all_combos.py

```
공통:
  --k K                검색 결과 수 (기본: 3)
  --no_ragas           RAGAS 평가 건너뛰기
  --reindex            기존 인덱스 삭제 후 재인덱싱

새 모드 (--preset 사용 시):
  --preset PRESET      quick | standard | full
  --pass1-only         레이턴시만 (RAGAS 없음)
  --top_n N            상위 N개만 RAGAS 평가
  --dry-run            조합 목록만 출력
  --layers             레이어별 기여도 분석

레거시 모드 (--preset 미사용 시):
  --combos 1,3,4       DenseSparse 조합 ID 지정
  --skip_colbert       ColBERT 전략 건너뛰기
  --skip_rerank        ColBERTRerank 건너뛰기
  --skip_graphrag      GraphRAG 건너뛰기
  --skip_contextual    Contextual Retrieval 건너뛰기
  --skip_flashrank     FlashRank Rerank 건너뛰기
  --contextual_base N  Contextual 기반 조합 ID (기본: 3)
```

### generate_qa.py

```
  --num_qa N           생성할 QA 쌍 수 (기본: 10)
  --sample_ratio F     청크 샘플링 비율 (기본: 0.3)
```

### run_bench.py

```
  --k K                검색 결과 수 (기본: 3)
```

## 새 전략 추가하기

`BaseRAGStrategy`를 상속하여 구현하면 벤치마크에 편입됩니다:

```python
from rag_bench.base import BaseRAGStrategy

class MyStrategy(BaseRAGStrategy):
    @property
    def name(self) -> str:
        return "My Strategy"

    @property
    def description(self) -> str:
        return "커스텀 RAG 전략"

    def index(self, documents):
        # 문서를 벡터 DB에 인덱싱
        ...

    def retrieve(self, query, k=5):
        # 쿼리로 검색하여 Document 리스트 반환
        ...

    def get_retriever(self, k=5):
        # LangChain BaseRetriever 호환 객체 반환
        ...
```

3-Layer 조합에 새 레이어 값을 추가하려면 `run_all_combos.py`의 `PRESETS` 딕셔너리를 수정합니다.

## 의존성

전체 의존성은 `pyproject.toml`을 참고하세요. 주요 항목:

**핵심:**
- `langchain-core`, `langchain-text-splitters`, `langchain-qdrant`
- `qdrant-client`, `pandas`, `python-dotenv`

**전략별:**
- Dense+Sparse: `konlpy` (OKt), `transformers` (SPLADE), `fastembed`, `sentence-transformers`
- ColBERT: `pylate`, `sentence-transformers`
- FlashRank: `flashrank` (ONNX 기반)
- Contextual: `langchain-openai` (GPT-4o-mini)
- GraphRAG: `lightrag-hku`, `nest-asyncio`

**평가:**
- `ragas>=0.4.3`, `datasets`

**Agentic RAG:**
- `langgraph`, `langchain-openai`
