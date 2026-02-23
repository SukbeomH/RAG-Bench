# rag_bench — 모듈화 RAG 벤치마크 시스템

Strategy Pattern 기반으로 다양한 RAG 방식을 통일된 인터페이스로 비교 벤치마크하는 LangChain/LangGraph 통합 패키지.

> **최근 변경**: HTML 보고서 고도화 — Executive Summary 추천 카드, 레이어별 기여도 분석 표, Contextual ON/OFF 탭 분리 RAGAS 테이블, IQR 이상치 감지 + 중앙값 부표시, 최고 vs 최저 레이더 비교, 산점도 ColBERT/non-ColBERT 색상 분리 + 3s 기준선 추가

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

### 왜 4-Layer 교차 조합인가

기존 벤치마크는 "Dense+Sparse 조합 4가지 + ColBERT + Reranker" 같은 개별 전략 비교였습니다. 하지만 실제 프로덕션에서는 **Dense 모델, Sparse 모델, 리랭커, Contextual 강화 여부** 네 축을 독립적으로 선택합니다.

- **Layer 1 — Dense Model**: 의미적 유사도 검색의 핵심 (kosimcse, e5, bge-m3, openai-large, upstage)
- **Layer 2 — Sparse Model**: 키워드 정확 매칭으로 Dense 보완 (korean_bm25, splade)
- **Layer 3 — Reranker**: Hybrid 결과를 정밀 재순위화 (none, colbert, flashrank)
- **Layer 4 — Contextual**: 인덱싱 시 LLM 문맥 부착, Layer 1~3 어떤 조합에도 독립 적용 가능 (none, contextual)

4-Layer 설계로 바꾸면:
- 각 레이어의 **독립적 기여도**를 분석할 수 있음 (예: "BGE-M3가 다른 Dense 모델보다 평균 15% 우수")
- **최적 조합**을 찾을 수 있음 (예: "e5+splade+flashrank가 의외로 1위")
- **비용 대비 효과**를 판단할 수 있음 (예: "contextual은 인덱싱 시 1회 비용으로 품질 X% 향상")

### 왜 2-Pass 실행인가

60개 조합 전부에 RAGAS를 돌리면 OpenAI API 비용이 과도합니다.

```
Pass 1: 60개 전략 × 20 쿼리 = 1,200회 검색  → 레이턴시만 (API 비용 $0)
Pass 2: 상위 10개 전략 × 20 쿼리 = 200회 평가 → RAGAS (API 비용 ~$2-5)
```

API 호출을 83% 절감하면서도 의미 있는 상위권 전략의 품질을 평가할 수 있습니다.

## 아키텍처

### 전체 시스템 구조

```
┌────────────────────────────────────────────────────────────────┐
│                       BenchmarkRunner                          │
│  ┌──────────┐ ┌──────────┐ ┌───────────────┐ ┌────────────┐    │
│  │  Dense   │ │  ColBERT │ │ColBERT Rerank │ │  FlashRank │    │
│  │ +Sparse  │ │ (PyLate) │ │ (2-stage)     │ │   Rerank   │    │
│  └────┬─────┘ └────┬─────┘ └──────┬────────┘ └──────┬─────┘   │
│       │       ┌──────────────┐ ┌────────────┐   │              │
│       │       │ Contextual   │ │  OpenAI    │   │              │
│       │       │  Retrieval   │ │  Embed     │   │              │
│       │       └──────┬───────┘ └─────┬──────┘   │              │
│       │       ┌──────────────┐        │          │              │
│       │       │   Upstage    │        │          │              │
│       │       │   Embed      │        │          │              │
│       │       └──────┬───────┘        │          │              │
│       └──────────────┴────────────────┴──────────┘              │
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

### 전체 실행 흐름

```
  docs/*.pdf                                 QA 데이터셋
      │                                           │
      │ [--sample_pages]                          │
      ▼                                           │
  pdfs_to_markdowns()  ──→  rag_bench/docs/*.md  │
                                    │             │
                                    ▼             │
                       create_parent_child_chunks()│
                                    │             │
                                    ▼             │
                       effective_num_qa 산출       │
                                    │             │
                                    ▼             │
                       RAGAS KG 구축 + QA 생성     │
                                    │             │
                                    ▼             ▼
                            qa_dataset.json ──────┤
                                                  │
                    ┌─────────────────────────────┘
                    │
                    ▼
              run_all_combos.py
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       Pass 1              Pass 2
   (60개 전략            (top_n 전략
    레이턴시 측정)         RAGAS 평가)
          │                   │
          └─────────┬─────────┘
                    ▼
        all_combos_latency.csv
        all_combos_ragas.csv
        e2e_report.md
        benchmark_report.html
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
          │  or BGE-M3      │ │                 │
          │  or OpenAI 3072d│ │                 │
          │  or Upstage 4096│ │                 │
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
├── run_tracker.py           # RunTracker — 수행 이력 추적 (플랫폼, 타이밍, 토큰)
├── cli.py                   # RAGChat — Agentic RAG 대화 인터페이스
│
├── strategies/              # RAG 전략 구현체 (7종)
│   ├── __init__.py          # 전략 클래스 일괄 export
│   ├── dense_sparse.py      # ★ Dense+Sparse Hybrid (5종 임베딩 + 2종 희소 검색)
│   ├── colbert.py           # ColBERT Late Interaction (PyLate, MaxSim)
│   ├── colbert_rerank.py    # ColBERT 2-stage Reranking (Decorator Pattern)
│   ├── flashrank_rerank.py  # FlashRank 경량 ONNX Reranking (Decorator)
│   ├── contextual_retrieval.py  # Contextual Retrieval (인덱싱 시 LLM 문맥 부착)
│   ├── openai_embed.py      # OpenAI text-embedding-3-small/large Dense 검색
│   └── upstage_embed.py     # Upstage solar-embedding-1-large (passage/query 분리)
│
├── indexing/                # 문서 처리 파이프라인
│   ├── __init__.py          # pdfs_to_markdowns, create_parent_child_chunks export
│   ├── pdf_converter.py     # PDF → Markdown 변환 (pymupdf4llm)
│   └── chunker.py           # Parent-Child 청킹 (Parent: 2000-10000자, Child: 300-500자)
│
├── evaluation/              # RAGAS 평가 모듈
│   ├── __init__.py          # RAGEvaluator, SCORING_PROFILES, MetricPreset export
│   ├── evaluator.py         # ExtendedRAGEvaluator — 샘플별 평가 + 비용 추적 + comprehensive 프로파일
│   ├── metrics.py           # MetricRegistry (Core 4종, Extended 5종, Lightweight 2종) + COMPREHENSIVE 프리셋
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
│   ├── generate_qa.py       # QA 데이터셋 자동 생성 (RAGAS KG, PDF 페이지 샘플링 통합)
│   ├── generate_html_report.py  # ★ HTML 벤치마크 보고서 생성 (Bootstrap, 차트 인라인, Executive Summary 추천 카드, 레이어 기여도 분석, Contextual ON/OFF 탭)
│   ├── run_bench.py         # 3종 전략 벤치마크 + RAGAS 평가
│   ├── run_all_combos.py    # ★ 60개 4-Layer 교차 조합 벤치마크 (2-Pass)
│   └── bench_visualize.ipynb # 시각화 노트북 (10섹션, 수행 이력 포함)
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
│   ├── e2e_report.md            # 종합 리포트 (실행 환경 + 비중% 포함)
│   └── run_history/             # 수행 이력 JSON (플랫폼, 타이밍, 토큰)
│       ├── run_*.json           # 실행별 상세 기록
│       └── latest.json          # 최신 실행 심링크
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

**Dense 모델 5종:**

| 키 | 모델 | 차원 | 한국어 | 비고 |
|----|------|------|:------:|------|
| `kosimcse` | KoSimCSE-roberta-multitask | 768 | ★★★ | HF |
| `e5` | multilingual-e5-large | 1024 | ★★ | HF |
| `bge-m3` | BGE-M3 | 1024 | ★★ | HF |
| `openai-large` | text-embedding-3-large | 3072 | ★★ | 유료 API |
| `upstage` | solar-embedding-1-query | 4096 | ★★★ | 유료 API |

**Sparse 모델 2종:**

| 키 | 방식 | 특징 |
|----|------|------|
| `korean_bm25` | BM25 + KoNLPy OKt | 한국어 형태소 분석 기반, 최고 한국어 성능 |
| `splade` | SPLADE (learned sparse) | 학습된 희소 표현, 다국어 지원 |

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

### 6. OpenAIEmbedStrategy (strategies/openai_embed.py)

OpenAI `text-embedding-3-small` / `text-embedding-3-large` 기반 순수 Dense 검색.

```python
from rag_bench.strategies import OpenAIEmbedStrategy

strategy = OpenAIEmbedStrategy(model="text-embedding-3-small")
strategy.index(documents)
results = strategy.retrieve("AI 산업 동향은?", k=3)
```

| 모델 | 차원 | 특징 |
|------|------|------|
| `text-embedding-3-small` | 1536 | 경제적, 빠름 |
| `text-embedding-3-large` | 3072 | 높은 품질 |

환경변수: `OPENAI_API_KEY`

### 7. UpstageEmbedStrategy (strategies/upstage_embed.py)

Upstage `solar-embedding-1-large` 기반 Dense 검색. 인덱싱(passage)과 쿼리(query) 모델을 분리 운용합니다.

```python
from rag_bench.strategies import UpstageEmbedStrategy

strategy = UpstageEmbedStrategy()  # passage 모델로 인덱싱, query 모델로 검색
strategy.index(documents)
results = strategy.retrieve("AI 산업 동향은?", k=3)
```

환경변수: `UPSTAGE_API_KEY`

### 8. 문서 처리 파이프라인 (indexing/)

```
PDF 문서 → pdf_converter.py → Markdown → chunker.py → Parent-Child 청크
                                          ├── Parent: 2000-10000자 (전체 문맥)
                                          └── Child: 300-500자 (검색 단위)
```

Parent-Child 전략: 검색은 작은 Child 청크로 정밀하게, 답변 생성은 큰 Parent 청크로 풍부하게.

### 9. RAGAS 평가 (evaluation/)

RAGAS v0.4+ 기반 메트릭 체계:

**Core 4종 (LLM 필요):**

| 메트릭 | 분류 | 평가 대상 |
|--------|------|----------|
| Context Precision | Retrieval | 검색 결과의 정밀도 |
| Context Recall | Retrieval | 검색 결과의 재현율 |
| Faithfulness | Generation | 답변의 사실 충실도 |
| Answer Relevancy | Generation | 답변의 질문 적합도 |

**Extended 5종 (COMPREHENSIVE 프리셋):**

| 메트릭 | 분류 | 평가 대상 |
|--------|------|----------|
| Context Entity Recall | Retrieval | 엔터티 재현율 |
| Response Relevancy | Generation | 응답 관련성 |
| String Presence | Lightweight | 정답 문자열 포함 |
| Exact Match | Lightweight | 정확 일치 |
| Non-LLM String Similarity | Lightweight | 문자열 유사도 |

**Scoring Profiles:** `default` (Core 4종), `comprehensive` (Core + Extended), `lightweight` (LLM 불필요)

`ExtendedRAGEvaluator`는 샘플별 평가 + API 토큰 사용량/비용 추적을 지원합니다.

### 10. 수행 이력 추적 (run_tracker.py)

벤치마크 실행의 상세 이력을 JSON으로 기록하는 추적 모듈.

**주요 클래스:**
- `RunTracker`: 벤치마크 수행 이력 추적기. `phase()` 컨텍스트 매니저로 단계별 시간 자동 측정.
- `TokenUsage`: LLM API 토큰 사용량 (prompt/completion/total/cost).
- `StrategyTiming`: 전략별 빌드/쿼리 타이밍 + RAGAS 점수.
- `BenchmarkRunRecord`: 전체 실행 기록 데이터 구조.

**주요 함수:**
- `collect_platform_info()`: 실행 환경 자동 수집 (OS, CPU, RAM, GPU, Apple Silicon, Git commit).
- `track_openai_tokens()`: LangChain `get_openai_callback()` 기반 토큰 추적 컨텍스트 매니저.

```python
from rag_bench.run_tracker import RunTracker, track_openai_tokens

tracker = RunTracker(output_dir=Path("_benchdata"))

with tracker.phase("chunking"):
    # 청킹 로직 — 소요 시간 자동 측정

with tracker.phase("ragas_evaluation"):
    with track_openai_tokens() as usage:
        # LLM 호출 — 토큰 사용량 자동 추적
    tracker.add_tokens(usage)

filepath = tracker.finalize()  # JSON 저장 + latest.json 심링크
```

**저장 형식:** `_benchdata/run_history/run_{YYYYMMDD_HHMMSS}.json`
- 콘솔에 단계별 비중(%) 요약 자동 출력
- `latest.json` 심링크로 최신 실행에 바로 접근

### 11. LangGraph 에이전트 (graph/)

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
# Upstage 전략 사용 시
echo "UPSTAGE_API_KEY=up_..." >> .env
```

### 1. 60개 조합 벤치마크 (권장)

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
    DenseSparseStrategy(dense="kosimcse", sparse_type="korean_bm25"),
    DenseSparseStrategy(dense="bge-m3", sparse_type="splade"),
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

strategy = DenseSparseStrategy(dense="kosimcse", sparse_type="korean_bm25")
strategy.index(children)
```

### 4. HTML 보고서 생성

```python
import pandas as pd
from rag_bench.scripts.generate_html_report import generate_html_report

latency_df = pd.read_csv("_benchdata/all_combos_latency.csv")
ragas_df = pd.read_csv("_benchdata/all_combos_ragas.csv")
generate_html_report(
    latency_df,
    ragas_df,
    output_path="_benchdata/benchmark_report.html",
    history_dir="_benchdata/run_history",  # 수행 이력 비교 활성화 (선택)
)
# → 브라우저에서 바로 열 수 있는 독립 HTML 파일 생성
```

**HTML 보고서 주요 섹션:**
- **Executive Summary**: RAGAS 가중 점수 1위 전략 자동 추천 + 선정 이유 + 레이턴시 배지
- **레이어별 기여도 분석**: Reranker(없음/ColBERT/FlashRank), Contextual(ON/OFF) 그룹별 평균 점수 + 향상폭(Δ) + 동일 base 전략 1:1 순수 효과 비교
- **RAGAS 테이블**: Contextual OFF / ON 탭으로 분리 (Bootstrap 탭 UI)
- **레이턴시 테이블**: IQR×1.5 이상치 감지 시 ⚠️ 배지 + 중앙값 부표시
- **산점도**: 품질 상위 10개 전략, ColBERT(파란색) / non-ColBERT(녹색) 색상 분리, 실용 한계 3s 기준선
- **레이더 차트**: 최고 전략(파란색) vs 최저 전략(빨간색) 1:1 비교

### 5. Agentic RAG 대화

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

## 4-Layer 조합 벤치마크 상세

### 프리셋

| 프리셋 | Layer 1 Dense | Layer 2 Sparse | Layer 3 Reranker | Layer 4 Contextual | 조합 수 |
|--------|:-----:|:------:|:--------:|:-----------:|:-------:|
| `quick` | 1 (bge-m3) | 1 (korean_bm25) | 2 (none, flashrank) | 1 (none) | **2** |
| `standard` | 5 (HF 3 + 유료 2) | 2 | 2 (none, flashrank) | 1 (none) | **20** |
| `full` | 5 | 2 | 3 (none, colbert, flashrank) | 2 (none, contextual) | **60** |

### 인덱스 캐싱 메커니즘

60개 조합이지만, 실제 인덱싱이 필요한 건 (Dense, Sparse) 쌍의 수 = 5×2 = 10개뿐.

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
Step 3: 60개 전략 생성 + 인덱싱 (10개 고유 인덱스)
Step 4: Pass 1 — 60개 전략 × 20 쿼리 = 1,200회 검색 → 레이턴시
Step 5: Pass 2 — 상위 10개 전략 × 20 쿼리 = 200회 검색 → RAGAS 평가
Step 6: 리포트 생성 (e2e_report.md)
```

### 레이어별 기여도 분석 (`--layers`)

`--layers` 플래그를 사용하면 각 레이어의 독립적 기여도를 분석합니다:

```
Layer 1 — Dense Model:
  kosimcse     → 0.234s (n=12)
  e5           → 0.456s (n=12)
  bge-m3       → 0.345s (n=12)
  openai-large → 0.512s (n=12)
  upstage      → 0.498s (n=12)

Layer 2 — Sparse Model:
  korean_bm25 → 0.312s (n=30)
  splade      → 0.289s (n=30)

Layer 3 — Reranker:
  none       → 0.156s (n=20)
  colbert    → 0.523s (n=20)
  flashrank  → 0.178s (n=20)

Layer 4 — Contextual:
  none        → 0.312s (n=30)
  contextual  → 0.298s (n=30)  ← 쿼리 레이턴시는 동일 수준, 인덱싱 시 1회 비용
```

## 벤치마크 CLI 옵션 전체

### run_all_combos.py

```
공통:
  --k K                검색 결과 수 (기본: 3)
  --no_ragas           RAGAS 평가 건너뛰기
  --reindex            기존 인덱스 삭제 후 재인덱싱

4-Layer 조합 모드 (--preset 사용):
  --preset PRESET      quick(2) | standard(20) | full(60)
  --pass1-only         레이턴시만 (RAGAS 없음)
  --top_n N            상위 N개만 RAGAS 평가
  --dry-run            조합 목록만 출력
  --layers             레이어별 기여도 분석 (Layer 1~4 독립 분석)
```

### generate_qa.py

```
  --num_qa N                생성할 QA 쌍 수 (기본: 20)
  --sample_pages            docs/*.pdf를 페이지 샘플링하여 rag_bench/docs/*.md 재생성
  --page_sample_ratio F     페이지 샘플링 비율 (기본: 0.1 = 10%%, --sample_pages와 함께 사용)
  --max_sample_pages N      최대 샘플 페이지 수 (기본: 5, --sample_pages와 함께 사용)
  --max_qa_per_page N       청크당 최대 QA 수 (기본: 2, --sample_pages와 함께 QA 상한 계산)
  --force                   캐시 무시하고 강제 재생성
  --build-kg-only           KG만 구축, QA 생성 안 함
  --reuse-kg                기존 KG 파일 재사용
  --query-dist DIST         쿼리 분포: single_hop | multi_hop | balanced (기본: balanced)
  --num-personas N          자동 페르소나 수 (기본: 3)
```

#### QA 생성 파이프라인 흐름

```
generate_qa.py main()
│
├─ [Step 0 — --sample_pages 시]
│    docs/*.pdf
│        │  pdfs_to_markdowns(
│        │      sample_pages=True,
│        │      page_sample_ratio=0.1,  # 10%
│        │      max_sample_pages=5
│        │  )
│        ▼
│    rag_bench/docs/*.md  (샘플링된 텍스트)
│
├─ [Step 1] create_parent_child_chunks(BENCH_DOCS_DIR)
│    └─→ parent_pairs: List[(parent_doc, [child_docs])]
│
├─ [Step 2] _compute_effective_num_qa(args, parent_pairs)
│    └─→ effective = min(args.num_qa, len(parent_pairs) × args.max_qa_per_page)
│
├─ [Step 3] _generate_qa_ragas(parent_pairs, num_qa=effective, ...)
│    ├─ KnowledgeGraph 구축 (저장: KG_SAVE_PATH)
│    │    또는 --reuse-kg 시 기존 KG 파일 로드
│    ├─ TestsetGenerator(llm, embeddings)
│    └─ generator.generate(num_qa, query_distribution=[...])
│         └─→ qa_pairs: List[{"question": ..., "ground_truth": ...}]
│
└─ [Step 4] 저장
     └─→ _benchdata/qa_dataset.json
          {"qa_pairs": [...], "num_qa": N, "sampled_pages": bool}
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

4-Layer 조합에 새 레이어 값을 추가하려면 `combo/spec.py`의 `PRESETS` 딕셔너리를 수정합니다.

## 의존성

전체 의존성은 `pyproject.toml`을 참고하세요. 주요 항목:

**핵심:**
- `langchain-core`, `langchain-text-splitters`, `langchain-qdrant`
- `qdrant-client`, `pandas`, `python-dotenv`

**전략별:**
- Dense+Sparse: `konlpy` (OKt), `transformers` (SPLADE), `fastembed`, `sentence-transformers`
- ColBERT: `pylate`, `sentence-transformers`
- FlashRank: `flashrank` (ONNX 기반)
- Contextual / OpenAI Embed: `langchain-openai` (OPENAI_API_KEY 필요)
- Upstage Embed: `langchain-upstage` (UPSTAGE_API_KEY 필요)

**평가:**
- `ragas>=0.4.3`, `datasets`

**Agentic RAG:**
- `langgraph`, `langchain-openai`
