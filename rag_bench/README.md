# rag_bench — 모듈화 RAG 벤치마크 시스템

Strategy Pattern 기반으로 다양한 RAG 방식을 통일된 인터페이스로 비교 벤치마크하는 LangChain/LangGraph 통합 패키지.

> 이 디렉토리 단독으로 공유 가능합니다. `pyproject.toml`, `uv.lock`, `docker-compose.yml`이 포함되어 있습니다.

## 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                      BenchmarkRunner                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐   │
│  │  Dense   │ │  ColBERT │ │  Graph   │ │ColBERT Rerank │   │
│  │ +Sparse  │ │ (PyLate) │ │(LightRAG)│ │ (2-stage)     │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬────────┘   │
│       │       ┌────────────┐ ┌──────────────┐   │            │
│       │       │ FlashRank  │ │ Contextual   │   │            │
│       │       │  Rerank    │ │  Retrieval   │   │            │
│       │       └─────┬──────┘ └──────┬───────┘   │            │
│       └─────────────┴──────┬────────┴───────────┘            │
│                  ┌─────────▼─────────┐                       │
│                  │  BaseRAGStrategy  │  ← ABC                │
│                  │  index()          │                        │
│                  │  retrieve()       │                        │
│                  │  get_retriever()  │                        │
│                  └───────────────────┘                        │
│                            │                                  │
│                  ┌─────────▼─────────┐                       │
│                  │   RAGEvaluator    │  ← RAGAS 평가          │
│                  └───────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

## 패키지 구조

```
rag_bench/
├── __init__.py              # 패키지 진입점
├── config.py                # 전역 설정 (경로, LLM, SSL)
├── base.py                  # BaseRAGStrategy ABC
├── runner.py                # BenchmarkRunner (전략 비교)
├── evaluation.py            # RAGEvaluator (RAGAS 평가)
├── cli.py                   # RAGChat (대화 인터페이스)
├── strategies/              # RAG 전략 모듈 (6종)
│   ├── dense_sparse.py      # 4가지 Dense+Sparse 조합
│   ├── colbert.py           # ColBERT Late Interaction (PyLate)
│   ├── colbert_rerank.py    # ColBERT 2-stage 리랭킹
│   ├── flashrank_rerank.py  # FlashRank 경량 리랭킹 (ONNX, CPU)
│   ├── contextual_retrieval.py # Contextual Retrieval (LLM 문맥 부착)
│   └── graph_rag.py         # GraphRAG (LightRAG 기반)
├── indexing/                # 문서 처리
│   ├── pdf_converter.py     # PDF → Markdown (pymupdf4llm)
│   └── chunker.py           # Parent-Child 청킹
├── graph/                   # LangGraph 에이전트
│   ├── state.py             # State, AgentState, QueryAnalysis
│   ├── prompts.py           # 시스템 프롬프트 4종
│   ├── nodes.py             # 그래프 노드 + 라우팅
│   └── builder.py           # build_agent_graph()
├── scripts/                 # 벤치마크 실행 스크립트
│   ├── generate_qa.py       # QA 데이터셋 자동 생성
│   ├── run_bench.py         # 3종 통합 벤치마크 + RAGAS
│   └── run_all_combos.py    # 전체 15종 조합 비교
├── docs/                    # 벤치마크 대상 문서 (*.md)
├── _benchdata/              # 벤치마크 중간 산출물 (.gitignore)
├── pyproject.toml           # 의존성 정의
├── uv.lock                  # 버전 잠금
├── docker-compose.yml       # Qdrant 컨테이너
└── .python-version          # Python 3.12
```

## 빠른 시작

### 0. 환경 설정

```bash
# 의존성 설치
uv sync

# Qdrant 실행 (DenseSparse 전략에 필요)
docker compose up -d

# .env 파일에 OpenAI API 키 설정
echo "OPENAI_API_KEY=sk-..." > .env
```

### 1. 벤치마크 실행 (CLI)

```bash
# Step 1: docs/에 Markdown 문서 배치 (이미 포함됨)

# Step 2: QA 데이터셋 자동 생성
python -m rag_bench.scripts.generate_qa --num_qa 20

# Step 3-A: 3종 벤치마크 (DenseSparse + ColBERT + ColBERTRerank)
python -m rag_bench.scripts.run_bench --k 3

# Step 3-B: 전체 15종 조합 비교
python -m rag_bench.scripts.run_all_combos

# Step 3-C: 특정 조합만 선택
python -m rag_bench.scripts.run_all_combos --combos 1,3,4 --skip_rerank
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

## 전략 상세

### DenseSparse 임베딩 4종 조합

| # | 조합 | Dense Model | Sparse Model | 한국어 | 비용 |
|---|------|-------------|-------------|--------|------|
| 1 | 한국어 최적 | KoSimCSE-roberta (768d) | BM25+OKt | ★★★ | 무료 |
| 2 | 다국어 균형 | E5-large (1024d) | SPLADE | ★★☆ | 무료 |
| 3 | 올인원 통합 | BGE-M3 (1024d) | BGE-M3 | ★★☆ | 무료 |
| 4 | 경량/빠른 | MiniLM-L6 (384d) | BM25 | ★☆☆ | 무료 |

### ColBERT / ColBERTRerank

| 전략 | 모델 | 방식 | 특징 |
|------|------|------|------|
| ColBERT | jina-colbert-v2 | Late Interaction MaxSim | 전체 코퍼스 인코딩, 높은 정확도 |
| ColBERTRerank | jina-colbert-v2 | 2-stage 리랭킹 | 후보 N개만 인코딩, 임의의 base 전략 위에 적용 |

### FlashRank Rerank

| 전략 | 모델 | 방식 | 특징 |
|------|------|------|------|
| FlashRank | ms-marco-MultiBERT-L-12 | ONNX 리랭킹 | CPU 전용, ~150MB, 100+ 언어 |

### Contextual Retrieval

| 전략 | 방식 | 특징 |
|------|------|------|
| Contextual | LLM 문맥 부착 + base 전략 | Anthropic 방식, 인덱싱 시 LLM으로 청크 문맥 생성 |

### GraphRAG

| 전략 | 백엔드 | 모드 | 특징 |
|------|--------|------|------|
| GraphRAG | LightRAG | local/global/hybrid | LLM 기반 엔터티/관계 추출, 지식 그래프 검색 |

## 벤치마크 스크립트

| 스크립트 | 설명 | 사용법 |
|----------|------|--------|
| `generate_qa.py` | docs/*.md에서 QA 자동 생성 (GPT-4o-mini) | `python -m rag_bench.scripts.generate_qa --num_qa 20` |
| `run_bench.py` | 3종 전략 벤치마크 + RAGAS 평가 | `python -m rag_bench.scripts.run_bench --k 3` |
| `run_all_combos.py` | 최대 15종 전체 조합 비교 | `python -m rag_bench.scripts.run_all_combos` |

### run_all_combos.py 옵션

```
--k K                검색 결과 수 (기본: 3)
--combos 1,3,4       DenseSparse 조합 ID 지정 (미지정 시 전체)
--skip_colbert       ColBERT 단독 전략 건너뛰기
--skip_rerank        ColBERTRerank 전략 건너뛰기
--skip_graphrag      GraphRAG 전략 건너뛰기
--skip_contextual    Contextual Retrieval 건너뛰기
--skip_flashrank     FlashRank Rerank 건너뛰기
--no_ragas           RAGAS 평가 건너뛰기 (레이턴시만 측정)
--reindex            기존 인덱스 삭제 후 재인덱싱 (기본: 기존 인덱스 재사용)
--contextual_base N  Contextual Retrieval 기반 조합 ID (기본: 3=BGE-M3)
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
        ...

    def retrieve(self, query, k=5):
        ...

    def get_retriever(self, k=5):
        ...
```

## 의존성

전체 의존성은 `pyproject.toml`을 참고하세요. 주요 항목:

**핵심:**
- `langchain-core`, `langchain-text-splitters`, `langchain-qdrant`
- `qdrant-client`, `pandas`, `python-dotenv`

**전략별:**
- 조합 1: `konlpy` (OKt 형태소 분석)
- 조합 2: `transformers` (SPLADE)
- 조합 3, 4: `fastembed`
- ColBERT: `pylate`, `sentence-transformers`
- FlashRank: `flashrank` (ONNX 기반)
- GraphRAG: `lightrag-hku`, `nest-asyncio`

**평가:**
- `ragas`, `datasets`

**Agentic RAG:**
- `langgraph`, `langchain-openai`
