# rag_bench — 모듈화 RAG 벤치마크 시스템

Strategy Pattern 기반으로 다양한 RAG 방식을 통일된 인터페이스로 비교 벤치마크하는 LangChain/LangGraph 통합 패키지.

## 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                    BenchmarkRunner                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │  Dense   │  │  ColBERT │  │  Graph   │  │  Custom  │ │
│  │ +Sparse  │  │(RAGatou) │  │(NodeRAG) │  │  (확장)  │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       └─────────────┴──────┬──────┴─────────────┘        │
│                  ┌─────────▼─────────┐                   │
│                  │  BaseRAGStrategy  │  ← ABC            │
│                  │  index()          │                    │
│                  │  retrieve()       │                    │
│                  │  get_retriever()  │                    │
│                  └───────────────────┘                    │
└──────────────────────────────────────────────────────────┘
```

## 패키지 구조

```
rag_bench/
├── __init__.py              # 패키지 진입점
├── config.py                # 전역 설정 (경로, LLM, SSL)
├── base.py                  # BaseRAGStrategy ABC
├── runner.py                # BenchmarkRunner (전략 비교)
├── cli.py                   # RAGChat (대화 인터페이스)
├── strategies/              # RAG 전략 모듈
│   ├── dense_sparse.py      # 6가지 Dense+Sparse 조합
│   ├── colbert.py           # ColBERT 스텁 (TODO)
│   └── graph_rag.py         # GraphRAG 스텁 (TODO)
├── indexing/                # 문서 처리
│   ├── pdf_converter.py     # PDF → Markdown (pymupdf4llm)
│   └── chunker.py           # Parent-Child 청킹
└── graph/                   # LangGraph 에이전트
    ├── state.py             # State, AgentState, QueryAnalysis
    ├── prompts.py           # 시스템 프롬프트 4종
    ├── nodes.py             # 그래프 노드 + 라우팅
    └── builder.py           # build_agent_graph()
```

## 빠른 시작

### 1. 전략 비교 벤치마크

```python
from rag_bench import BenchmarkRunner
from rag_bench.strategies import DenseSparseStrategy

# 비교할 전략 선택
strategies = [
    DenseSparseStrategy(combo_id=1),  # 한국어 최적 (KoSimCSE + BM25/OKt)
    DenseSparseStrategy(combo_id=4),  # 경량 (MiniLM + BM25)
]

# 벤치마크 실행
queries = ["쿠버네티스 Pod란?", "Docker와 VM의 차이"]
runner = BenchmarkRunner(strategies, queries)
results = runner.run()
runner.compare()

# pandas DataFrame 변환
df = runner.to_dataframe()
```

### 2. 문서 인덱싱 파이프라인

```python
from rag_bench.indexing import pdfs_to_markdowns, create_parent_child_chunks
from rag_bench.strategies import DenseSparseStrategy

# PDF → Markdown → Parent-Child 청킹
pdfs_to_markdowns(docs_dir="docs", output_dir="markdown")
parents, children = create_parent_child_chunks(
    markdown_dir="markdown",
    parent_store_path="parent_store",
)

# 전략에 인덱싱
strategy = DenseSparseStrategy(combo_id=1)
strategy.index(children)
```

### 3. Agentic RAG 대화

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

## 임베딩 조합 상세

| # | 조합 | Dense Model | Sparse Model | 한국어 | 비용 |
|---|------|-------------|-------------|--------|------|
| 1 | 한국어 최적 | KoSimCSE-roberta (768d) | BM25+OKt | ★★★ | 무료 |
| 2 | 다국어 균형 | E5-large (1024d) | SPLADE | ★★☆ | 무료 |
| 3 | 올인원 통합 | BGE-M3 (1024d) | BGE-M3 | ★★☆ | 무료 |
| 4 | 경량/빠른 | MiniLM-L6 (384d) | BM25 | ★☆☆ | 무료 |
| 5 | 고성능 API | OpenAI Large (3072d) | SPLADE | ★★☆ | 유료 |
| 6 | 한국어 API | Upstage Solar (4096d) | BM25+OKt | ★★★ | 유료 |

## 새 전략 추가하기

`BaseRAGStrategy`를 상속하여 5개 메서드를 구현하면 자동으로 벤치마크에 편입됩니다:

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
        # 문서 인덱싱 로직
        ...

    def retrieve(self, query, k=5):
        # 검색 로직
        ...

    def get_retriever(self, k=5):
        # LangChain Retriever 반환
        ...
```

## Future TODO

- [ ] **RAGAS 벤치마크 통합** — LLM 기반 평가 메트릭 (Faithfulness, Context Precision 등)
- [ ] **ColBERT/RAGatouille 실제 구현** — `jinaai/jina-colbert-v2` 한국어 벤치마크
- [ ] **NodeRAG/GraphRAG 실제 구현** — 이질적 그래프 기반 다중 홉 추론
- [ ] **통합 노트북** — `benchmark_lab.ipynb` 작성

## 의존성

**필수:**
- `langchain-core`, `langchain-text-splitters`
- `qdrant-client`, `langchain-qdrant`
- `pydantic`

**전략별 선택:**
- 조합 1, 6: `konlpy` (OKt 형태소 분석)
- 조합 2, 5: `transformers` (SPLADE)
- 조합 3, 4: `langchain-qdrant[fastembed]`
- 조합 5: `langchain-openai`
- 조합 6: `langchain-upstage`

**Agentic RAG:**
- `langgraph`, `langchain-openai`
