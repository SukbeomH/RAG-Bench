# ARCHITECTURE.md

## Overview

**rag-bench** is a modular RAG (Retrieval-Augmented Generation) benchmarking system.
Its core purpose is to exhaustively compare combinations of retrieval strategies across
four configurable layers (Dense model × Sparse model × Reranker × Contextual LLM enhancement)
and evaluate them with RAGAS metrics in a reproducible, two-pass pipeline.

---

## Directory Layout

```
autorag/
├── pyproject.toml                  # Project metadata and dependency declarations
├── docker-compose.yml              # Qdrant server (optional remote mode)
├── .env                            # API keys (OPENAI_API_KEY, UPSTAGE_API_KEY, …)
│
├── rag_bench/                      # Core library package
│   ├── __init__.py
│   ├── base.py                     # BaseRAGStrategy + StrategyRetriever (ABC)
│   ├── config.py                   # Global paths, model constants, SSL bypass
│   ├── runner.py                   # BenchmarkRunner — 2-Pass execution engine
│   ├── run_tracker.py              # RunTracker — timing, token, platform telemetry
│   ├── cli.py                      # RAGChat — Jupyter/REPL interactive interface
│   │
│   ├── combo/                      # 4-Layer combination management
│   │   ├── __init__.py             # Re-exports all combo symbols
│   │   ├── spec.py                 # ComboSpec dataclass + PRESETS + generator
│   │   ├── cache.py                # CacheConfig + IndexCacheManager
│   │   └── builder.py              # build_strategy_from_spec()
│   │
│   ├── strategies/                 # Concrete RAG strategy implementations
│   │   ├── __init__.py
│   │   ├── dense_sparse.py         # DenseSparseStrategy (base hybrid retriever)
│   │   ├── colbert.py              # ColBERTStrategy (standalone ColBERT retrieval)
│   │   ├── colbert_rerank.py       # ColBERTRerankStrategy (Decorator over any base)
│   │   ├── flashrank_rerank.py     # FlashRankRerankStrategy (Decorator over any base)
│   │   ├── contextual_retrieval.py # ContextualRetrievalStrategy (Layer 4 enrichment)
│   │   ├── openai_embed.py         # OpenAIEmbedStrategy
│   │   └── upstage_embed.py        # UpstageEmbedStrategy
│   │
│   ├── evaluation/                 # RAGAS evaluation subsystem
│   │   ├── __init__.py
│   │   ├── evaluator.py            # ExtendedRAGEvaluator, EvaluationReport, rank_strategies
│   │   └── metrics.py              # MetricPreset enum + METRIC_REGISTRY + create_metrics()
│   │
│   ├── graph/                      # LangGraph agentic RAG
│   │   ├── __init__.py
│   │   ├── builder.py              # build_agent_graph() — compiles LangGraph pipeline
│   │   ├── state.py                # State, AgentState, QueryAnalysis (Pydantic)
│   │   ├── nodes.py                # Graph node functions
│   │   └── prompts.py              # Prompt templates
│   │
│   ├── indexing/                   # Document preprocessing
│   │   ├── __init__.py
│   │   ├── pdf_converter.py        # PDF → Markdown via pymupdf4llm
│   │   └── chunker.py              # Parent-Child chunking pipeline
│   │
│   ├── utils/                      # Shared utilities
│   │   ├── __init__.py
│   │   ├── device.py               # detect_device() (CUDA → CPU)
│   │   ├── qa_loader.py            # load_qa_dataset()
│   │   └── report.py               # Console table formatters
│   │
│   ├── scripts/                    # Runnable entry-point scripts
│   │   ├── run_all_combos.py       # Main CLI: 4-Layer benchmark orchestrator
│   │   ├── run_bench.py            # Single strategy benchmark runner
│   │   ├── generate_qa.py          # RAGAS KG-based QA dataset generator
│   │   ├── generate_html_report.py # HTML report generator
│   │   ├── prefetch_models.py      # Pre-download HuggingFace models
│   │   ├── timing_report.py        # Timing analysis report
│   │   └── bench_visualize.ipynb   # Visualization notebook
│   │
│   └── docs/                       # Benchmark target markdown documents (runtime)
│
├── rag_bench_colab/                # Google Colab environment adapter
│   ├── __init__.py
│   ├── colab_config.py             # Drive paths, CUDA device config
│   ├── colab_runner.py             # ColabBenchmarkRunner + CheckpointManager
│   ├── colab_visualizer.py         # Colab-specific visualisation helpers
│   └── rag_benchmark.ipynb         # Main Colab notebook
│
└── scripts/                        # Project-level utility scripts
    ├── verify_env.py
    ├── verify_rag_bench.py
    ├── verify_ragas_eval.py
    ├── verify_graphrag.py
    └── cleanup_legacy_indexes.py
```

---

## Core Architecture Patterns

### 1. Strategy Pattern — `BaseRAGStrategy`

Every retrieval approach (dense+sparse, ColBERT, contextual, etc.) implements the same
abstract interface defined in `rag_bench/base.py`:

```
BaseRAGStrategy (ABC)
├── name: str             (property)
├── description: str      (property)
├── index(documents)      (abstract)
├── retrieve(query, k)    (abstract)
├── get_retriever(k)      (abstract)
├── is_ready: bool        (optional override)
└── cleanup()             (optional override)
```

`StrategyRetriever` adapts any `BaseRAGStrategy` into the LangChain `BaseRetriever` interface,
eliminating per-strategy boilerplate.

### 2. Decorator Pattern — Reranker and Contextual Layers

Rerankers and the contextual enrichment layer wrap a `base_strategy` and delegate
`index()` and `retrieve()` to it, adding their own logic on top:

```
DenseSparseStrategy (Layer 1+2)
  └─ wrapped by ColBERTRerankStrategy    (Layer 3)
  └─ wrapped by FlashRankRerankStrategy  (Layer 3)

DenseSparseStrategy (Layer 1+2)
  └─ wrapped by ContextualRetrievalStrategy (Layer 4)
       └─ wrapped by ColBERTRerankStrategy  (Layer 3, stacked)
```

Stacking is handled by `combo/builder.py`: the builder first constructs the base strategy,
then applies the contextual layer (if any), then wraps in a reranker (if any).

### 3. 4-Layer Cartesian Product — `combo/spec.py`

```
Layer 1  Dense Model    kosimcse | e5 | bge-m3 | openai-large | upstage
Layer 2  Sparse Model   korean_bm25 | splade
Layer 3  Reranker       none | colbert | flashrank
Layer 4  Contextual     none | contextual
```

`generate_valid_combinations(config)` produces the full Cartesian product.
Presets control which values are active: `quick` (2 combos), `standard` (20),
`full` (60).

### 4. Index Cache — `combo/cache.py`

`IndexCacheManager` ensures that identical `(dense, sparse)` pairs share the same
Qdrant index on disk. Reranker and contextual decorators reuse the base index, so
a 60-combo run may require only 10 distinct Qdrant index builds.
Heavy models (ColBERT, FlashRank ranker) are also singleton-loaded once and shared.

### 5. 2-Pass Benchmark Execution — `runner.py`

```
Pass 1 — Latency measurement (BenchmarkRunner.run())
  ↓  optional: --top_n N to select fastest strategies
Pass 2 — RAGAS evaluation (BenchmarkRunner.evaluate())
  ↓  inject_results() reuses Pass 1 retrieval results to avoid re-querying
```

Both passes support parallelism via `ThreadPoolExecutor`:
- `parallel_strategies`: run multiple strategies concurrently (Pass 1)
- `parallel_queries`: run queries concurrently within a strategy (Pass 1)
- `parallel_eval`: run RAGAS evaluation concurrently across strategies (Pass 2)

### 6. LangGraph Agentic RAG — `graph/`

A compiled `StateGraph` wraps any `BaseRAGStrategy` for multi-turn conversational use:

```
START → summarize → analyze_rewrite → [process_question | human_input]
                                             ↓
                                          aggregate → END
```

`process_question` is an agent sub-graph (ReAct loop) that binds the strategy's
retriever as a LangChain tool. `InMemorySaver` provides per-thread conversation state.

### 7. Data Flow

```
docs/*.pdf
  └─ pdfs_to_markdowns()              (pdf_converter.py)
       └─ create_parent_child_chunks()  (chunker.py)
            Parent chunks  → contextual LLM prefix generation (optional)
            Child chunks   → Qdrant index (DenseSparseStrategy)
                               ↕ retrieval
            qa_dataset.json  ← RAGAS KG QA generator (generate_qa.py)
                               ↓
                          BenchmarkRunner.run()   [Pass 1]
                               ↓
                          BenchmarkRunner.evaluate() [Pass 2]
                               ↓
                          EvaluationReport / rank_strategies()
                               ↓
                          e2e_report.md / all_combos_ragas.csv
```

---

## External Integration Points

| System | Integration | Location |
|---|---|---|
| OpenAI API | ChatOpenAI (answer gen, eval, contextual LLM) | runner.py, evaluator.py, contextual_retrieval.py |
| OpenAI Embeddings | OpenAIEmbeddings via langchain-openai | dense_sparse.py |
| Upstage Solar | UpstageEmbeddings via langchain-upstage | dense_sparse.py, upstage_embed.py |
| Qdrant | QdrantClient (local disk or Docker remote) | dense_sparse.py |
| HuggingFace Hub | HuggingFaceEmbeddings, AutoModel | dense_sparse.py, colbert.py |
| RAGAS | evaluate(), EvaluationDataset, KnowledgeGraph | evaluator.py, generate_qa.py |
| LangGraph | StateGraph, ToolNode, InMemorySaver | graph/builder.py |
| Google Drive | Colab checkpoint/result persistence | rag_bench_colab/ |

---

## Key Data Artifacts (Runtime, not committed)

| Path | Content |
|---|---|
| `rag_bench/_benchdata/qdrant_db_{dense}_{sparse}/` | Per-(dense,sparse) Qdrant index |
| `rag_bench/_benchdata/qdrant_db_ctx_{dense}_{sparse}/` | Contextual-enriched Qdrant index |
| `rag_bench/_benchdata/qa_dataset.json` | Generated QA pairs |
| `rag_bench/_benchdata/contextual_cache.json` | LLM-generated contextual prefixes |
| `rag_bench/_benchdata/parent_store/parents.json` | Serialised parent chunks |
| `rag_bench/_benchdata/all_combos_latency.csv` | Pass 1 per-query latency results |
| `rag_bench/_benchdata/all_combos_ragas.csv` | Pass 2 RAGAS metric scores |
| `rag_bench/_benchdata/combo_timing.csv` | Per-combo build/pass1/pass2 timing |
| `rag_bench/_benchdata/run_history/run_YYYYMMDD_HHMMSS.json` | Full RunTracker record |
| `rag_bench/_benchdata/e2e_report.md` | Markdown benchmark report |
