# AutoRAG Benchmark — Architecture

## 1. System Overview

AutoRAG Benchmark is a Korean-first RAG evaluation system that measures retrieval quality across 5 document categories (GENERAL, LEGAL, BUSINESS, MEDICAL, TECHNICAL) using a 4-layer combinatorial strategy space. The system supports two execution environments: local sequential runs and a 2-phase Kubernetes parallel batch.

```
┌─────────────────────────────────────────────────────────────────┐
│  Execution Environments                                         │
│                                                                 │
│  Local (run_service_bench.py)   K8s (orchestrator.py)           │
│  ─────────────────────────────  ───────────────────────────     │
│  Sequential categories          Phase1 Prep Jobs (×4 parallel)  │
│  Checkpoint resume              Phase2 Bench Jobs (×24 parallel) │
│  ColBERT singleton shared       Direct HF embedding in Pod       │
└──────────────────────┬──────────────────────────────────────────┘
                       │ BenchmarkRunner.run() + evaluate()
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Strategy Layer (Decorator Chain)                               │
│                                                                 │
│  FlashRankRerankStrategy ──┐                                    │
│  ColBERTRerankStrategy ────┼── wraps ──▶ ContextualRetrieval    │
│                            │            Strategy ──▶ DenseSparse│
│                            │                         Strategy   │
│  (Layer 3: Reranker)       │  (Layer 4: LLM Support) (L1+L2)   │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Infrastructure Layer                                           │
│  Qdrant (file-mode)   KoreanBM25 / SpladeEncoder               │
│  HuggingFaceEmbeddings / OpenAIEmbeddings                       │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Evaluation Layer                                               │
│  RAGAS v0.4+ (core_only preset, OpenAI gpt-4o-mini fixed)       │
│  Metrics: faithfulness, answer_relevancy, context_precision,    │
│           llm_context_recall                                    │
│  Scoring Profiles: balanced / precision_critical / speed /      │
│                    comprehensive                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
autorag/
├── rag_bench/                   # Core Python package (installable via pyproject.toml)
│   ├── __init__.py              # Exports: BaseRAGStrategy, BenchmarkRunner
│   ├── base.py                  # Abstract base: BaseRAGStrategy + StrategyRetriever
│   ├── config.py                # All global constants, make_llm(), SSL bypass, model cache
│   ├── runner.py                # BenchmarkRunner: Pass1 latency + Pass2 RAGAS
│   ├── run_tracker.py           # RunTracker: execution history, token tracking
│   ├── cli.py                   # CLI entry point
│   │
│   ├── strategies/              # Concrete strategy implementations
│   │   ├── dense_sparse.py      # DenseSparseStrategy + KoreanBM25Encoder + SpladeEncoder
│   │   ├── contextual_retrieval.py  # ContextualRetrievalStrategy (Layer 4 decorator)
│   │   ├── colbert_rerank.py    # ColBERTRerankStrategy (Layer 3 decorator)
│   │   ├── flashrank_rerank.py  # FlashRankRerankStrategy (Layer 3 decorator)
│   │   ├── colbert.py           # Standalone ColBERT strategy (not used in combos)
│   │   ├── openai_embed.py      # OpenAI embedding wrapper
│   │   └── upstage_embed.py     # Upstage Solar embedding wrapper
│   │
│   ├── combo/                   # Combination management
│   │   ├── __init__.py          # Public API: ComboSpec, PRESETS, IndexCacheManager, etc.
│   │   ├── spec.py              # ComboSpec dataclass + PRESETS dict + generate_valid_combinations()
│   │   ├── cache.py             # CacheConfig + IndexCacheManager (index reuse + model sharing)
│   │   └── builder.py           # build_strategy_from_spec() — 4-layer assembly
│   │
│   ├── evaluation/              # RAGAS evaluation subsystem
│   │   ├── __init__.py          # Public API: ExtendedRAGEvaluator, EvaluationReport, etc.
│   │   ├── evaluator.py         # ExtendedRAGEvaluator + EvaluationReport + SCORING_PROFILES
│   │   └── metrics.py           # MetricPreset enum + METRIC_REGISTRY + create_metrics()
│   │
│   ├── datasets/                # HuggingFace dataset loading
│   │   └── hf_loader.py         # HFDatasetLoader + BeirDataset + beir_to_parent_child_chunks()
│   │
│   ├── indexing/                # Document ingestion
│   │   ├── chunker.py           # create_parent_child_chunks() (Markdown-file based)
│   │   ├── multi_parser.py      # parse_directory() for user docs
│   │   └── pdf_converter.py     # PDF → Markdown conversion
│   │
│   ├── document_types/          # 5-category taxonomy
│   │   ├── types.py             # DocType enum + DOC_TYPE_METADATA
│   │   ├── classifier.py        # classify_document()
│   │   └── sampler.py           # sample_text() per DocType
│   │
│   ├── analysis/                # Post-benchmark analysis pipeline
│   │   ├── pipeline.py          # run_analysis_pipeline() → AnalysisResult
│   │   ├── ranker.py            # load_results() + rank_by_doc_type()
│   │   ├── insight.py           # analyze_strengths_weaknesses()
│   │   ├── deduplication.py     # compress_similar_results() (tie detection)
│   │   ├── selector.py          # generate_selection_report() → SelectionReport
│   │   ├── reporter.py          # Technical report renderer
│   │   └── reporter_exec.py     # Executive report renderer
│   │
│   ├── graph/                   # LangGraph RAG agent (experimental)
│   │   ├── builder.py           # Graph construction
│   │   ├── nodes.py             # Node functions
│   │   ├── state.py             # TypedDict state
│   │   └── prompts.py           # LangChain prompts
│   │
│   ├── scripts/                 # Runnable scripts (python -m rag_bench.scripts.*)
│   │   ├── run_service_bench.py # Main local orchestrator (--mode hf | docs)
│   │   ├── run_all_combos.py    # All-combo runner (legacy)
│   │   ├── run_bench.py         # Single-combo runner
│   │   ├── generate_qa.py       # RAGAS QA generation
│   │   ├── merge_service_results.py  # Result merger → HTML report
│   │   ├── generate_html_report.py   # HTML report generator
│   │   ├── prefetch_models.py   # Pre-download HF models
│   │   └── timing_report.py     # Latency report
│   │
│   ├── utils/
│   │   ├── device.py            # detect_device() — CUDA → CPU, MPS excluded
│   │   ├── qa_loader.py         # QA pair loading
│   │   └── report.py            # print_ragas_table()
│   │
│   └── _benchdata/              # Runtime artifacts (gitignored)
│       ├── qdrant_db_*/         # Per-(dense,sparse) Qdrant indexes
│       ├── contextual_cache.json
│       ├── parent_store/
│       └── *.csv / *.html       # Result files
│
├── k8s/                         # Kubernetes deployment subsystem
│   ├── orchestrator.py          # 2-phase K8s orchestrator (CLI)
│   ├── worker_entrypoint.py     # Worker: phase_prep() + phase_bench()
│   ├── Dockerfile               # Multi-stage, CPU-only torch, venv copy
│   ├── requirements-worker.txt  # Worker-only dependency set
│   └── manifests/
│       ├── namespace.yaml
│       ├── results-pvc.yaml     # ReadWriteMany PVC for results
│       ├── model-cache-pvc.yaml # ReadWriteMany PVC for HF model cache
│       ├── prep-job-template.yaml
│       └── bench-job-template.yaml
│
├── rag_bench_local/             # Local notebook runner
│   ├── local_runner.py
│   ├── local_config.py
│   └── visualizer.py
│
├── rag_bench_colab/             # Google Colab runner
│   ├── colab_runner.py
│   ├── colab_config.py
│   └── colab_visualizer.py
│
├── pdf_parser/                  # Standalone PDF→Markdown module
│   ├── smart_router.py          # Complexity-based routing
│   ├── category1_simple.py      # Simple PDF parser
│   ├── category2_medium.py      # Medium complexity parser
│   ├── category3_complex.py     # Complex/mixed parser
│   ├── hybrid_backend.py        # Backend abstraction
│   └── quality_checker.py       # Parse quality assessment
│
├── scripts/                     # Dev/debug scripts
│   ├── verify_env.py
│   ├── verify_rag_bench.py
│   ├── verify_ragas_eval.py
│   ├── verify_graphrag.py
│   └── cleanup_legacy_indexes.py
│
├── docs/                        # Research and design documents
│   └── research/                # Study notes per topic
│
├── pyproject.toml               # Package config (uv, hatch)
└── docker-compose.yml           # Local service stack
```

---

## 3. Architectural Patterns

### 3.1 Strategy + Decorator Pattern

The retrieval layer uses a strict layered composition model:

```
Layer 1 (Dense)    + Layer 2 (Sparse)
     └──── DenseSparseStrategy (BaseRAGStrategy)
                 ↓ wrapped by
Layer 4 (LLM Support)
     └──── ContextualRetrievalStrategy(base_strategy=DenseSparse)
                 ↓ wrapped by
Layer 3 (Reranker)
     └──── ColBERTRerankStrategy(base_strategy=Contextual)
        or FlashRankRerankStrategy(base_strategy=Contextual)
```

- All strategies implement `BaseRAGStrategy` (abstract): `index()`, `retrieve()`, `get_retriever()`
- Decorators delegate `index()` upward to the wrapped strategy
- `retrieve()` in rerankers calls `base_strategy.retrieve(rerank_n)` then re-scores the candidates
- `StrategyRetriever` (Pydantic `BaseRetriever`) wraps any `BaseRAGStrategy` for LangChain compatibility
- Every strategy constructor's default for `get_retriever()` returns a `StrategyRetriever`

### 3.2 4-Layer Combination Space

```
ComboSpec:
  dense       ∈ {kosimcse, e5, bge-m3, openai-large, upstage}       (Layer 1)
  sparse      ∈ {korean_bm25, splade}                                (Layer 2)
  reranker    ∈ {colbert, flashrank}                                  (Layer 3)
  llm_support ∈ {contextual}                                         (Layer 4)

Presets:
  quick    →  1 combo   (bge-m3 × korean_bm25 × flashrank × contextual)
  service  →  6 combos  (HF 3 × sparse 2 × colbert × contextual)
  standard → 20 combos  (all 5 × sparse 2 × flashrank × contextual)
  full     → 40 combos  (all 5 × sparse 2 × [colbert,flashrank] × contextual)
```

`ComboSpec.__post_init__` enforces all 4 fields non-empty — lone Dense or Sparse strategies are not allowed by design.

### 3.3 IndexCacheManager — Index Reuse

`IndexCacheManager` provides two caches keyed on `(dense, sparse)`:

- `cache[dense:sparse]` → base `DenseSparseStrategy` + Qdrant path
- `ctx_cache[ctx:dense:sparse]` → `ContextualRetrievalStrategy`

When multiple combos share the same Dense+Sparse pair (e.g., `bge-m3+korean_bm25+colbert` and `bge-m3+korean_bm25+flashrank`), the second combo reuses the Qdrant index and the already-loaded embedding model objects via `share_embeddings()`.

ColBERT and FlashRank models are also singletons inside `IndexCacheManager` protected by a `threading.Lock` for multi-threaded reranking.

BM25 vocabulary is persisted to `{qdrant_path}_bm25_vocab.json` to survive process restart without re-fitting.

### 3.4 2-Phase K8s Benchmark

```
Orchestrator (local machine)
│
├── setup_infrastructure()          # namespace, PVCs, bench-secrets
│
├── Phase 1: Prep Jobs × len(categories)  [parallel]
│   Each Job (worker_entrypoint.py --phase prep):
│     HFDatasetLoader.load() → beir_to_parent_child_chunks()
│     ContextualRetrievalStrategy.enrich_only()  [LLM calls]
│     Serialize to /results/<category>/prepared/{child_chunks,
│       parent_pairs, qa_pairs, enriched_chunks, DONE}.json
│
├── _verify_prep_data()             # busybox Pod: assert DONE files visible
│
└── Phase 2: Bench Jobs × (categories × combos)  [parallel, up to 24]
    Each Job (worker_entrypoint.py --phase bench):
      Deserialize Phase 1 data from PVC
      build_strategy_from_spec()   # builds the 4-layer stack
      BenchmarkRunner.run()        # Pass 1: latency
      BenchmarkRunner.evaluate()   # Pass 2: RAGAS
      Serialize results to /results/<category>/<combo_label>/
        {latency.csv, ragas.csv, result.json, DONE}
```

Atomic writes use `.tmp → rename` to prevent partial-read by the collector.
DONE files are the sole completion signal.

### 3.5 Contextual Retrieval

Implements Anthropic's Contextual Retrieval paper:
1. During `index()`, for each child chunk, the LLM generates a short context prefix using the parent chunk as the surrounding document
2. `enriched_content = prefix + "\n\n" + original_chunk`
3. The enriched text is indexed; original text is preserved in `metadata["original_content"]`
4. During `retrieve()`, results are restored to original content before being returned to the caller
5. Context generation is cached per-chunk (SHA-256 key) to `contextual_cache.json`

---

## 4. Key Data Flows

### 4.1 Local Benchmark Flow

```
run_service_bench.py main()
  ├── parse_args() → preset="service"
  ├── generate_valid_combinations(PRESETS["service"]) → 6 ComboSpecs
  ├── For each DocType:
  │     _prepare_hf_data() or _prepare_docs_data()
  │       └── HFDatasetLoader.load() → BeirDataset
  │           beir_to_parent_child_chunks() → (parent_pairs, child_chunks)
  │     _run_category_bench()
  │       ├── ContextualRetrievalStrategy.enrich_only()  [once per category]
  │       ├── For each ComboSpec:
  │       │     build_strategy_from_spec() → decorated strategy stack
  │       ├── BenchmarkRunner.run() → _results dict
  │       └── BenchmarkRunner.evaluate() → ragas_df
  └── _save_category_result() → result.json, ragas.csv
```

### 4.2 RAGAS Evaluation Flow

```
BenchmarkRunner.evaluate()
  ├── For each strategy in parallel (parallel_eval):
  │     Generate answers via LLM (parallel_workers=8)
  │     ExtendedRAGEvaluator.evaluate(questions, contexts, answers, ground_truths)
  │       └── ragas.evaluate(EvaluationDataset, metrics=[Faithfulness,...])
  │           RunConfig(max_workers=16, timeout=180)
  │     EvaluationReport(per_sample_df, aggregate_dict)
  └── Return pd.DataFrame of aggregate scores per strategy
```

### 4.3 Analysis Pipeline Flow

```
run_analysis_pipeline(run_dir)
  1. load_results() → {category: raw_result_dict}
  2. rank_by_doc_type() → {category: ranked_df}
  3. analyze_strengths_weaknesses() → {combo_label: insight_dict}
  4. compress_similar_results() → deduplicated ranked_df (5% threshold)
  5. generate_selection_report() → SelectionReport
  → AnalysisResult
```

---

## 5. External Integrations

| Integration | Usage | File |
|---|---|---|
| Qdrant (file-mode) | Vector store, hybrid retrieval | `strategies/dense_sparse.py` |
| HuggingFace Hub | Model download + Dataset streaming | `datasets/hf_loader.py`, `config.py` |
| OpenAI API | Embeddings (text-embedding-3-large), RAGAS eval LLM (gpt-4o-mini) | `strategies/dense_sparse.py`, `evaluation/evaluator.py` |
| Upstage API | Solar embeddings (embedding-query) | `strategies/upstage_embed.py` |
| Ollama | Local LLM for contextual enrichment (default) | `config.make_llm()` |
| Harbor Registry | Container image push | `k8s/Dockerfile` |
| EKS (ap-northeast-2) | Job execution cluster | `k8s/orchestrator.py` |
| KoNLPy (OKt) | Korean morphological tokenization for BM25 | `strategies/dense_sparse.py` |
| pylate | ColBERT model + MaxSim reranking | `strategies/colbert_rerank.py` |
| flashrank | ONNX lightweight reranker | `strategies/flashrank_rerank.py` |
| RAGAS v0.4+ | RAG evaluation framework | `evaluation/evaluator.py` |

---

## 6. Module Dependency Graph

```
rag_bench (top-level)
├── config.py                     # no internal imports
├── base.py                       # ← langchain_core
│
├── strategies/
│   ├── dense_sparse.py           # ← base, config, utils/device
│   ├── contextual_retrieval.py   # ← base, config (make_llm)
│   ├── colbert_rerank.py         # ← base, utils/device
│   └── flashrank_rerank.py       # ← base
│
├── combo/
│   ├── spec.py                   # ← strategies/dense_sparse (DENSE_MODELS, SPARSE_TYPES)
│   ├── cache.py                  # ← config, combo/spec, strategies/*
│   └── builder.py                # ← combo/spec, combo/cache, strategies/*
│
├── datasets/hf_loader.py         # ← document_types/types
├── indexing/chunker.py           # ← langchain_text_splitters
│
├── evaluation/
│   ├── metrics.py                # ← ragas.metrics (dynamic getattr)
│   └── evaluator.py             # ← config, evaluation/metrics, ragas, langchain_openai
│
├── runner.py                     # ← base, config, evaluation/evaluator
│
└── scripts/run_service_bench.py  # ← config, combo/*, runner, datasets, utils
    k8s/worker_entrypoint.py      # ← rag_bench.combo, rag_bench.runner, rag_bench.evaluation
    k8s/orchestrator.py           # no rag_bench imports (kubectl subprocess only)
```

Notable coupling points:
- `combo/spec.py` imports `DENSE_MODELS` and `SPARSE_TYPES` from `strategies/dense_sparse.py` — the registry lives in the strategy module, not spec
- `config.py` runs `load_dotenv()` and `ensure_model_cache()` as side effects on import
- `run_service_bench.py` manually clones `IndexCacheManager` per category while sharing ColBERT/FlashRank models across categories via `_colbert_model` attribute access

---

## 7. Configuration and Environment Variables

### Core (`config.py`)
| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen3:4b-instruct-2507-q4_K_M` | Base Ollama model |
| `OLLAMA_CONTEXTUAL_MODEL` | same | Contextual enrichment LLM |
| `OPENAI_API_KEY` | — | OpenAI API |

### K8s Worker (`worker_entrypoint.py`)
| Variable | Purpose |
|---|---|
| `WORKER_PHASE` | `prep` or `bench` |
| `BENCH_CATEGORY` | Category name |
| `COMBO_DENSE` / `COMBO_SPARSE` / `COMBO_RERANKER` / `COMBO_LLM_SUPPORT` | Strategy spec |
| `COMBO_LABEL` | K8s-safe combo label (orchestrator injected) |
| `RESULTS_DIR` | `/results` (PVC mount) |
| `WORKSPACE_DIR` | `/workspace` (emptyDir) |

### Path Overrides
| Variable | Overrides |
|---|---|
| `RAG_BENCH_DATA_DIR` | `BENCH_DATA_DIR` |
| `RAG_BENCH_DOCS_DIR` | `BENCH_DOCS_DIR` |
| `RAG_BENCH_DOCS_SRC` | `DOCS_DIR` |
