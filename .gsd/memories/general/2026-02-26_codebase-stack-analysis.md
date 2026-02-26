# AutoRAG Benchmark — Technology Stack

## Language & Runtime

| Item | Version | Notes |
|---|---|---|
| Python | 3.12 | Required minimum (`requires-python = ">=3.12"`) |
| Package manager | uv | `uv.lock` present |
| Build backend | hatch | `pyproject.toml` |
| Container base | python:3.12-slim | Multi-stage Docker build |

---

## Core Framework

| Library | Version | Role |
|---|---|---|
| `langchain` | ≥1.0 | Core RAG pipeline abstraction |
| `langchain-core` | ≥0.3 | `Document`, `BaseRetriever`, `Embeddings` |
| `langchain-community` | ≥0.3 | OpenAI callback tracker |
| `langchain-huggingface` | ≥0.1 | `HuggingFaceEmbeddings` |
| `langchain-openai` | ≥0.2 | `OpenAIEmbeddings`, `ChatOpenAI` |
| `langchain-ollama` | ≥0.2 | `ChatOllama` |
| `langchain-upstage` | ≥0.3 | `UpstageEmbeddings` |
| `langchain-qdrant` | ≥0.2 | `QdrantVectorStore`, `RetrievalMode.HYBRID` |
| `langchain-text-splitters` | ≥0.3 | `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter` |
| `langgraph` | ≥0.2 | LangGraph agent (graph/ module, experimental) |

---

## Vector Database

| Library | Version | Mode | Notes |
|---|---|---|---|
| `qdrant-client` | ≥1.11 | File-mode (local path) | No separate server. One DB dir per (dense, sparse) pair. |

Qdrant is used exclusively in embedded file mode (`QdrantClient(path=...)`). In-memory mode (`:memory:`) is available for testing. Collections use hybrid vectors: one dense vector field + one sparse vector field named `"sparse"`.

---

## Embedding Models

### Dense (HuggingFace local)

| Key | Model ID | Dimension | Notes |
|---|---|---|---|
| `kosimcse` | `BM-K/KoSimCSE-roberta-multitask` | 768 | Korean SimCSE contrastive |
| `e5` | `intfloat/multilingual-e5-large` | 1024 | 560M, instruction-prefix style |
| `bge-m3` | `BAAI/bge-m3` | 1024 | 100+ langs, MIRACL Korean SOTA |
| `snowflake-ko` | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` | 1024 | Korean retrieval SOTA |

### Dense (External API)

| Key | Model | Notes |
|---|---|---|
| `openai-large` | `text-embedding-3-large` | 3072-dim, OpenAI API |
| `upstage` | `embedding-query` | 4096-dim, Upstage Solar API |

### Sparse

| Type | Implementation | Notes |
|---|---|---|
| `korean_bm25` | `KoreanBM25Encoder` (custom) | KoNLPy OKt morpheme tokenization + BM25 |
| `splade` | `SpladeEncoder` (custom) | `naver/splade-cocondenser-ensembledistil`, term expansion |
| `fastembed_bm25` | `FastEmbedSparse` (langchain-qdrant) | `Qdrant/bm25`, optional |

---

## Rerankers

| Library | Version | Type | Model |
|---|---|---|---|
| `pylate` | ≥1.0 | ColBERT MaxSim | `jinaai/jina-colbert-v2` |
| `flashrank` | ≥0.2 | ONNX cross-encoder | `ms-marco-MultiBERT-L-12` (150MB, multilingual) |

FlashRank runs CPU-only without Torch (ONNX runtime). ColBERT requires Torch and is serialized behind a `threading.Lock` when shared across multiple strategy instances.

---

## Evaluation

| Library | Version | Role |
|---|---|---|
| `ragas` | ≥0.4.3 | RAG evaluation framework |

**RAGAS metrics used (core_only preset):**

| Metric | Requires Reference | Notes |
|---|---|---|
| `Faithfulness` | No | LLM-based |
| `AnswerRelevancy` | No | LLM-based, multi-perspective reverse questions |
| `ContextPrecision` | Yes | LLM-based |
| `LLMContextRecall` | Yes | LLM-based |

The evaluation LLM is hardcoded to `gpt-4o-mini` (OpenAI) regardless of `LLM_PROVIDER` for reproducibility. A custom `_MultiPerspectiveLLM` wrapper produces n>1 reverse questions in a single structured OpenAI call using `ChatOpenAI.with_structured_output`.

**Scoring profiles:**

| Profile | Weights |
|---|---|
| `balanced` | all 4 metrics × 0.25 |
| `precision_critical` | faithfulness 0.4, context_precision 0.3, answer_relevancy 0.3 |
| `speed_critical` | answer_relevancy 0.5, faithfulness 0.5 |
| `comprehensive` | core 4 + factual_correctness + context_entity_recall + response_relevancy |

---

## NLP / ML

| Library | Version | Role |
|---|---|---|
| `konlpy` | ≥0.6.0 | Korean NLP — OKt tokenizer for BM25 |
| `sentence-transformers` | ≥3.1 | Dense embedding inference |
| `transformers` | ≥4.40 | SPLADE model (AutoModelForMaskedLM) |
| `torch` | ≥2.2 (CPU-only in K8s) | SPLADE + ColBERT inference |
| `fastembed` | ≥0.4 | Optional BM25 via ONNX |
| `einops` | ≥0.8.2 | Tensor operations (ColBERT dependency) |

MPS (Apple Silicon GPU) is explicitly excluded from `detect_device()` due to OOM risk with SPLADE and ColBERT. CUDA is used when available; otherwise CPU.

---

## Datasets (HuggingFace)

| Category | Primary Dataset | Subset / Split |
|---|---|---|
| GENERAL | `klue/klue` (mrc) | streaming train |
| LEGAL | `yjoonjang/markers_bm` | corpus=law |
| BUSINESS | `yjoonjang/markers_bm` | corpus=finance+public+commerce |
| MEDICAL | `xhluca/publichealth-qa` | korean |
| TECHNICAL | `sionic-ai/nanobeir-ko` | NanoSCIDOCS |

Datasets are loaded in BeIR format (corpus, queries, qrels) and cached to JSON on first load.

---

## Document Parsing

| Library | Version | Role |
|---|---|---|
| `pymupdf4llm` | ≥0.0.17 | PDF → Markdown (core path) |
| `python-docx` | ≥1.1 | DOCX parsing |
| `beautifulsoup4` | ≥4.12 | HTML parsing |
| `lxml` | ≥5.0 | XML/HTML backend |

The `pdf_parser/` module provides a standalone complexity-routing PDF parser (`smart_router.py`) with 3 category backends (simple/medium/complex) and quality checking, decoupled from the main `rag_bench` package.

---

## Data / Analysis

| Library | Version | Role |
|---|---|---|
| `pandas` | ≥2.2 | Results DataFrames, CSV output |
| `matplotlib` | ≥3.10.8 | Charts (analysis reports) |
| `rapidfuzz` | ≥3.14.3 | Fuzzy string similarity (tie detection) |

---

## Infrastructure & Networking

| Tool / Library | Role |
|---|---|
| `httpx` | HTTP client for OpenAI/Upstage API (SSL bypass support) |
| `python-dotenv` | `.env` loading |
| `psutil` | Memory/CPU info in RunTracker |
| `kubectl` | Subprocess wrapper in orchestrator.py |
| Harbor Registry | OCI container registry for worker images |
| EKS (ap-northeast-2) | AWS Kubernetes cluster |
| K8s PVC (ReadWriteMany) | Shared NFS/CephFS for results and model cache |
| TEI (text-embeddings-inference) | HuggingFace embedding server (optional) |

---

## Developer Tooling

| Tool | Config | Role |
|---|---|---|
| `ruff` | `.ruff_cache/` | Linting |
| `mypy` | `.mypy_cache/` | Type checking |
| `jupyter` | ≥1.0 | Notebook runner (local/colab) |
| `ipython` | ≥8.27 | Interactive shell |

---

## K8s Worker Image

The Dockerfile uses a multi-stage build:

1. **Stage 1 (builder):** `python:3.12-slim` + `g++`, `git` → installs CPU-only PyTorch from `https://download.pytorch.org/whl/cpu`, then all worker deps, then force-reinstalls CPU torch to remove any GPU variant pulled by transitive deps, then installs `rag_bench` package.
2. **Stage 2 (runtime):** `python:3.12-slim` + `default-jre-headless` (KoNLPy/Java) + copied venv → `worker_entrypoint.py`.

Build target: `linux/amd64`. Runtime savings: ~2.5 GB vs CUDA torch bundle.

---

## Technical Debt Inventory

The following issues were identified during codebase analysis. None are critical blockers, but they represent maintenance and reliability risks.

### TD-1: SSL verification globally disabled (High Risk)
**File:** `/Users/sukbeom/Desktop/autorag/rag_bench/config.py` — `setup_ssl_bypass()`
The function monkeypatches `requests.Session.request` to force `verify=False` on every HTTP request for the entire process lifetime. This cannot be undone. It also patches `ssl._create_default_https_context` at the global level. While intended for corporate proxy environments, it silently disables MITM protection system-wide.
**Recommendation:** Use per-client `verify=False` passed to `httpx.Client(verify=False)` (already done for OpenAI), and restrict the requests monkeypatch to HuggingFace downloads only.

### TD-2: `IndexCacheManager` internal state accessed directly across module boundaries
**File:** `/Users/sukbeom/Desktop/autorag/rag_bench/scripts/run_service_bench.py` (lines ~317-320)
`cat_cache._colbert_model = index_cache._colbert_model` and `cat_cache._flashrank_ranker = index_cache._flashrank_ranker` access private attributes of `IndexCacheManager` to share singleton models across category caches. This bypasses the public `get_colbert_model()` / `get_flashrank_ranker()` API.
**Recommendation:** Add `share_models_from(other: IndexCacheManager)` method.

### TD-3: K8s Job name truncation silently corrupts label uniqueness
**File:** `/Users/sukbeom/Desktop/autorag/k8s/orchestrator.py` (lines ~236-237)
```python
if len(job_name) > 63:
    job_name = job_name[:63].rstrip("-")
```
Truncating a job name can produce duplicate names for different (category, combo) pairs. The orchestrator does not check for collisions.
**Recommendation:** Use a stable hash suffix: `f"bench-{hash(key)[:8]}"` when truncation is needed.

### TD-4: No test suite
No `tests/` directory exists. No pytest, unittest, or any testing infrastructure is present. The `scripts/verify_*.py` files are manual smoke tests, not automated assertions.
**Recommendation:** Add pytest with at minimum: unit tests for `ComboSpec` validation, `KoreanBM25Encoder`, `build_strategy_from_spec`, and integration tests for the full pipeline with an in-memory Qdrant store.

### TD-5: `COMBO_LABEL` divergence risk between orchestrator and worker
**File:** `/Users/sukbeom/Desktop/autorag/k8s/worker_entrypoint.py` (line ~188)
```python
combo_label = os.environ.get("COMBO_LABEL", _safe_label(...))
```
The worker has a fallback `_safe_label()` that uses `+` separator, while the orchestrator's `_safe_label()` uses `-` separator. If `COMBO_LABEL` is not injected, result directory paths will differ from what the orchestrator expects.
**Recommendation:** Remove the fallback entirely; make `COMBO_LABEL` a required env var and `sys.exit(1)` if absent.

### TD-6: Hardcoded `source_name` cache keys duplicated between worker and run_service_bench
**Files:** `k8s/worker_entrypoint.py` and `rag_bench/scripts/run_service_bench.py`
Both files contain an identical `source_names` dict mapping `DocType` → HF source name string. If a dataset is renamed, both files must be updated in sync.
**Recommendation:** Move the mapping to `document_types/types.py` as part of `DOC_TYPE_METADATA`.

### TD-7: Contextual cache is a single flat JSON file
**File:** `/Users/sukbeom/Desktop/autorag/rag_bench/strategies/contextual_retrieval.py`
`contextual_cache.json` grows unboundedly as more documents are processed. There is no cache invalidation, size limit, or TTL. The entire file is loaded and re-written on every 10-chunk interval.
**Recommendation:** Use SQLite with a `chunk_hash` primary key, or partition the cache by document source.

### TD-8: MPS device excluded with no opt-in mechanism
**File:** `/Users/sukbeom/Desktop/autorag/rag_bench/utils/device.py`
`detect_device()` skips MPS entirely. On Apple Silicon machines, all model inference runs on CPU. Users with M-series Macs cannot benefit from the GPU even when memory is not a concern.
**Recommendation:** Add `FORCE_DEVICE=mps` environment variable override; keep the default as CPU.

### TD-9: `_release_memory()` duplicated in three files
**Files:** `rag_bench/scripts/run_service_bench.py`, `k8s/worker_entrypoint.py`, and potentially others
Identical `gc.collect() + torch.cuda.empty_cache() + torch.mps.empty_cache()` pattern is copy-pasted.
**Recommendation:** Move to `rag_bench/utils/memory.py`.

### TD-10: `rag_bench_local` and `rag_bench_colab` are partially parallel implementations
**Files:** `rag_bench_local/` and `rag_bench_colab/`
Both directories have structurally identical `*_runner.py`, `*_config.py`, `*_visualizer.py` files with environment-specific differences. Changes to benchmark logic must be applied in three places (rag_bench, local, colab).
**Recommendation:** Unify into `rag_bench` with environment adapters; use the `rag_bench_local/README.md` to track migration status.

### TD-11: `graph/` module is experimental and untested
**File:** `/Users/sukbeom/Desktop/autorag/rag_bench/graph/`
The LangGraph agent module (`builder.py`, `nodes.py`, `state.py`, `prompts.py`) exists but is not wired into any combo, benchmark, or test. Its relationship to the Strategy pattern is undefined.
**Recommendation:** Either document it as an experimental extension point or remove it from the main package.

### TD-12: `GENERAL` category has 4 secondary datasets never used in K8s worker
**File:** `/Users/sukbeom/Desktop/autorag/rag_bench/datasets/hf_loader.py`
`HFDatasetLoader.load_secondary()` loads Ko-StrategyQA, Belebele, and MrTiDy as supplement to MIRACL. However, `worker_entrypoint.py` and `run_service_bench.py` only call `loader.load(doc_type)` (primary only). The K8s worker maps GENERAL to `"klue-mrc"` (not miracl-ko), while `run_service_bench.py` maps it to `"miracl-ko"` in its cache-key dict — creating a cache miss mismatch.
**Recommendation:** Align the source name mapping between all callers; decide whether secondary datasets are part of the benchmark scope.
