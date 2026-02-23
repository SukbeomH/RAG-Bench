# STACK.md

## Runtime Environment

| Item | Value |
|---|---|
| Language | Python >= 3.12 |
| Package manager | uv / pip (pyproject.toml, hatch build backend) |
| Python version pin | `.python-version` (3.12) |

---

## Core Dependencies

### Retrieval Framework

| Package | Role |
|---|---|
| `langchain >= 1.0` | Core chain abstractions, Document, BaseRetriever |
| `langchain-core >= 0.3` | Embeddings, Runnable protocol |
| `langchain-community >= 0.3` | get_openai_callback (token tracking) |
| `langchain-text-splitters >= 0.3` | MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter |
| `langchain-qdrant >= 0.2` | QdrantVectorStore, RetrievalMode.HYBRID, FastEmbedSparse |
| `qdrant-client >= 1.11` | Qdrant disk-based and in-memory vector store client |
| `langgraph >= 0.2` | Agentic RAG pipeline (StateGraph, ToolNode, InMemorySaver) |

### Embedding Models

| Package | Role |
|---|---|
| `langchain-huggingface >= 0.1` | HuggingFaceEmbeddings (kosimcse, e5, bge-m3) |
| `langchain-openai >= 0.2` | OpenAIEmbeddings (text-embedding-3-large), ChatOpenAI |
| `langchain-upstage >= 0.3` | UpstageEmbeddings (Solar embedding-query) |
| `sentence-transformers >= 3.1` | Sentence embedding backbone |
| `transformers >= 4.40` | AutoModelForMaskedLM (SPLADE), AutoTokenizer |
| `torch >= 2.2` | Inference backend for SPLADE, ColBERT |
| `fastembed >= 0.4` | FastEmbedSparse (optional BM25 sparse encoder) |
| `einops >= 0.8.2` | Tensor manipulation (ColBERT dependency) |

### Retrieval / Reranking Models

| Package | Role |
|---|---|
| `pylate >= 1.0` | ColBERT model loading and MaxSim reranking (`models.ColBERT`, `rank.rerank`) |
| `flashrank >= 0.2` | ONNX-based lightweight reranker (CPU, no Torch) |
| `konlpy >= 0.6.0` | KoNLPy Okt — Korean morphological analysis for BM25 |

### Evaluation

| Package | Role |
|---|---|
| `ragas >= 0.4.3` | RAGAS evaluation framework — Faithfulness, AnswerRelevancy, ContextPrecision, LLMContextRecall, FactualCorrectness, and more |
| `pandas >= 2.2` | Per-sample evaluation DataFrames, CSV persistence |
| `rapidfuzz >= 3.14.3` | String similarity metric (NonLLMStringSimilarity) |

### Document Processing

| Package | Role |
|---|---|
| `pymupdf4llm >= 0.0.17` | PDF → LLM-optimised Markdown conversion |

### Utilities

| Package | Role |
|---|---|
| `python-dotenv >= 1.0` | `.env` file loading for API keys |
| `psutil >= 5.9` | Platform memory info collection |
| `matplotlib >= 3.10.8` | Visualisation in notebooks |
| `jupyter >= 1.0` | Interactive notebook support |
| `ipython >= 8.27` | Jupyter kernel and display utilities |

---

## Embedding Model Registry

| Key | Model ID | Source | Dim |
|---|---|---|---|
| `kosimcse` | BM-K/KoSimCSE-roberta-multitask | HuggingFace | 768 |
| `e5` | intfloat/multilingual-e5-large | HuggingFace | 1024 |
| `bge-m3` | BAAI/bge-m3 | HuggingFace | 1024 |
| `openai-large` | text-embedding-3-large | OpenAI API | 3072 |
| `upstage` | embedding-query | Upstage API | 4096 |

## Sparse Encoder Registry

| Key | Implementation | Notes |
|---|---|---|
| `korean_bm25` | `KoreanBM25Encoder` (custom) | Okt morphological analysis + BM25 TF-IDF |
| `splade` | `SpladeEncoder` (custom) | naver/splade-cocondenser-ensembledistil + log1p(relu(logits)) term expansion |
| `fastembed_bm25` | `FastEmbedSparse` (langchain_qdrant) | Qdrant/bm25 ONNX model |

## Reranker Registry

| Key | Implementation | Backend |
|---|---|---|
| `colbert` | `ColBERTRerankStrategy` | jinaai/jina-colbert-v2 via pylate; PyTorch; thread-safe lock |
| `flashrank` | `FlashRankRerankStrategy` | ms-marco-MultiBERT-L-12 via flashrank; ONNX; CPU-only |

---

## Infrastructure

| Component | Implementation |
|---|---|
| Vector database | Qdrant (local disk path or Docker container on port 6333) |
| Checkpoint/persistence | JSON files in `_benchdata/` and `run_history/` |
| Google Colab persistence | Google Drive mount via `rag_bench_colab/` adapter |
| Model cache | Project-local `_models/hub/` with symlinks to `~/.cache/huggingface/hub` |

---

## RAGAS Metric Tiers

### Core (always enabled in default preset)
- Faithfulness
- AnswerRelevancy
- ContextPrecision *(requires ground truth)*
- LLMContextRecall *(requires ground truth)*

### Extended (enabled in `comprehensive` / `full` presets)
- FactualCorrectness, AnswerCorrectness, NoiseSensitivity *(require ground truth)*
- ContextEntityRecall, ResponseRelevancy

### Lightweight (no LLM required)
- SemanticSimilarity, BleuScore, RougeScore, ExactMatch, StringPresence, NonLLMStringSimilarity

---

## CLI Entry Points

```bash
# Main benchmark (all 60 combos, full preset)
python -m rag_bench.scripts.run_all_combos --preset full

# Quick preset, latency only
python -m rag_bench.scripts.run_all_combos --preset quick --pass1-only

# Dry run: list all combinations without executing
python -m rag_bench.scripts.run_all_combos --preset full --dry-run

# Generate QA dataset from PDFs
python -m rag_bench.scripts.generate_qa

# Pre-download HuggingFace models
python -m rag_bench.scripts.prefetch_models

# Generate HTML report from existing results
python -m rag_bench.scripts.generate_html_report

# Verify environment
python scripts/verify_env.py
python scripts/verify_rag_bench.py
```

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI LLM and embedding API |
| `UPSTAGE_API_KEY` | Upstage Solar embedding API |
| `RAG_BENCH_DATA_DIR` | Override `_benchdata/` path (Colab use) |
| `RAG_BENCH_DOCS_DIR` | Override benchmark docs markdown dir |
| `RAG_BENCH_DOCS_SRC` | Override source PDF docs directory |
| `RAG_BENCH_PARALLEL` | Default parallel query workers |
| `RAG_BENCH_PARALLEL_STRATEGIES` | Default parallel strategy workers |
| `RAG_BENCH_PARALLEL_EVAL` | Default parallel eval workers |
| `HF_HOME` | Set automatically to `_models/` for local model cache |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | Set to `0.0` to prevent MPS OOM |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Set to `1` for MPS → CPU fallback |
