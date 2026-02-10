# AutoRAG 리서치 문서

## 1. AutoRAG 개요

AutoRAG는 **RAG AutoML 도구**로, 주어진 데이터에 대해 최적의 RAG 파이프라인을 자동으로 탐색하고 평가하는 프레임워크이다.

### 핵심 기능
- **Data Creation**: 원시 문서로부터 평가 데이터셋 생성
- **Optimization**: 자동화된 실험으로 최적 RAG 파이프라인 탐색
- **Deployment**: YAML 설정으로 파이프라인 배포 (API, Web, Code)

### 설치
```bash
uv pip install AutoRAG
```

---

## 2. 데이터 포맷

AutoRAG는 **parquet 형식**의 두 가지 데이터셋이 필수이다.

### QA Dataset (qa.parquet)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `qid` | string | 질문 고유 식별자 (중복 불가) |
| `query` | string | 사용자 질문 |
| `retrieval_gt` | 2D list/1D list/string | 검색 정답 문서 ID |
| `generation_gt` | list/string | 생성 정답 (답변) |

### Corpus Dataset (corpus.parquet)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `doc_id` | string | 문서 고유 식별자 (중복 불가) |
| `contents` | string | 청킹된 문서 내용 |
| `metadata` | dict | `last_modified_datetime` 필수 포함 |

---

## 3. 데이터 생성 파이프라인 (4단계)

### Stage 1: Parsing (문서 파싱)
```yaml
modules:
  - module_type: langchain_parse
    parse_method: pdfminer
```
```python
from autorag.parser import Parser
parser = Parser(data_path_glob="docs/*.pdf", project_dir="./parse_project_dir")
parser.start_parsing("parse_config.yaml")
```

### Stage 2: Chunking (문서 청킹)
```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: Token
    chunk_size: 1024
    chunk_overlap: 24
    add_file_name: en
```
```python
from autorag.chunker import Chunker
chunker = Chunker.from_parquet(
    parsed_data_path="parsed_data.parquet",
    project_dir="./chunk_project_dir"
)
chunker.start_chunking("chunk_config.yaml")
```

### Stage 3: QA 생성 (LLM 기반)
```python
from autorag.data.qa.schema import Raw, Corpus
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.generation_gt.llama_index_gen_gt import make_basic_gen_gt, make_concise_gen_gt
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.sample import random_single_hop

llm = OpenAI()
raw_df = pd.read_parquet("parsed.parquet")
raw_instance = Raw(raw_df)
corpus_df = pd.read_parquet("corpus.parquet")
corpus_instance = Corpus(corpus_df, raw_instance)

initial_qa = (
    corpus_instance.sample(random_single_hop, n=3)
    .map(lambda df: df.reset_index(drop=True))
    .make_retrieval_gt_contents()
    .batch_apply(factoid_query_gen, llm=llm)
    .batch_apply(make_basic_gen_gt, llm=llm)
    .batch_apply(make_concise_gen_gt, llm=llm)
    .filter(dontknow_filter_rule_based, lang="en")
)
initial_qa.to_parquet('./qa.parquet', './corpus.parquet')
```

### Stage 4: QA-Corpus 리매핑 (다중 청킹 전략 비교 시)
```python
new_qa = qa.update_corpus(Corpus(new_corpus_df, raw))
```

---

## 4. YAML 설정 구조

### 기본 구조
```yaml
node_lines:
  - node_line_name: node_line_1
    nodes:
      - node_type: retrieval
        top_k: 10
        strategy:
          metrics: [retrieval_f1, retrieval_recall]
          speed_threshold: 10
        modules:
          - module_type: bm25
          - module_type: vectordb
            vectordb: default
```

### 핵심 개념
- **node_lines**: 노드 그룹 (순차 처리)
- **nodes**: 처리 단계 (retrieval, reranker, generator 등)
- **strategy**: 평가 메트릭 및 성능 기준
- **modules**: 구체적 구현체 (bm25, vectordb 등)

### 파라미터 타입
- **고정값**: 모든 평가에 적용
- **리스트**: 조합별 순차 테스트 → `[value1, value2]`
- **튜플**: 단일 파라미터 → `(4, 80)`

### 환경 변수 참조
```yaml
top_k: ${TOP_K}
```

---

## 5. 지원 노드 및 모듈

### Query Processing
- Query Decompose, HyDE, Multi Query Expansion

### Retrieval
- **BM25**: 한국어 토크나이저 지원 (`ko_kiwi`, `ko_kkma`, `ko_okt`)
- **VectorDB**: Chroma, Milvus, Weaviate, Pinecone, Qdrant 지원
- **Hybrid RRF**: BM25 + VectorDB 결합 (Reciprocal Rank Fusion)
- **Hybrid CC**: BM25 + VectorDB 결합 (Convex Combination)

### Passage Enhancement
- **Reranker**: FlashRank, MonoT5, Cohere, RankGPT, Jina, ColBERT 등 15+
- **Filter**: Similarity Threshold, Percentile Cutoff, Recency Filter
- **Compressor**: Tree Summarize, Refine, Long LLM Lingua
- **Augmenter**: Prev Next Augmenter

### Generation
- **llama_index_llm**: OpenAI, HuggingFace, Ollama 지원
- **vLLM**: GPU 가속 로컬 모델

---

## 6. VectorDB 설정 (Qdrant)

```yaml
vectordb:
  - name: openai_qdrant
    db_type: qdrant
    embedding_model: openai_embed_3_large
    collection_name: openai_embed_3_large
    client_type: docker
    embedding_batch: 50
    similarity_metric: cosine
    dimension: 1536
```

### 파라미터
| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `db_type` | `qdrant` | - |
| `embedding_model` | 임베딩 모델 ID | - |
| `collection_name` | 컬렉션 이름 | - |
| `client_type` | `docker` 또는 `cloud` | - |
| `dimension` | 벡터 차원 | 1536 |
| `similarity_metric` | `cosine`, `l2`, `ip` | cosine |
| `embedding_batch` | 임베딩 배치 크기 | 100 |

### 임베딩 모델 등록 (v0.3.13+)
```yaml
vectordb:
  - name: huggingface_qdrant
    db_type: qdrant
    embedding_batch: 16
    embedding_model:
      - type: huggingface
        model_name: intfloat/multilingual-e5-large-instruct
```

### 레거시 방식 (Python 코드)
```python
import autorag
from autorag import LazyInit
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

autorag.embedding_models['kosimcse'] = LazyInit(
    HuggingFaceEmbedding,
    model_name="BM-K/KoSimCSE-roberta-multitask"
)
```

---

## 7. BM25 한국어 토크나이저

```yaml
modules:
  - module_type: bm25
    bm25_tokenizer: [ko_kiwi, ko_okt, ko_kkma, porter_stemmer, space]
```

| 토크나이저 | 설명 | 요구사항 |
|-----------|------|---------|
| `ko_kiwi` | 한국어 (권장) | - |
| `ko_okt` | 한국어 (OKt) | konlpy |
| `ko_kkma` | 한국어 형태소 | konlpy |
| `porter_stemmer` | 영어 (기본값) | - |
| `space` | 공백 분리 (다국어) | - |

---

## 8. Reranker (FlashRank)

```yaml
modules:
  - module_type: flashrank_reranker
    batch: 32
    model: "ms-marco-MiniLM-L-12-v2"
```

---

## 9. Generator (LLM)

### OpenAI
```yaml
modules:
  - module_type: llama_index_llm
    llm: openai
    model: [gpt-4o-mini, gpt-3.5-turbo]
    temperature: [0, 0.5]
```

### Ollama (로컬)
```yaml
modules:
  - module_type: llama_index_llm
    llm: ollama
    model: qwen3:4b-instruct-2507-q4_K_M
```

---

## 10. 실행 명령어

### Validation (설정 검증)
```bash
autorag validate \
  --config config.yaml \
  --qa_data_path qa.parquet \
  --corpus_data_path corpus.parquet
```

### Evaluation (벤치마크 실행)
```bash
autorag evaluate \
  --config config.yaml \
  --qa_data_path qa.parquet \
  --corpus_data_path corpus.parquet \
  --project_dir ./benchmark_results
```

### Dashboard (결과 시각화)
```bash
autorag dashboard --trial_dir ./benchmark_results/0
```

### Best Config 추출
```bash
autorag extract_best_config \
  --trial_path ./benchmark_results/0 \
  --output_path best_pipeline.yaml
```

### 배포
```bash
# API 서버
autorag run_api --trial_dir ./benchmark_results/0 --host 0.0.0.0 --port 8000

# Web UI
autorag run_web --trial_path ./benchmark_results/0
```

---

## 11. 결과 구조

```
benchmark_results/
├── 0/                          # trial 폴더
│   ├── trial.json              # 평가 시간 정보
│   ├── summary.csv             # ★ 최적 모듈/파라미터 요약
│   ├── node_line_1/
│   │   ├── retrieval/
│   │   │   ├── summary.csv
│   │   │   └── ...
│   │   ├── reranker/
│   │   └── generator/
│   └── ...
```

**`summary.csv`가 가장 중요한 파일**: 데이터셋에 최적인 모듈과 파라미터를 보여준다.

---

## 12. 현재 노트북 RAG 구조 분석 (embedding_combinations_lab.ipynb)

### 파이프라인 구조
```
PDF → Markdown → Parent-Child Chunking → Qdrant (Dense+Sparse Hybrid)
                                              ↓
Query → Summarize → Analyze/Rewrite → Agent (Search+Retrieve) → Aggregate → Response
```

### 6가지 임베딩 조합

| # | Dense Model | Sparse Model | 차원 |
|---|-------------|-------------|------|
| 1 | KoSimCSE-roberta | BM25+OKt | 768 |
| 2 | multilingual-e5-large | SPLADE | 1024 |
| 3 | BGE-M3 | BGE-M3 (내장) | 1024 |
| 4 | all-MiniLM-L6-v2 | BM25 (FastEmbed) | 384 |
| 5 | OpenAI text-embedding-3-large | SPLADE | 3072 |
| 6 | Upstage Solar | BM25+OKt | 4096 |

### AutoRAG 벤치마크 매핑

노트북의 6가지 조합을 AutoRAG에서 테스트하려면:
- **Dense (VectorDB)**: 각 임베딩 모델별 Qdrant 컬렉션 생성
- **Sparse (BM25)**: `ko_okt`, `ko_kiwi` 토크나이저 비교
- **Hybrid**: `hybrid_rrf` 모듈로 Dense+Sparse 결합
- **Reranker**: FlashRank 등으로 후처리 최적화
- **Generator**: OpenAI GPT-4o-mini 또는 Ollama 로컬 모델
