# AutoRAG 벤치마크 통합 리서치

## 1. AutoRAG 개요

**AutoRAG**는 한국의 Markr(마커) 사에서 개발한 오픈소스 RAG 파이프라인 최적화 프레임워크로, AutoML 스타일의 자동화된 실험을 통해 주어진 데이터셋에 최적인 RAG 파이프라인을 탐색한다.

- **GitHub**: [Marker-Inc-Korea/AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) (4.1k+ stars)
- **논문**: [arXiv:2410.20878](https://arxiv.org/abs/2410.20878)
- **PyPI**: v0.3.21 (2025-11-14), Python 3.10/3.11/3.12
- **라이선스**: Apache 2.0

### 핵심 기능 3축

| 기능 | 설명 |
|------|------|
| Data Creation | 원시 문서 -> 파싱 -> 청킹 -> LLM 기반 QA 데이터셋 자동 생성 |
| Optimization | YAML 설정 기반 모든 노드/모듈 조합 자동 실험 → 최적 파이프라인 탐색 |
| Deployment | 최적 파이프라인을 API 서버, Web UI, Python 코드로 즉시 배포 |

### Greedy Optimization 알고리즘

1. Query Expansion → Retrieval → Passage Augmentation → Reranking → Prompt Making → Generation 순차 진행
2. 각 단계(Node)에서 메트릭 기반 최적 모듈 선택 후 고정
3. `m x n` 조합을 `m + n`으로 축소하여 탐색 효율화

## 2. 지원 노드 및 모듈

### 8개 노드 타입, 30개+ 모듈

| Node | 주요 모듈 |
|------|----------|
| Query Expansion | query_decompose, hyde, multi_query_expansion |
| Retrieval | bm25(ko_kiwi/ko_okt), vectordb(Qdrant/Chroma/Milvus), hybrid_rrf, hybrid_cc |
| Passage Augmenter | prev_next_augmenter |
| Passage Reranker | colbert_reranker, flashrank, monot5, cohere, rankgpt, jina, koreranker 등 16+ |
| Passage Filter | similarity_threshold_cutoff, percentile_cutoff, recency_filter |
| Passage Compressor | tree_summarize, refine, long_llm_lingua |
| Prompt Maker | fstring, chat_fstring, long_context_reorder |
| Generator | llama_index_llm, openai_llm, vllm, vllm_api |

## 3. rag_bench 전략 매핑 분석

| rag_bench 전략 | AutoRAG 매핑 | 상세 |
|---------------|-------------|------|
| DenseSparseStrategy | **직접 매핑** | vectordb + bm25 + hybrid_rrf/cc. SPLADE는 미지원 |
| ColBERTStrategy | **리랭커만** | colbert_reranker 모듈 지원. 전체 코퍼스 1차 검색은 불가 |
| ColBERTRerankStrategy | **직접 매핑** | passage_reranker 노드의 colbert_reranker |
| GraphRAGStrategy | **매핑 불가** | Knowledge Graph RAG 미지원 |

## 4. 통합 계획

### Phase 1: 데이터 포맷 브릿지 (난이도: 낮)
- `qa_dataset.json` → `qa.parquet` (qid, query, retrieval_gt, generation_gt)
- child_chunks → `corpus.parquet` (doc_id, contents, metadata)
- 이미 `autorag_benchmark/data/`에 parquet 파일 존재 — 생성 파이프라인 참조 가능

### Phase 2: YAML 설정 확장 (난이도: 낮)
- 기존 `hybrid_benchmark_config.yaml` 확장
- colbert_reranker, flashrank_reranker, koreranker, pass_reranker 추가

### Phase 3: 결과 통합 비교 (난이도: 중)
- AutoRAG summary.csv + rag_bench all_combos_ragas.csv 통합 대시보드

### Phase 4: 커스텀 모듈 추가 (난이도: 높, 선택)
- AutoRAG Fork 필요 — 권장하지 않음

### 권장 접근: 병행 운영

```
rag_bench (자체)                     AutoRAG (보조)
────────────────────               ────────────────────
ColBERT 전체 코퍼스 검색             Dense 임베딩 모델 비교
GraphRAG                           BM25 토크나이저 비교
RAGAS 심층 평가                     Hybrid 가중치 + 리랭커 조합 자동 탐색
커스텀 전략 자유도                    프롬프트/LLM 비교 + 배포
```

## 5. AutoRAG vs rag_bench 비교

### AutoRAG 장점
- 자동 조합 탐색 (YAML 선언적)
- 내장 메트릭 풍부 (retrieval_f1, bleu, rouge 등)
- Greedy 최적화로 효율적 탐색
- 배포 + 대시보드 즉시 가능
- BM25 한국어 토크나이저 내장 (ko_kiwi, ko_okt)

### AutoRAG 단점
- 커스텀 검색 모듈 확장 어려움 (공식 플러그인 API 없음)
- GraphRAG, SPLADE, ColBERT 1차 검색 미지원
- Parquet 포맷 강제
- LangChain 구버전 의존성 충돌 가능

### rag_bench 장점
- BaseRAGStrategy ABC로 어떤 전략이든 통합 가능
- RAGAS LLM 기반 심층 평가
- 경량 의존성, 기업 네트워크 환경 대응
- 세밀한 코드 수준 제어

### rag_bench 단점
- 조합 탐색 수동
- 배포/대시보드 기능 없음
- 메트릭 수동 추가

## 6. 프로젝트 기존 AutoRAG 자산

| 파일 | 설명 |
|------|------|
| `autorag_benchmark/config/benchmark_config.yaml` | Dense only 벤치마크 |
| `autorag_benchmark/config/hybrid_benchmark_config.yaml` | Hybrid + Reranker 벤치마크 |
| `autorag_benchmark/data/qa.parquet` | QA 데이터 |
| `autorag_benchmark/data/corpus.parquet` | 코퍼스 데이터 |
| `autorag_benchmark/results/` | 기존 실행 결과 (trial 0, 1) |
| `autorag_research.md` | 기존 AutoRAG 리서치 문서 |
