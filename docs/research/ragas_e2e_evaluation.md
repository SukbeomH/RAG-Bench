# RAGAS E2E 평가 파이프라인 — 리서치 문서

**작성일**: 2026-02-12
**목적**: 3-Layer 조합 벤치마크를 위한 RAGAS v0.4+ API, 메트릭 카탈로그, 설계 근거 정리

---

## 1. RAGAS v0.4+ API 변경사항

### 1.1 Class-based Metrics (v0.4+)

RAGAS v0.4부터 메트릭이 class-based로 전환되었다. LLM/Embeddings를 명시적으로 주입한다.

```python
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LLM 래핑
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

# 메트릭 인스턴스 생성
faithfulness = Faithfulness(llm=evaluator_llm)
answer_relevancy = AnswerRelevancy(llm=evaluator_llm)
```

### 1.2 SingleTurnSample (v0.4+)

개별 샘플 단위 평가를 위한 데이터 클래스:

```python
from ragas import SingleTurnSample

sample = SingleTurnSample(
    user_input="질문",
    response="답변",
    retrieved_contexts=["context1", "context2"],
    reference="정답 (optional)",
)

score = await faithfulness.single_turn_ascore(sample)
```

### 1.3 EvaluationDataset + evaluate()

배치 평가:

```python
from ragas import EvaluationDataset, evaluate

dataset = EvaluationDataset(samples=[sample1, sample2, ...])
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
)
# results.to_pandas() → DataFrame (per-sample scores)
```

---

## 2. 사용 가능 메트릭 카탈로그

### 2.1 Core 4개 (LLM 필요)

| 메트릭 | 클래스명 | Reference 필요 | 설명 |
|--------|---------|:-----------:|------|
| Faithfulness | `Faithfulness` | No | 답변이 제공된 컨텍스트에 충실한가 |
| Answer Relevancy | `AnswerRelevancy` | No | 답변이 질문에 관련있는가 |
| Context Precision | `ContextPrecision` | Yes | 검색된 컨텍스트 중 관련 비율 |
| LLM Context Recall | `LLMContextRecall` | Yes | 정답에 필요한 정보가 컨텍스트에 포함되었는가 |

### 2.2 Extended 5개 (LLM 필요)

| 메트릭 | 클래스명 | Reference 필요 | 설명 |
|--------|---------|:-----------:|------|
| Answer Correctness | `AnswerCorrectness` | Yes | 답변의 사실적 정확성 |
| Factual Correctness | `FactualCorrectness` | Yes | 정답 대비 팩트 일치율 |
| Noise Sensitivity | `NoiseSensitivity` | Yes | 노이즈 컨텍스트에 대한 민감도 |
| Context Relevance | `ContextEntityRecall` | Yes | 엔티티 기반 컨텍스트 관련성 |
| Response Relevancy | `ResponseRelevancy` | No | 응답의 관련성 (별도 측정) |

### 2.3 Lightweight 3개 (LLM 불필요)

| 메트릭 | 클래스명 | Reference 필요 | 설명 |
|--------|---------|:-----------:|------|
| Semantic Similarity | `SemanticSimilarity` | Yes | 답변-정답 임베딩 유사도 |
| BLEU Score | `BleuScore` | Yes | BLEU 텍스트 유사도 |
| ROUGE Score | `RougeScore` | Yes | ROUGE 텍스트 유사도 |

---

## 3. 3-Layer 조합 설계 근거

### 3.1 5-Layer → 3-Layer 축소 이유

기존 review_report.md의 5-Layer 설계(288→134 조합)에서 불필요한 복잡도를 제거:

1. **Retrieval Mode (Dense Only / Sparse Only / Hybrid)**: Hybrid 전용으로 고정
   - 이유: Dense Only / Sparse Only는 하이브리드의 열등한 서브셋. 벤치마크 가치 낮음
   - 효과: 레이어 1개 제거, 무효 조합 규칙 대폭 단순화

2. **Reranker + LLM Support → Retrieval Mode 레이어 통합**:
   - Reranker (None/ColBERT/FlashRank) × LLM Support (None/Contextual) = 6가지 모드
   - 이를 단일 `retrieval_mode` 레이어로 표현

### 3.2 3-Layer 구조

```
Layer 1: Dense Model    Layer 2: Sparse Model    Layer 3: Retrieval Mode
┌────────────────┐     ┌──────────────────┐     ┌──────────────────────────────────────────┐
│ kosimcse (768d)│     │ korean_bm25      │     │ hybrid                                   │
│ e5 (1024d)     │  ×  │ splade           │  ×  │ hybrid_with_colbert_rerank                │
│ bge-m3 (1024d) │     │ fastembed_bm25   │     │ hybrid_with_flashrank_rerank              │
│ minilm (384d)  │     │                  │     │ hybrid_with_llm_support                   │
└────────────────┘     └──────────────────┘     │ hybrid_with_colbert_rerank_and_llm_support│
    4종                     3종                  │ hybrid_with_flashrank_rerank_and_llm_support│
                                                 └──────────────────────────────────────────┘
                                                     6종 (reranker 3 × llm_support 2)
```

### 3.3 유효 조합 수 계산

**기본 교차 조합**: 4 × 3 × 6 = **72개**
- Dense: kosimcse, e5, bge-m3, minilm (4종)
- Sparse: korean_bm25, splade, fastembed_bm25 (3종)
- Retrieval Mode: 6종 (reranker 3가지 × llm_support 2가지)

**독립 전략**: ColBERT 단독(1) + GraphRAG(1) = **2개**

**총 유효 조합**: 72 + 2 = **74개**

### 3.4 vs 5-Layer 비교

| 항목 | 5-Layer (이전) | 3-Layer (현재) |
|------|:------:|:------:|
| 이론적 조합 | 288 | 72 |
| 유효 조합 | 134 | 72 |
| 무효 조합 규칙 | 복잡 (4개) | 없음 |
| Qdrant 인덱스 | ~30개 | 12개 |
| Contextual 인덱싱 | ~30회 | 12회 |
| 복잡도 | 높음 | 낮음 |

---

## 4. 2-Pass 실행 전략 분석

### 4.1 Pass 1: 레이턴시 전수 조사

- 모든 유효 조합에 대해 검색만 수행 (RAGAS 없음)
- 조합당 20 QA × ~1-10s = ~2-5s 평균
- 총: 72 조합 × ~5s = ~6분 (인덱싱 시간 별도)
- 출력: `all_combos_latency.csv` (strategy, avg_latency, p50, p90, p99)

### 4.2 Pass 2: 선별 RAGAS 평가

- Pass 1 결과에서 상위 N 조합 선별 (기본: 레이턴시 기준)
- 선별된 조합만 RAGAS Core 4개 메트릭 평가
- 조합당 20 QA × ~3min = ~3min
- 상위 20개 평가 시: ~1시간

### 4.3 2-Pass의 장점

1. **비용 절감**: 72개 전체 RAGAS → 상위 20개만 = 70% 비용 절감
2. **빠른 피드백**: 레이턴시만 ~6분이면 레이어별 기여도 1차 파악
3. **점진적 확대**: 필요 시 top_n 늘려서 추가 평가

---

## 5. 비용/시간 분석

### 5.1 인덱싱 비용

| 항목 | 횟수 | 단위 시간 | 총 시간 |
|------|:----:|:-------:|:------:|
| Qdrant 인덱스 (dense+sparse) | 12 | ~90s | ~18분 |
| Contextual LLM 전처리 | 12 | ~4분 | ~48분 |
| ColBERT 모델 로드 | 1 | ~30s | ~30s |
| FlashRank 모델 로드 | 1 | ~10s | ~10s |

### 5.2 실행 비용 (프리셋별)

| 시나리오 | 조합 수 | Pass 1 (레이턴시) | Pass 2 (RAGAS) | API 비용 |
|----------|:------:|:---------:|:---------:|:--------:|
| quick | 4 | ~2분 | ~12분 | ~$0.15 |
| standard | 24 | ~10분 | ~1시간 | ~$1.00 |
| full | 72 | ~30분 | ~3시간 | ~$3.00 |
| full + top_n 20 | 72→20 | ~30분 | ~1시간 | ~$1.50 |

### 5.3 API 비용 상세

| 항목 | 모델 | 비용 |
|------|------|:----:|
| RAGAS 평가 (72조합 × 20QA) | gpt-4o-mini | ~$2.50 |
| Contextual Retrieval (12 base × 763 chunks) | gpt-4o-mini | ~$1.00 |
| GraphRAG 인덱싱 (33 parents) | gpt-4.1-nano | ~$0.01 |
| **합계** | | **~$3.50** |

---

## 6. 핵심 구현 원칙

1. **DenseSparseStrategy는 hybrid 전용**: Retrieval Mode 변형은 Decorator가 처리
2. **Decorator 패턴 완벽 활용**: ColBERTRerank, FlashRank, ContextualRetrieval 코드 변경 없음
3. **인덱스 캐싱**: 동일 (dense, sparse) 쌍은 Qdrant 컬렉션 공유 → 12회만 인덱싱
4. **하위 호환**: combo_id 기반 기존 코드 그대로 동작
5. **4파일 제한**: evaluation 서브패키지는 `__init__.py`, `legacy.py`, `metrics.py`, `evaluator.py`
