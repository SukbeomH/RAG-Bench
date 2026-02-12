# RAG 파이프라인 조합형 벤치마크 리팩토링 — 실현 가능성 평가 보고서

**작성일**: 2026-02-12
**대상**: rag_bench 레이어 분할 + 전수 조합 테스트 개선안
**개정**: v2 — 5-Layer(134개 조합) → 3-Layer(74개 조합) 설계 반영

---

## 1. 개선 목표

현재 **전략 단위 비교**(DenseSparse, ColBERT, GraphRAG 등)를
**레이어 단위 조합 비교**로 전환하여, 각 구성 요소의 기여도를 독립적으로 측정한다.

```
현재 (Strategy = 고정 조합)            개선안 (Layer = 독립 조합)
┌──────────────────────┐           ┌─────────┐ ┌──────────┐ ┌──────────────┐
│ DenseSparse Combo 1  │           │  Dense   │×│  Sparse  │×│  Retrieval   │
│ (KoSimCSE + BM25/OKt)│    →      │  Model   │ │  Model   │ │  Mode        │
└──────────────────────┘           └─────────┘ └──────────┘ └──────────────┘
```

---

## 2. 현재 아키텍처 진단

### 2.1 커플링 분석

| 컴포넌트 | 현재 결합도 | 문제점 |
|----------|:---------:|--------|
| **DenseSparseStrategy** | **강결합** | dense model + sparse model + Qdrant가 `combo_id`로 하드코딩. 개별 교체 불가 |
| **ColBERTRerankStrategy** | 약결합 | `base_strategy`를 감싸는 Decorator 패턴. 잘 설계됨 |
| **FlashRankRerankStrategy** | 약결합 | 위와 동일한 Decorator 패턴. 잘 설계됨 |
| **ContextualRetrievalStrategy** | 중간 | `base_strategy` + `parent_pairs` 필요. 인덱싱 전 전처리 |
| **ColBERTStrategy** | 독립 | 자체 인덱싱/검색. 조합 대상 아닌 독립 전략 |
| **GraphRAGStrategy** | 독립 | 자체 지식 그래프. 조합 대상 아닌 독립 전략 |

### 2.2 핵심 병목: DenseSparseStrategy

현재 `COMBO_DEFINITIONS`가 dense+sparse를 1:1 고정:

```python
# 현재: combo_id로 묶여 있어 교차 조합 불가
1: KoSimCSE + BM25/OKt      # KoSimCSE + SPLADE는 테스트 불가
2: E5 + SPLADE              # E5 + BM25/OKt는 테스트 불가
3: BGE-M3 + BGE-M3          # BGE-M3 + BM25는 테스트 불가
4: MiniLM + FastEmbed BM25  # MiniLM + SPLADE는 테스트 불가
```

### 2.3 Reranker/LLM 레이어는 이미 분리됨

ColBERTRerank, FlashRank, Contextual Retrieval은 이미 `base_strategy`를 감싸는
Decorator 패턴으로 구현되어 있어, **임의의 base에 적용 가능**하다.
이 레이어들은 리팩토링 없이 그대로 조합에 참여할 수 있다.

---

## 3. 제안 레이어 아키텍처

### 3.1 3-Layer 파이프라인

5-Layer 설계에서 Dense Only/Sparse Only 모드를 제거하고, reranker와 llm_support를
Retrieval Mode 레이어의 on/off 조합으로 통합한 간결한 3-Layer 구조:

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

**핵심 설계 원칙**:
- DenseSparseStrategy는 항상 Hybrid 모드로 동작하는 **base retriever** 역할만 한다
- Retrieval mode 변형(reranker, llm_support)은 기존 Decorator 패턴이 처리
- `retrieval_mode` 파라미터를 DenseSparseStrategy에 추가하지 **않는다**

### 3.2 조합 수 계산

| Layer | 옵션 수 | 항목 |
|-------|:------:|------|
| Dense Model | 4 | kosimcse, e5, bge-m3, minilm |
| Sparse Model | 3 | korean_bm25, splade, fastembed_bm25 |
| Retrieval Mode | 6 | reranker(None/ColBERT/FlashRank) × llm_support(None/Contextual) |

**교차 조합**: 4 × 3 × 6 = **72개**

무효 조합 규칙 없음 — 모든 조합이 유효하다.

**+ 독립 전략:**

| 전략 | 조합 수 | 비고 |
|------|:------:|------|
| ColBERT 단독 | 1 | 자체 MaxSim, reranker 불필요 |
| GraphRAG | 1 | 별도 파이프라인 |

**총 유효 조합 = 72 + 2 = 74개**

### 3.3 vs 이전 5-Layer 설계 비교

| 항목 | 5-Layer (v1) | 3-Layer (v2) |
|------|:------:|:------:|
| 이론적 조합 | 288 | 72 |
| 유효 조합 | 134 | 72 |
| 무효 조합 규칙 | 4개 (복잡) | 0개 |
| Qdrant 인덱스 | ~30개 | 12개 |
| 코드 복잡도 | 높음 | 낮음 |

---

## 4. 리소스 비용 분석

### 4.1 시간 비용 (20 QA 기준)

| 단계 | 단위 비용 | 조합당 | 총 소요 |
|------|----------|:------:|:------:|
| **Dense 인덱싱** (763 chunks) | 30s~3min | 재사용 | 4 모델 × ~90s = **6분** |
| **Sparse 인덱싱** (BM25 fit) | 5~15s | 재사용 | 3종 × ~10s = **30초** |
| **Qdrant 컬렉션 생성** | 10~30s | base별 | 12개 × 20s = **4분** |
| **ColBERT Rerank 모델 로드** | 20~40s | 1회 | **30초** |
| **FlashRank 모델 로드** | 5~10s | 1회 | **10초** |
| **Contextual LLM** (763 chunks) | 3~5min | base별 1회 | 12 base × 4min = **48분** |
| **검색** (20 QA) | 1~10s | 조합마다 | 72 × ~5s = **6분** |
| **RAGAS 평가** (20 QA) | 2~5min | 조합마다 | 72 × ~3min = **216분** |

### 4.2 총 예상 소요 시간

| 시나리오 | 조합 수 | 예상 시간 |
|----------|:------:|:---------:|
| **quick (프리셋)** | 4 | **~15분** |
| **standard (프리셋)** | 24 | **~1.5시간** |
| **full + Pass1만** | 72 | **~30분** |
| **full + top_n 20** | 72→20 | **~1시간** |
| **Full (모든 조합 + RAGAS)** | 72 | **~4시간** |

### 4.3 API 비용

| 항목 | 모델 | 사용량 | 비용 |
|------|------|--------|:----:|
| RAGAS 평가 | gpt-4o-mini | 72조합 × 20QA × ~2K tokens | **~$2.00** |
| Contextual Retrieval | gpt-4o-mini | 12 base × 763 chunks × ~500 tokens | **~$0.70** |
| GraphRAG 인덱싱 | gpt-4.1-nano | 33 parents × ~2K tokens | **~$0.01** |
| | | **합계** | **~$2.70** |

### 4.4 디스크/메모리

| 리소스 | 예상 사용량 |
|--------|:---------:|
| Qdrant 컬렉션 (12개) | ~1.5 GB |
| ColBERT 모델 | ~1.2 GB |
| FlashRank 모델 | ~150 MB |
| Dense 임베딩 모델 (4종) | ~4 GB |
| **총 메모리 피크** | **~7 GB** |

---

## 5. 구현 방안

### 5.1 리팩토링 범위

| 파일 | 변경 | 난이도 | LOC |
|------|------|:------:|:---:|
| `strategies/dense_sparse.py` | **리팩토링** — dense/sparse 독립 파라미터 + combo_id 하위 호환 | 높음 | ~280 |
| `evaluation/` | **서브패키지 전환** — legacy.py, metrics.py, evaluator.py | 중간 | ~350 |
| `scripts/run_all_combos.py` | **리팩토링** — ComboSpec, 조합 생성기, 2-Pass, 인덱스 캐싱 | 높음 | ~500 |
| `base.py` | 변경 없음 | - | 0 |
| `runner.py` | 변경 없음 | - | 0 |
| 리랭킹 전략들 | **변경 없음** (이미 Decorator 패턴) | - | 0 |
| **합계** | | | **~1130** |

### 5.2 핵심 설계 변경

#### A. DenseSparseStrategy 분해

```python
# 현재: combo_id 기반 고정 조합
strategy = DenseSparseStrategy(combo_id=1)

# 개선: 독립 파라미터 기반 (combo_id 하위 호환 유지)
strategy = DenseSparseStrategy(
    dense_model="kosimcse",         # 짧은 키 또는 전체 모델명
    sparse_type="korean_bm25",      # "korean_bm25" | "splade" | "fastembed_bm25"
    qdrant_path="auto",
)

# 하위 호환: 기존 combo_id도 동작
strategy = DenseSparseStrategy(combo_id=1)  # 내부에서 파라미터로 변환
```

**주의**: `retrieval_mode` 파라미터는 DenseSparseStrategy에 추가하지 않는다.
DenseSparseStrategy는 항상 hybrid 모드로 동작하는 base retriever 역할만 한다.
Retrieval mode 변형은 기존 Decorator 패턴(ColBERTRerank, FlashRank, ContextualRetrieval)이 처리.

#### B. 인덱스 캐싱 매니저

동일 (dense, sparse) 쌍은 Qdrant 컬렉션 공유:

```python
cache_key = f"{dense}:{sparse}"
if cache_key in cache:
    # 기존 인덱스 재사용
else:
    strategy.index(child_chunks)
    cache[cache_key] = (strategy, qdrant_path)
```

**인덱싱 횟수**: 72 조합이지만 실제 인덱싱은 **12회** (4 dense × 3 sparse)

#### C. 조합 생성기

```python
@dataclass
class ComboSpec:
    dense: str               # DENSE_MODELS 키
    sparse: str              # SPARSE_TYPES 값
    reranker: Optional[str]  # None | "colbert" | "flashrank"
    llm_support: Optional[str]  # None | "contextual"

def generate_valid_combinations(config) -> List[ComboSpec]:
    """3-Layer 카테시안 곱으로 유효 조합 생성."""
```

### 5.3 실행 전략: 2-Pass 방식

```
Pass 1: 레이턴시 전수 조사 (~6분)
  → N 조합 × 20 QA → 레이턴시 측정
  → all_combos_latency.csv 저장
  → 레이어별 평균 레이턴시 요약

Pass 2: 품질 선별 평가 (상위 N만 RAGAS)
  → 레이턴시 기준 상위 N 조합 선별
  → ExtendedRAGEvaluator로 RAGAS 평가
  → all_combos_ragas.csv 저장
  → 프로파일별 순위
```

---

## 6. 실현 가능성 판단

### 6.1 기술적 실현 가능성: **높음**

| 항목 | 평가 | 근거 |
|------|:----:|------|
| DenseSparse 분해 | **가능** | dense/sparse 초기화가 이미 `_init_dense()`, `_init_sparse()`로 분리됨 |
| 인덱스 캐싱 | **가능** | Qdrant path가 이미 파라미터화됨 |
| Reranker 자유 조합 | **즉시** | 이미 Decorator 패턴으로 완벽 분리됨 |
| LLM Support 조합 | **즉시** | ContextualRetrieval이 이미 base_strategy 래핑 |
| 무효 조합 필터링 | **불필요** | 3-Layer 설계로 모든 조합 유효 |

### 6.2 리스크

| 리스크 | 심각도 | 완화 방안 |
|--------|:------:|----------|
| **메모리 부족** (Dense 4종 + ColBERT 동시 로드) | 중간 | 모델 lazy load + 사용 후 unload |
| **Qdrant 컬렉션 과다** (12개) | 낮음 | 키 기반 네이밍 + 자동 정리 |
| **RAGAS 시간** (72 × 3분) | 중간 | 2-Pass 방식 (레이턴시 → 선별 RAGAS) |
| **Contextual LLM 비용** | 낮음 | 캐싱 + 12 base만 적용 |

### 6.3 투입 공수: ~1130 LOC

기존 아키텍처(BaseRAGStrategy ABC, Decorator 패턴)가 잘 설계되어 있어,
**DenseSparseStrategy 1곳만 분해하면 나머지는 기존 패턴 재사용** 가능.

---

## 7. 권장 실행 단계

### Phase 1: 리서치 문서 + 브랜치

```
목표: 3-Layer 설계 근거 문서화
범위: docs/research/ragas_e2e_evaluation.md, review_report.md 업데이트
```

### Phase 2: DenseSparse 분해 (핵심 전제)

```
목표: dense_model과 sparse_type을 독립 파라미터로 분리
범위: dense_sparse.py 리팩토링 + combo_id 하위 호환
검증: 기존 4 combo 동일 동작 + 교차 조합 동작 확인
```

### Phase 3: evaluation 서브패키지

```
목표: ExtendedRAGEvaluator + MetricRegistry + per-sample 평가
범위: evaluation/ 서브패키지 전환 (4파일)
검증: from rag_bench.evaluation import RAGEvaluator 호환
```

### Phase 4: run_all_combos.py 리팩토링

```
목표: 조합 생성기 + 인덱스 캐싱 + 2-Pass 실행
범위: ComboSpec, generate_valid_combinations, IndexCacheManager, --preset/--dry-run/--top_n
검증: --dry-run으로 72개 조합 출력, --preset quick --pass1-only 실행
```

### Phase 5: 검증

```
1. DenseSparse 하위 호환 (combo_id=1~4)
2. 교차 조합 동작 (kosimcse + splade)
3. dry-run 조합 생성 (72개)
4. 기존 CLI 호환 (--combos, --skip_*)
```

---

## 8. 결론

| 항목 | 평가 |
|------|------|
| **기술적 실현 가능성** | **높음** — Decorator 패턴 기반 기존 아키텍처 활용 |
| **리팩토링 핵심** | DenseSparseStrategy 1곳만 분해하면 나머지는 기존 패턴 재사용 |
| **유효 조합** | 74개 (3-Layer 72개 + 독립 2개) |
| **실행 시간** | 레이턴시만 ~6분, 선별 RAGAS ~1시간 (2-Pass 전략) |
| **API 비용** | ~$2.70 (저렴) |
| **최대 가치** | 레이어별 기여도 분석 — "KoSimCSE + SPLADE + FlashRank"처럼 현재 불가능한 교차 조합을 과학적으로 측정 |
| **권장** | **Phase 1-4 순차 진행** |
