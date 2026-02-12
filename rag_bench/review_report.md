# RAG 파이프라인 조합형 벤치마크 리팩토링 — 실현 가능성 평가 보고서

**작성일**: 2026-02-12
**대상**: rag_bench 레이어 분할 + 전수 조합 테스트 개선안

---

## 1. 개선 목표

현재 **전략 단위 비교**(DenseSparse, ColBERT, GraphRAG 등)를
**레이어 단위 조합 비교**로 전환하여, 각 구성 요소의 기여도를 독립적으로 측정한다.

```
현재 (Strategy = 고정 조합)            개선안 (Layer = 독립 조합)
┌──────────────────────┐           ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐
│ DenseSparse Combo 1  │           │  Dense   │×│  Sparse  │×│ Reranker │×│  LLM    │
│ (KoSimCSE + BM25/OKt)│    →      │  Model   │ │  Model   │ │  Layer   │ │ Support │
└──────────────────────┘           └─────────┘ └──────────┘ └──────────┘ └─────────┘
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

### 3.1 5-Layer 파이프라인

```
Layer 1          Layer 2          Layer 3           Layer 4          Layer 5
Dense Model  →   Sparse Model →   Retrieval Mode →  Reranker     →   LLM Support

KoSimCSE(768d)   BM25/OKt         Dense Only        None             None
E5(1024d)        SPLADE           Sparse Only       ColBERT Rerank   Contextual
BGE-M3(1024d)    FastEmbed BM25   Hybrid(RRF)       FlashRank
MiniLM(384d)     None(skip)
```

### 3.2 조합 수 계산

| Layer | 옵션 수 | 항목 |
|-------|:------:|------|
| Dense Model | 4 | KoSimCSE, E5, BGE-M3, MiniLM |
| Sparse Model | 4 | BM25/OKt, SPLADE, FastEmbed BM25, None |
| Retrieval Mode | 3 | Dense Only, Sparse Only, Hybrid |
| Reranker | 3 | None, ColBERT Rerank, FlashRank |
| LLM Support | 2 | None, Contextual Retrieval |

**이론적 최대**: 4 × 4 × 3 × 3 × 2 = **288 조합**

### 3.3 무효 조합 제거

| 제약 조건 | 이유 |
|----------|------|
| Sparse=None + Mode=Sparse Only | 스파스 모델 없이 스파스 검색 불가 |
| Sparse=None + Mode=Hybrid | 하이브리드에 스파스 필수 |
| Mode=Dense Only + Sparse 선택 | Dense Only 시 Sparse 선택 무의미 (1가지로 축소) |
| BGE-M3 Sparse + 타 Dense | BGE-M3 sparse는 BGE-M3 dense 전용 (모델 아키텍처 제약) |

**유효 조합 계산:**

```
Case A: Dense Only (sparse 무관)
  → 4 dense × 1(sparse=None) × 3 reranker × 2 llm = 24

Case B: Sparse Only
  → 유효 (dense, sparse) 쌍:
    4 dense × 2 범용sparse(BM25/OKt, SPLADE) = 8
    + 1(BGE-M3) × 3 sparse(BM25/OKt, SPLADE, BGE-M3) = 3  → BGE-M3 전용 1개 추가
    = 9 쌍 × 3 reranker × 2 llm = 54

Case C: Hybrid (동일 제약)
  = 54

유효 조합 = 24 + 54 + 54 = 132
```

**+ 독립 전략:**

| 전략 | 조합 수 | 비고 |
|------|:------:|------|
| ColBERT 단독 | 1 | 자체 MaxSim, reranker 불필요 |
| GraphRAG | 1 | 별도 파이프라인 |

**총 유효 조합 ≈ 134개**

---

## 4. 리소스 비용 분석

### 4.1 시간 비용 (20 QA 기준)

| 단계 | 단위 비용 | 조합당 | 총 소요 |
|------|----------|:------:|:------:|
| **Dense 인덱싱** (763 chunks) | 30s~3min | 재사용 | 4 모델 × ~90s = **6분** |
| **Sparse 인덱싱** (BM25 fit) | 5~15s | 재사용 | 3종 × ~10s = **30초** |
| **Qdrant 컬렉션 생성** | 10~30s | 조합마다 | ~30개 × 20s = **10분** |
| **ColBERT Rerank 모델 로드** | 20~40s | 1회 | **30초** |
| **FlashRank 모델 로드** | 5~10s | 1회 | **10초** |
| **Contextual LLM** (763 chunks) | 3~5min | base별 1회 | ~30 base × 4min = **120분** |
| **검색** (20 QA) | 1~10s | 조합마다 | 134 × ~5s = **11분** |
| **RAGAS 평가** (20 QA) | 2~5min | 조합마다 | 134 × ~3min = **402분** |

### 4.2 총 예상 소요 시간

| 시나리오 | 조합 수 | 예상 시간 |
|----------|:------:|:---------:|
| **Full (모든 유효 조합 + RAGAS)** | 134 | **~9시간** |
| **레이턴시만 (--no_ragas)** | 134 | **~30분** |
| **Reranker 제외** | ~44 | **~3시간** |
| **LLM Support 제외** | ~66 | **~3.5시간** |
| **최소 검증 (core 20)** | ~20 | **~1시간** |

### 4.3 API 비용

| 항목 | 모델 | 사용량 | 비용 |
|------|------|--------|:----:|
| RAGAS 평가 | gpt-3.5-turbo | 134조합 × 20QA × ~2K tokens | **~$2.70** |
| Contextual Retrieval | gpt-4o-mini | ~30 base × 763 chunks × ~500 tokens | **~$1.70** |
| GraphRAG 인덱싱 | gpt-4.1-nano | 33 parents × ~2K tokens | **~$0.01** |
| | | **합계** | **~$4.40** |

### 4.4 디스크/메모리

| 리소스 | 예상 사용량 |
|--------|:---------:|
| Qdrant 컬렉션 (~30개) | ~3 GB |
| ColBERT 모델 | ~1.2 GB |
| FlashRank 모델 | ~150 MB |
| Dense 임베딩 모델 (4종) | ~4 GB |
| **총 메모리 피크** | **~8 GB** |

---

## 5. 구현 방안

### 5.1 리팩토링 범위

| 파일 | 변경 | 난이도 | LOC |
|------|------|:------:|:---:|
| `strategies/dense_sparse.py` | **대폭 리팩토링** — `COMBO_DEFINITIONS` 제거, dense/sparse 독립 파라미터 | 높음 | ~280 |
| `scripts/run_all_combos.py` | **전면 재작성** — 조합 생성기, 인덱스 캐싱, 단계별 실행 | 높음 | ~300 |
| `strategies/__init__.py` | export 갱신 | 낮음 | ~5 |
| `base.py` | 변경 없음 | - | 0 |
| `runner.py` | 변경 없음 | - | 0 |
| `evaluation.py` | 변경 없음 | - | 0 |
| 리랭킹 전략들 | **변경 없음** (이미 Decorator 패턴) | - | 0 |
| **합계** | | | **~585** |

### 5.2 핵심 설계 변경

#### A. DenseSparseStrategy 분해

```python
# 현재: combo_id 기반 고정 조합
strategy = DenseSparseStrategy(combo_id=1)

# 개선: 독립 파라미터 기반 (combo_id 하위 호환 유지)
strategy = DenseSparseStrategy(
    dense_model="BM-K/KoSimCSE-roberta-multitask",
    sparse_type="korean_bm25",     # "korean_bm25" | "splade" | "fastembed_bm25" | None
    retrieval_mode="hybrid",       # "hybrid" | "dense_only" | "sparse_only"
    qdrant_path="auto",            # 자동 해시 기반 경로
)

# 하위 호환: 기존 combo_id도 동작
strategy = DenseSparseStrategy(combo_id=1)  # 내부에서 파라미터로 변환
```

#### B. 인덱스 캐싱 매니저

동일 (dense_model, sparse_type, docs_hash) 조합은 Qdrant 컬렉션 공유:

```python
index_key = sha256(dense_model + sparse_type + docs_hash)
if index_key in cache:
    strategy.load_existing(cache[index_key])
else:
    strategy.index(child_chunks)
    cache[index_key] = strategy.qdrant_path
```

**인덱싱 횟수**: 134 조합이지만 실제 인덱싱은 **~16회** (4 dense × 4 sparse 고유 조합)

#### C. 조합 생성기

```python
@dataclass
class ComboSpec:
    dense: str           # 모델 이름
    sparse: Optional[str]  # sparse 타입 또는 None
    mode: str            # "dense_only" | "sparse_only" | "hybrid"
    reranker: Optional[str]  # None | "colbert" | "flashrank"
    llm_support: Optional[str]  # None | "contextual"

def generate_valid_combinations() -> List[ComboSpec]:
    """무효 조합을 필터링한 유효 조합 목록 생성."""
    ...
```

### 5.3 실행 전략: 2-Pass 방식

```
Pass 1: 레이턴시 전수 조사 (--no_ragas, ~30분)
  → 134 조합 레이턴시 측정
  → 레이어별 평균 레이턴시 히트맵 생성

Pass 2: 품질 선별 평가 (상위 N만 RAGAS, ~1시간)
  → Pass 1에서 상위 20~30 조합 선별
  → RAGAS 4개 메트릭 평가
  → 레이어별 품질 기여도 분석
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
| 무효 조합 필터링 | **가능** | 규칙 기반 필터 단순 구현 |

### 6.2 리스크

| 리스크 | 심각도 | 완화 방안 |
|--------|:------:|----------|
| **메모리 부족** (Dense 4종 + ColBERT 동시 로드) | 중간 | 모델 lazy load + 사용 후 unload |
| **Qdrant 컬렉션 과다** (~30개) | 낮음 | 해시 기반 네이밍 + 자동 정리 |
| **RAGAS 시간 폭발** (134 × 3분) | 높음 | 2-Pass 방식 (레이턴시 → 선별 RAGAS) |
| **Contextual LLM 비용** | 중간 | 캐싱 + 대표 base만 적용 |

### 6.3 투입 공수: ~585 LOC

기존 아키텍처(BaseRAGStrategy ABC, Decorator 패턴)가 잘 설계되어 있어,
**DenseSparseStrategy 1곳만 분해하면 나머지는 기존 패턴 재사용** 가능.

---

## 7. 권장 실행 단계

### Phase 1: DenseSparse 분해 (핵심)

```
목표: dense_model과 sparse_type을 독립 파라미터로 분리
범위: dense_sparse.py 리팩토링 + combo_id 하위 호환
검증: 기존 4 combo 동일 결과 재현
```

### Phase 2: 조합 생성기 + 인덱스 캐싱

```
목표: 유효 조합 자동 생성 + 인덱싱 중복 제거
범위: run_all_combos.py 재작성
검증: --dry_run으로 조합 목록 출력 후 확인
```

### Phase 3: 전체 조합 벤치마크

```
1차: --no_ragas 레이턴시 전수 조사 (~30분)
2차: 상위 20 조합 RAGAS 평가 (~1시간)
3차: Contextual 조합 추가 평가
```

### Phase 4: 레이어별 기여도 분석

```
목표: 각 레이어의 독립적 기여도 시각화
  - Dense 모델별 평균 성능 (다른 레이어 고정)
  - Sparse 모델별 평균 성능
  - Reranker 추가 시 성능 향상률
  - Contextual Retrieval 추가 시 성능 향상률
```

---

## 8. 결론

| 항목 | 평가 |
|------|------|
| **기술적 실현 가능성** | **높음** — Decorator 패턴 기반 기존 아키텍처 활용 |
| **리팩토링 핵심** | DenseSparseStrategy 1곳만 분해하면 나머지는 기존 패턴 재사용 |
| **유효 조합** | 134개 (이론적 288개 중 무효 제거) |
| **실행 시간** | 레이턴시만 ~30분, 선별 RAGAS ~1시간 (2-Pass 전략) |
| **API 비용** | ~$4.40 (저렴) |
| **최대 가치** | 레이어별 기여도 분석 — "KoSimCSE + SPLADE + FlashRank"처럼 현재 불가능한 교차 조합을 과학적으로 측정 |
| **권장** | **Phase 1-2 즉시 진행 가능** |
