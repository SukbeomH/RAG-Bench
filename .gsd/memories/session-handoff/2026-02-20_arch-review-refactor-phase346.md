# Session Handoff: 아키텍처 리뷰 + 리팩토링 Phase 3/4/6

## Date: 2026-02-20
## Branch: master

---

## What Was Done

### 1. 버그 수정 — ctx_cache cleanup cascade (이전 세션 이월)
- **증상**: ColBERT/FlashRank Rerank + Contextual Retrieval 조합에서 `index()를 먼저 호출하세요` 에러
- **원인**: `strategy.cleanup()` 호출이 `ctx_cache`에 캐싱된 `ContextualRetrievalStrategy._is_ready`를 False로 오염
- **수정**: `colab_runner.py` — strategy 또는 `_base_strategy`가 `ctx_cache`에 있으면 cleanup cascade 건너뜀
- **커밋**: `8d2579f` fix(colab): ctx_cache cleanup cascade 방지

### 2. 평가 LLM 초기화 방식 수정
- **문제**: `llm_factory(AsyncOpenAI)` → `agenerate_text()` 미지원으로 RAGAS 평가 실패
- **수정**: `evaluator.py` — `ChatOpenAI + LangchainLLMWrapper`로 교체, 기본 모델 `gpt-4o-mini`
- **커밋**: `6d75351` fix(eval): LLM 초기화 llm_factory → LangchainLLMWrapper

### 3. 전략 병렬 실행 지원
- `runner.py`: `parallel_strategies` 파라미터 추가, `ThreadPoolExecutor`로 전략 동시 실행
- `run_all_combos.py`: `--pass1-workers` CLI 인수 추가
- **커밋**: `9946a38` feat(runner): 전략 병렬 실행 지원

### 4. 아키텍처 리뷰 수행
- CRITICAL 2건, MAJOR 6건, MINOR 8건 식별
- 리팩토링 플랜 수립: `.gsd/plans/2026-02-20_refactor-modularity.md`
- **Phase 순서**: 4(device) → 3(StrategyRetriever) → 6(LLM 상수) → 1(combo/) → 5(utils) → 2(DI)

### 5. Phase 4 — `utils/device.py` 디바이스 감지 일원화
- `rag_bench/utils/__init__.py` 신설
- `rag_bench/utils/device.py` 신설: `detect_device()` (CUDA→CPU, MPS 제외)
- `colbert.py`, `colbert_rerank.py`: `_detect_device()` 인스턴스 메서드 제거 → `detect_device()` 유틸 사용
- `dense_sparse.py`: HuggingFaceEmbeddings `device="cpu"` 하드코딩 → `detect_device()`
- **커밋**: `7136f3e` refactor(base): StrategyRetriever 공통화 + detect_device() 유틸 추출

### 6. Phase 3 — StrategyRetriever 제네릭 클래스 통합
- `base.py`에 `StrategyRetriever(BaseRetriever)` 추가
- 5개 전략별 중복 Retriever 클래스 제거:
  - `ColBERTRetriever` (colbert.py)
  - `ColBERTRerankRetriever` (colbert_rerank.py)
  - `FlashRankRerankRetriever` (flashrank_rerank.py)
  - `ContextualRetrievalRetriever` (contextual_retrieval.py)
  - `_UpstageRetriever` (upstage_embed.py)
- **커밋**: `7136f3e`, `ef219c2`

### 7. Phase 6 — LLM 모델명 상수 중앙화
- `config.py`에 용도별 상수 추가:
  - `DEFAULT_ANSWER_LLM = "gpt-4o-mini"` (기존 `gpt-3.5-turbo`에서 업그레이드)
  - `DEFAULT_EVAL_LLM = "gpt-4o-mini"`
  - `DEFAULT_CONTEXTUAL_LLM = "gpt-4o-mini"`
- `colab_config.py`에 Colab 전용 상수 추가:
  - `DEFAULT_COLAB_ANSWER_LLM = "gpt-4o-mini"`
  - `DEFAULT_COLAB_EVAL_LLM = "gpt-4o-mini"`
- `runner.py`, `evaluator.py`, `contextual_retrieval.py`, `colab_runner.py` 상수 참조로 교체
- **커밋**: `4af9e41` refactor(config): LLM 모델명 상수 중앙화

---

## Current Architecture State

### 완료된 리팩토링
| Phase | 내용 | 상태 |
|-------|------|------|
| 3 | StrategyRetriever 5개 → 1개 통합 | ✅ 완료 |
| 4 | detect_device() 5곳 → utils/device.py 1곳 | ✅ 완료 |
| 6 | LLM 모델명 4곳 하드코딩 → config.py 상수 | ✅ 완료 |

### 미완료 리팩토링 (플랜 수립됨)
| Phase | 내용 | 우선순위 |
|-------|------|---------|
| 1 | `run_all_combos.py` → `rag_bench/combo/` 패키지 분리 | 🟡 단기 |
| 5 | 중복 유틸리티 공유 모듈 추출 | 🟡 단기 |
| 2 | `colab_config.py` monkey-patch → DI 패턴 | 🟢 중기 |

---

## Key Files Changed This Session

| 파일 | 변경 내용 |
|------|-----------|
| `rag_bench/base.py` | `StrategyRetriever` 제네릭 클래스 추가 |
| `rag_bench/utils/device.py` | `detect_device()` 신규 |
| `rag_bench/config.py` | `DEFAULT_ANSWER/EVAL/CONTEXTUAL_LLM` 상수 추가, MPS 패치 개선 |
| `rag_bench/strategies/colbert.py` | `ColBERTRetriever` 제거, `detect_device()` 사용 |
| `rag_bench/strategies/colbert_rerank.py` | `ColBERTRerankRetriever` 제거, `detect_device()` 사용 |
| `rag_bench/strategies/flashrank_rerank.py` | `FlashRankRerankRetriever` 제거 |
| `rag_bench/strategies/contextual_retrieval.py` | `ContextualRetrievalRetriever` 제거, `DEFAULT_CONTEXTUAL_LLM` 사용 |
| `rag_bench/strategies/upstage_embed.py` | `_UpstageRetriever` 제거 |
| `rag_bench/strategies/dense_sparse.py` | HuggingFace `device` 하드코딩 제거, OpenAI/Upstage 임베딩 추가 |
| `rag_bench/runner.py` | `DEFAULT_ANSWER_LLM`, `parallel_strategies` |
| `rag_bench/evaluation/evaluator.py` | `DEFAULT_EVAL_LLM`, `LangchainLLMWrapper` |
| `rag_bench_colab/colab_config.py` | `DEFAULT_COLAB_ANSWER/EVAL_LLM` 추가 |
| `rag_bench_colab/colab_runner.py` | ctx_cache cascade fix, 상수 참조 |
| `.gsd/plans/2026-02-20_refactor-modularity.md` | 리팩토링 플랜 (6 Phase) |

---

## What Needs To Be Done Next

### 즉시
1. **Phase 1 실행**: `run_all_combos.py`에서 `ComboSpec`, `IndexCacheManager`, `build_strategy_from_spec()`을 `rag_bench/combo/` 패키지로 추출
   - `combo/spec.py`, `combo/cache.py`, `combo/builder.py` 신설
   - `colab_runner.py` import 경로 업데이트

### 단기
2. **Phase 5 실행**: 중복 유틸리티 공유 모듈 추출
   - `_load_qa_dataset()` 중복 (`run_bench.py`, `run_all_combos.py`)
   - Markdown 리포트 생성 중복 (`run_all_combos.py`, `colab_runner.py`)
   - 답변 생성 로직 중복 (`runner.py`, `colab_runner.py`)

### 중기
3. **Phase 2 실행**: `colab_config.py` monkey-patch 제거
   - `DenseSparseStrategy` 생성자 `device` 파라미터 추가
   - `config.py` 환경 변수 기반 경로 설정

### 기타
4. **Colab 벤치마크 실행**: 126개 조합 (7 dense × 3 sparse × 6 mode) 검증
5. **시각화 Phase 2**: H-2 Violin, M-1 Pipeline, M-3 Gantt, M-4 Cost-Efficiency 구현

---

## Commits This Session
```
4af9e41 refactor(config): LLM 모델명 상수 중앙화 — DEFAULT_ANSWER/EVAL/CONTEXTUAL_LLM
ef219c2 refactor(strategies): 잔여 Retriever 래퍼 StrategyRetriever로 통합
7136f3e refactor(base): StrategyRetriever 공통화 + detect_device() 유틸 추출
0fbecbb fix(config): torch.set_default_device → mps.is_available 패치로 교체
d934d59 fix(dense): KoreanBM25Encoder vocab thread-safety + OOM 분석 문서 추가
9946a38 feat(runner): 전략 병렬 실행 지원 — parallel_strategies / --pass1-workers
6d75351 fix(eval): LLM 초기화 llm_factory → LangchainLLMWrapper, 기본 모델 gpt-4o-mini
8d2579f fix(colab): ctx_cache cleanup cascade 방지 — ContextualRetrieval _is_ready 오염 수정
e037d6a feat(dense): OpenAI + Upstage 임베딩 추가 — 126개 조합으로 확장
62aff16 feat(colab): flash-attn 설치 추가 — XLM-RoBERTa Flash Attention 가속
```

---

## Critical Notes
- `dense_sparse.py`의 `IndexCacheManager`는 여전히 `_dense_embeddings` private 속성에 직접 접근 (M-5 미해결)
- Phase 1 실행 시 `colab_runner.py`의 `run_all_combos` import 경로 변경 필수
- `run_tracker.py`의 `_detect_gpu()`는 설명 문자열 반환 용도이므로 `detect_device()`와 별도 유지 (역할 다름)
- Colab의 `gpt-4o-nano` → `gpt-4o-mini`로 변경됨 (비용 증가 가능, 성능 향상)
