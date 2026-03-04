---
title: "Execution Summary: RAGAS E2E 3-Layer 조합 벤치마크 파이프라인"
tags:
  - execution-summary
  - branch:feat/ragas-e2e-pipeline
  - ragas
  - 3-layer
  - benchmark
type: execution-summary
created: 2026-02-12T05:30:00Z
contextual_description: "3-Layer 교차 조합 벤치마크 파이프라인 전체 구현 — DenseSparse 분해, evaluation 서브패키지, run_all_combos 리팩토링"
keywords:
  - DenseSparseStrategy
  - ComboSpec
  - IndexCacheManager
  - ExtendedRAGEvaluator
  - MetricRegistry
  - 2-Pass
  - preset
  - dry-run
---

## Execution Summary: RAGAS E2E 3-Layer 조합 벤치마크 파이프라인

### 개요

`feat/ragas-e2e-pipeline` 브랜치에서 5-Phase 계획을 단일 세션으로 완료.
기존 4개 고정 조합(combo_id 1~4)만 가능하던 구조를 72개 교차 조합이 가능한 3-Layer 아키텍처로 전환.

### 커밋

- `3850eba` feat: 3-Layer 조합 벤치마크 파이프라인 구현 (72개 교차 조합 + 2-Pass 실행)
  - 8 files changed, +1453, -223 lines

### 변경 파일 상세

| 파일 | 상태 | 핵심 변경 |
|------|:----:|----------|
| `docs/research/ragas_e2e_evaluation.md` | 신규 | RAGAS v0.4 API 리서치, 3-Layer 설계 근거, 비용/시간 분석 |
| `rag_bench/review_report.md` | 수정 | 5-Layer(134개) → 3-Layer(74개) 설계 전면 업데이트 |
| `rag_bench/strategies/dense_sparse.py` | 리팩토링 | `dense_model`/`sparse_type` 독립 파라미터, `DENSE_MODELS`/`SPARSE_TYPES` 레지스트리, combo_id 하위 호환 |
| `rag_bench/evaluation.py` → `evaluation/legacy.py` | 이동 | 기존 RAGEvaluator 그대로 보존 |
| `rag_bench/evaluation/__init__.py` | 신규 | RAGEvaluator + ExtendedRAGEvaluator 재수출 |
| `rag_bench/evaluation/metrics.py` | 신규 | MetricTier/MetricPreset enum, METRIC_REGISTRY(10개), create_metrics() |
| `rag_bench/evaluation/evaluator.py` | 신규 | ExtendedRAGEvaluator, CostTracker, EvaluationReport, rank_strategies, SCORING_PROFILES |
| `rag_bench/scripts/run_all_combos.py` | 리팩토링 | ComboSpec, generate_valid_combinations, IndexCacheManager, build_strategy_from_spec, 2-Pass 실행, 레이어 기여도 분석, 리포트 생성 |

### 아키텍처 설계 결정

1. **DenseSparseStrategy는 hybrid 전용 base retriever**
   - `retrieval_mode` 파라미터를 추가하지 않음
   - Dense Only/Sparse Only 제거 → 무효 조합 규칙 0개로 단순화

2. **기존 Decorator 패턴 100% 재사용**
   - ColBERTRerankStrategy, FlashRankRerankStrategy, ContextualRetrievalStrategy: 코드 변경 없음
   - base.py, runner.py, colbert.py, graph_rag.py: 변경 없음

3. **3-Layer 구조**
   - Layer 1: Dense Model (4종) — kosimcse, e5, bge-m3, minilm
   - Layer 2: Sparse Model (3종) — korean_bm25, splade, fastembed_bm25
   - Layer 3: Retrieval Mode (6종) — reranker(None/ColBERT/FlashRank) × llm_support(None/Contextual)
   - 유효 조합: 4 × 3 × 6 = 72개 + 독립(ColBERT, GraphRAG) = 74개

4. **레거시 + 새 모드 공존**
   - `--preset` 없으면: 기존 `--combos`/`--skip_*` 레거시 모드
   - `--preset` 있으면: 새 3-Layer 조합 생성기 모드

5. **인덱스 캐싱**: 동일 (dense, sparse) 쌍은 Qdrant 컬렉션 공유 → 72조합이지만 12회만 인덱싱

### 새 CLI 옵션

```
--preset quick|standard|full   프리셋 선택
--pass1-only                   레이턴시만 측정
--top_n N                      Pass 1 후 상위 N만 RAGAS
--dry-run                      조합 목록만 출력
--layers                       레이어별 기여도 분석
```

### 프리셋 조합 수

| 프리셋 | 조합 수 | 고유 인덱스 |
|--------|:------:|:--------:|
| quick | 4 | 2 |
| standard | 24 | 12 |
| full | 72 | 12 |

### 검증 결과

- DenseSparse combo_id=1~4 하위 호환: PASS
- 교차 조합 (kosimcse+splade): PASS
- 전체 모델명 입력: PASS
- ValueError 처리 (combo_id 없고 dense_model도 없을 때): PASS
- `from rag_bench.evaluation import RAGEvaluator`: PASS
- `from rag_bench.evaluation import ExtendedRAGEvaluator`: PASS
- dry-run full=72개, standard=24개, quick=4개: PASS
- 기존 CLI 플래그 유지: PASS

### 미완료 / 후속 작업

- [ ] 실제 벤치마크 실행 (`--preset quick --pass1-only`)
- [ ] RAGAS v0.4 class-based 메트릭 실 동작 검증 (ExtendedRAGEvaluator 실행)
- [ ] GraphRAG/ColBERT 독립 전략을 preset 모드에 통합
- [ ] per-sample CSV 출력 (`e2e_per_sample.csv`)
- [ ] PR 생성 및 main 머지
