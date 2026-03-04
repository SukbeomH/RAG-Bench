---
title: "버그 수정 + 최적화 + 기능 확장 세션"
tags: [bugfix, optimization, ragas-kg, evaluation-metrics, contextual-retrieval, colab]
type: general
created: 2026-02-19
contextual_description: "레이어별 기여도 분석 빈 출력 버그 수정, Contextual Retrieval 중복 초기화 최적화, RAGAS KG QA 생성, evaluation 메트릭 확장, Colab 경로 수정"
keywords: [_build_latency_summary, _ensure_initialized, COMPREHENSIVE, MetricPreset, SCORING_PROFILES, EvaluationReport, KnowledgeGraph]
related: [run_all_combos.py, dense_sparse.py, contextual_retrieval.py, generate_qa.py, runner.py, metrics.py, evaluator.py, colab_config.py]
---

## 세션 요약

이 세션에서는 벤치마크 로그 분석 → 버그 수정 → 최적화 → 기능 확장 → 문서 최신화를 순차 진행.

### 1. 레이어별 기여도 분석 빈 출력 버그 (ba25dcf)
- **원인**: `to_dataframe()`이 쿼리별 raw 행(`latency_ms`)을 반환, `_print_layer_contribution()`이 `avg_latency` 컬럼 기대
- **수정**: `_build_latency_summary()` 헬퍼 — raw DataFrame → 전략별 요약 (avg/min/max/p50, ms→s)
- **영향 범위**: `_print_layer_contribution()`, Pass 2 선별, `_generate_report()` 3곳 모두 수정

### 2. Contextual Retrieval 중복 초기화 최적화 (216d7e4)
- **원인**: `get_or_build_contextual()`이 새 DenseSparseStrategy 생성 → 모델 재로드
- **수정**:
  - 캐시된 base의 `_dense_embeddings`/`_sparse_embeddings` 객체 주입
  - `_ensure_initialized()`에 `elif self._client is None` 분기 (Qdrant만 초기화)
  - 캐시 100% 히트 시 진행 로그 억제

### 3. RAGAS KG QA 생성 (d2759e5)
- `--method ragas`: KnowledgeGraph → TestsetGenerator 파이프라인
- `--build-kg-only`, `--reuse-kg` 옵션
- `runner.py`: EvaluationReport 클래스 통합

### 4. Evaluation 메트릭 확장 (df07626)
- COMPREHENSIVE 프리셋: Core 4종 + Extended 핵심 5종
- Extended: context_entity_recall, response_relevancy, string_presence, exact_match, non_llm_string_similarity
- Scoring Profiles: default, comprehensive, lightweight

### 5. Colab 경로 수정 (70e5689)
- `autorag` → `RAG-Bench` 프로젝트명 통일

## 핵심 패턴/기법
- `_build_latency_summary()`: raw per-query → strategy-level aggregation
- 모델 객체 공유: IndexCacheManager에서 base→contextual base로 주입
- 조건부 초기화: `_ensure_initialized()`의 분기 확장
