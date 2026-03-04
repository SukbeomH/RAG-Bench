---
title: "Execution Summary: RAG Bench 평가 시스템 전면 업그레이드"
tags:
  - execution-summary
  - ragas
  - metrics
  - evaluation
  - knowledge-graph
  - qa-generation
type: execution-summary
created: 2026-02-19T13:15:00+09:00
contextual_description: "RAGAS 메트릭 확장(COMPREHENSIVE 프리셋), ExtendedRAGEvaluator 메인 파이프라인 통합, RAGAS KG 기반 QA 생성 — 3 Phase 전체 구현 완료"
keywords:
  - MetricPreset.COMPREHENSIVE
  - SCORING_PROFILES
  - ExtendedRAGEvaluator
  - EvaluationReport
  - per-sample
  - weighted-score
  - KnowledgeGraph
  - TestsetGenerator
  - SingleHopQuerySynthesizer
  - MultiHopAbstractQuerySynthesizer
  - generate_qa
  - --metric-preset
  - --scoring-profile
  - --legacy-evaluator
related:
  - 2026-02-12_ragas-e2e-pipeline-implementation
---

## Execution Summary: RAG Bench 평가 시스템 전면 업그레이드

### 개요

`master` 브랜치에서 3-Phase 계획을 단일 세션으로 완료.
기존 Core 4개 메트릭만 사용하던 평가 파이프라인을 COMPREHENSIVE 7개 메트릭으로 확장하고,
`ExtendedRAGEvaluator`를 메인 파이프라인에 통합하며,
RAGAS KnowledgeGraph 기반 다양한 QA 데이터셋 생성 기능을 추가.

### 변경 파일 6개

| 파일 | Phase | 변경 내용 |
|------|:-----:|----------|
| `rag_bench/evaluation/metrics.py` | 1 | 5개 메트릭 추가, COMPREHENSIVE 프리셋 |
| `rag_bench/evaluation/evaluator.py` | 1 | comprehensive 스코어링 프로파일 (7메트릭 가중치) |
| `rag_bench/evaluation/__init__.py` | 1 | MetricPreset, SCORING_PROFILES, create_metrics export |
| `rag_bench/runner.py` | 2 | evaluate() 양방향 호환(dict/EvaluationReport), reports 프로퍼티 |
| `rag_bench/scripts/run_all_combos.py` | 2 | CLI 3개 추가, Evaluator 교체, 가중 점수 테이블, per-sample CSV |
| `rag_bench/scripts/generate_qa.py` | 3 | RAGAS KG QA 생성, CLI 5개 추가, 하위 호환 출력 포맷 |

### Phase 1: 메트릭 확장

- METRIC_REGISTRY에 5개 추가: `context_entity_recall`, `response_relevancy`, `string_presence`, `exact_match`, `non_llm_string_similarity`
- `MetricPreset.COMPREHENSIVE` = Core 4 + factual_correctness + context_entity_recall + response_relevancy (7개)
- `SCORING_PROFILES["comprehensive"]` = 7개 메트릭 균등에 가까운 가중치 (합=1.0)
- `NonLLMStringSimilarity`는 `rapidfuzz` 미설치 시 자동 스킵 (정상 동작)

### Phase 2: ExtendedRAGEvaluator 통합

- `BenchmarkRunner.evaluate()`: 반환값이 `EvaluationReport`이면 `aggregate_dict` 추출 + `_reports`에 보관
- `BenchmarkRunner.reports` 프로퍼티: per-sample 데이터 접근
- `run_all_combos.py` CLI: `--metric-preset`, `--scoring-profile`, `--legacy-evaluator`
- 기본 evaluator가 `ExtendedRAGEvaluator`로 변경 (`--legacy-evaluator`로 이전 RAGEvaluator 강제 가능)
- `_print_ragas_table()`: 마지막 열에 현재 프로파일 가중 점수 추가
- `_generate_report()`: 모든 프로파일별 가중 점수 테이블 추가
- per-sample CSV: `_benchdata/per_sample/{strategy_name}.csv` 자동 저장

### Phase 3: RAGAS KG QA 생성

- `generate_qa.py` CLI: `--method ragas|legacy`, `--build-kg-only`, `--reuse-kg`, `--num-personas`, `--query-dist`
- `_generate_qa_ragas()`: KG 구축(Node+default_transforms) → TestsetGenerator → SingleHop/MultiHop 분포
- KG 저장/로드: `_benchdata/ragas_knowledge_graph.json`
- 출력 포맷 하위 호환: 기존 필드(question, ground_truth, parent_id, source) 유지, 신규 필드(synthesizer_name, query_type, reference_contexts) 추가 전용

### 검증 결과

- Phase 1: COMPREHENSIVE 7개 메트릭 생성 확인, 기존 프리셋 회귀 없음, 가중치 합 1.0
- Phase 2: dry-run CLI 정상, 가중 점수 테이블 출력 확인, 잘못된 인자 거부
- Phase 3: 모든 RAGAS testset import 성공, 레거시 캐시 읽기 호환, 출력 포맷 양방향 호환

### 주의사항

- `NonLLMStringSimilarity`는 `pip install rapidfuzz` 필요 (없으면 자동 skip)
- RAGAS KG QA 생성은 API 비용 발생 (gpt-4o-mini, OpenAI Embeddings)
- `--method ragas`의 실제 E2E 테스트는 API 키 필요
