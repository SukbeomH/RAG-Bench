---
title: "rag_bench 기술부채 11건 해결 — B1/B2 버그 + H/D/C 개선"
tags:
  - rag_bench
  - tech-debt
  - refactoring
  - bug-fix
type: execution-summary
created: 2026-02-23T08:07:03Z
contextual_description: "rag_bench 기술부채 B1/B2 버그 + H1-3 하드코딩 + D1-4 중복코드 + C1-4 복잡도 총 11건 해결"
keywords:
  - 기술부채
  - 리팩토링
  - 버그수정
  - 하드코딩
  - 중복코드
  - 복잡도
  - httpx팩토리
  - BM25
  - 레이어정의
  - RAGAS
---

## rag_bench 기술부채 11건 해결 — B1/B2 버그 + H/D/C 개선

## 작업 개요
rag_bench 코드베이스 분석 후 기술부채 11건을 2회 세션에 걸쳐 수정.

## 1차: 버그 수정 (B1, B2)
- B1: _print_layer_analysis_preview() 레이어 문자열 매칭 버그
  - layer_name 전체 문자열로 비교해야 하는데 부분 문자열 비교 → count 항상 0
  - 수정: get_val 람다를 튜플에 추가하고 문자열 비교 제거
  - 파일: scripts/run_all_combos.py:687-696
- B2: contextual_retrieval.py 타입 힌트 불일치
  - parent_pairs: List[...] = None → Optional[List[...]] = None
  - 파일: strategies/contextual_retrieval.py:78

## 2차: H/D/C 개선 11건

### H (하드코딩) 3건
- H1: config.py에 QDRANT_COLLECTION_NAME 상수 추가, DenseSparseStrategy에 collection_name 파라미터화
- H2: generate_qa.py의 gpt-4o-mini 하드코딩 → DEFAULT_CONTEXTUAL_LLM 참조
- H3: CacheConfig 기본값을 DEFAULT_COLBERT_MODEL / DEFAULT_FLASHRANK_MODEL 상수로 통일

### D (중복 코드) 4건
- D1: make_http_client() / make_async_http_client() 팩토리를 config.py에 추가, runner/evaluator/generate_qa 3곳 교체
- D2: IndexCacheManager._restore_bm25_vocab() 헬퍼 추출, get_or_build + get_or_build_contextual 공유
- D3: DenseSparseStrategy._dense_short 프로퍼티 추가, name/description에서 재사용
- D4: _LAYER_DEFINITIONS 모듈 상수 추출, 3개 함수가 공유

### C (복잡도) 4건
- C1: _generate_report() 140줄 → _report_env_section / _report_latency_section / _report_timing_section / _report_ragas_section 4개 헬퍼로 분리
- C2: _step5_pass2() → _select_eval_strategies() / _save_ragas_results() 분리
- C3: generate_qa.py 내부 함수 _compute_effective_num_qa / _generate_qa_ragas → 공개 API compute_effective_num_qa / generate_qa_ragas 승격
- C4: UpstageEmbedStrategy.get_raw_token_usages() 공개 메서드 추가, 비공개 속성 직접 접근 제거

## 영향 파일
config.py, dense_sparse.py, combo/cache.py, runner.py, evaluation/evaluator.py,
scripts/generate_qa.py, strategies/upstage_embed.py, scripts/run_all_combos.py

## 검증
python -m py_compile 전체 파일 통과 (구문 오류 없음)
