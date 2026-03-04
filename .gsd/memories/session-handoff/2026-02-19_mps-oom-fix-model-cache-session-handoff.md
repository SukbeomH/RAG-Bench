# Session Handoff: MPS OOM 수정 + 모델 캐시 + 성능 최적화 + RAGAS 리서치

## Date: 2026-02-19
## Branch: main

## What Was Done

### 1. 72개 벤치마크 실행 → MPS OOM 실패 → 수정 완료
- `--preset full --top_n 10 --layers` 옵션으로 실행 → MPS OOM (exit code 144)
- 원인: `_detect_device()`가 MPS 자동 선택 + 72개 전략 각각 ColBERT 새로 로드
- **수정**: ColBERT/FlashRank CPU 강제, 싱글톤 캐시, 메모리 해제 로직 추가
- **커밋**: `cb5165c` fix: MPS OOM 해결

### 2. HF 모델 로컬 캐시 구현
- `config.py`: MODELS_DIR, REQUIRED_HF_MODELS (6종), ensure_model_cache()
- `prefetch_models.py`: 신규 프리페치 스크립트 (--status, --force)
- 검증: 6개 모델 심링크 정상 생성
- **커밋**: `f684e28` feat: HF 모델 로컬 캐시

### 3. RAGAS Testset Generation v2 리서치
- `docs/research/ragas_testset_generation_v2_research.md` 작성 (304줄)
- Knowledge Graph 기반 진화적 QA 생성 파이프라인 분석
- **커밋**: `8460952` docs: RAGAS v2 리서치 문서

### 4. 벤치마크 실행 최적화 (HIGH 4개)
- **FlashRank 싱글톤**: IndexCacheManager.get_flashrank_ranker() — 24회 → 1회 ONNX 로드
- **Pass 1→2 결과 재사용**: BenchmarkRunner.inject_results() — 재검색 제거
- **Answer 생성 병렬화**: ThreadPoolExecutor(8) + lazy LLM 초기화
- **SPLADE 배치 처리**: embed_documents() batch_size=32
- **검증**: `--preset quick --pass1-only` 4개 전략 × 20 쿼리 = 80회 성공 (MPS OOM 없음)
- **커밋**: `bf321b6` perf: 벤치마크 실행 최적화

### 5. 코드베이스 최적화 분석 (MEDIUM/LOW 미적용)
- cleanup() 인덱스 보존 옵션 (MEDIUM)
- ColBERT shared_model cleanup 정확화 (MEDIUM)
- 전략 실행 병렬화 (MEDIUM)
- contextual_cache 중간 저장 (MEDIUM)
- parent_store 단일 JSON (LOW)
- dense 모델 차원 정적 테이블 (LOW)

## What Needs To Be Done Next
1. **72개 조합 풀 벤치마크 재실행**: `--preset full --top_n 10 --layers` (MPS 수정 + 최적화 검증)
2. **QA 고도화**: generate_qa.py에 RAGAS v2 방식 `--method ragas` 구현
3. **evaluation 확장**: Extended 메트릭 + per-sample 점수
4. **MEDIUM 최적화 적용** (선택): cleanup 인덱스 보존, 전략 실행 병렬화 등
5. **벤치마크 시각화 갱신**: bench_visualize.ipynb 72개 조합 결과 대응

## Critical Notes
- `hs_err_pid28354.log`는 Java 에러 로그 → .gitignore 추가 고려
- FlashRank rerank 레이턴시가 ~1.8초로 높음 (CPU ONNX 추론) — 벤치마크 결과 해석 시 참고
- SPLADE 배치 처리는 quick 프리셋(fastembed_bm25)에서는 경로를 타지 않음 — full에서 검증 필요

## Key Files
- `rag_bench/config.py` — 전역 설정 + 모델 캐시 + MPS 워크어라운드
- `rag_bench/runner.py` — BenchmarkRunner (inject_results, 병렬 answer gen, lazy LLM)
- `rag_bench/scripts/run_all_combos.py` — 72개 조합 벤치마크 (FlashRank 싱글톤, Pass 재사용)
- `rag_bench/strategies/dense_sparse.py` — SPLADE 배치 처리
- `rag_bench/strategies/colbert.py` — ColBERT 전략 (CPU 전용)
- `rag_bench/strategies/colbert_rerank.py` — ColBERT 리랭킹 (shared_model)
- `rag_bench/strategies/flashrank_rerank.py` — FlashRank 리랭킹 (shared_ranker)

## Commits This Session
```
bf321b6 perf: 벤치마크 실행 최적화 — FlashRank 싱글톤 + Pass 결과 재사용 + LLM 병렬화 + SPLADE 배치
8460952 docs: RAGAS v2 리서치 문서 + MEMORY 세션 기록 갱신
f684e28 feat: HF 모델 로컬 캐시 — 6종 모델 심링크 + prefetch 스크립트
cb5165c fix: MPS OOM 해결 — ColBERT CPU 강제 + 싱글톤 캐시 + 메모리 해제
```
