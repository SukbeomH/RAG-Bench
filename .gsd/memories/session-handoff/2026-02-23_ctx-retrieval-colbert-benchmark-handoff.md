---
title: "Contextual Retrieval 최적화 + ColBERT 스레드 수정 + 벤치마크 재시작"
tags:
  - handoff
  - session
  - contextual-retrieval
  - colbert
  - benchmark
  - thread-safety
  - worktree
type: session-handoff
created: 2026-02-23T16:30:00+09:00
contextual_description: "PLAN-1.1+1.2 실행(enrich_only 1회 호출), pre_enriched BM25 fix, Pass2 병렬화/진행로그, ColBERT 스레드 충돌 수정. 벤치마크 PID 29936 Pass1 진행 중."
keywords:
  - enrich_only
  - pre_enriched
  - ContextualRetrievalStrategy
  - _COLBERT_INFERENCE_LOCK
  - parallel_eval
  - pass2-workers
  - worktree-ctx-retrieval-optimization
  - BM25 fit
related:
  - 2026-02-23_combo-reorg-colab-optimization-handoff
  - 2026-02-23_handoff-worktree-merge
---

## Contextual Retrieval 최적화 + ColBERT 스레드 수정 + 벤치마크 재시작

### 현재 상태 (세션 종료 시점)
- **벤치마크 PID 29936** 실행 중 (Pass 1 진행)
  - 명령: `python rag_bench/scripts/run_all_combos.py --preset full --top_n 60 --layers --pass1-workers 4 --pass2-workers 4`
  - 로그: `/tmp/bench_v3.log`
  - Pass 1 완료 후 Pass 2 자동 시작 (parallel_eval=4, RunConfig max_workers=16)
- **브랜치**: `worktree-ctx-retrieval-optimization` (미머지)
  - master 대비 2커밋 앞: `be71acb` + `2d86a2f`

### 이번 세션 구현 내용

#### 1. PLAN-1.1+1.2: enrich_only 1회 호출 (be71acb)
- `ContextualRetrievalStrategy.enrich_only(child_chunks)` 추가
  - `_enrich_chunks()` 래퍼, LLM prefix 부착만 하고 인덱싱 없음
- `cache.get_or_build_contextual(pre_enriched=None)` 파라미터 추가
  - 캐시 히트 경로: `fit_docs = pre_enriched if pre_enriched is not None else child_chunks`
  - 신규 인덱스 경로: `pre_enriched` 있으면 `ctx_base.index(pre_enriched)` 직접 실행
- `builder.build_strategy_from_spec(pre_enriched=None)` 전달 체인
- `run_all_combos.py` Step 2.5 삽입: `_has_contextual` 체크 후 1회만 enrich_only 실행

#### 2. pre_enriched BM25 fit 버그 수정 (2d86a2f)
- **버그**: 캐시 히트 경로에서 BM25를 plain `child_chunks`로 fit → 인덱스 어휘 불일치
- **수정**: `fit_docs = pre_enriched if pre_enriched is not None else child_chunks`
- 파일: `rag_bench/combo/cache.py`

#### 3. Pass 2 병렬화 (5769e14) — master
- `rag_bench/evaluation/evaluator.py`: `RunConfig(max_workers=16, timeout=180)` 추가
- `rag_bench/runner.py`: `parallel_eval` 파라미터 + `ThreadPoolExecutor` 병렬 평가
- `rag_bench/scripts/run_all_combos.py`: `--pass2-workers N` CLI 인자

#### 4. Pass 2 진행률 로그 (fae0faf) — master
- `runner.py evaluate()`: `[N/total] 전략명  Xs | ETA Ym Zs` 형식 로그
- `_completed_count`, `_t_pass2_start` 클로저 변수 활용
- 완료 시: `Pass 2 완료 — N개 전략, 총 Xm Ys`

#### 5. ColBERT 스레드 충돌 수정 (ebad011) — master
- **오류**: `--pass1-workers 4` 병렬 실행 시 `tensor size mismatch` + `to() invalid args`
- **원인**: PyLate ColBERT `encode()` + `rank.rerank()` 스레드 비안전 (PyTorch 배치 텐서 공유)
- **수정**: `colbert_rerank.py`에 모듈 레벨 `_COLBERT_INFERENCE_LOCK = threading.Lock()` 추가
  - `retrieve()` 내 encode + rerank를 `with _COLBERT_INFERENCE_LOCK:` 으로 직렬화
- **검증**: FlashRank(ONNX) 스레드 안전, DenseSparse `_vocab_lock` 이미 존재 확인

### 다음 세션 할 일
1. **벤치마크 완료 대기**: PID 29936 — Pass 2까지 완료 후 RAGAS CSV 생성 확인
2. **HTML 보고서 재생성**: `python rag_bench/scripts/generate_html_report.py` 실행
3. **worktree → master 머지**: `worktree-ctx-retrieval-optimization` 브랜치의 `be71acb`, `2d86a2f` 머지
4. **벤치마크 결과 분석**: Contextual 전략 성능 비교 (pre_enriched 효과)

### 핵심 파일 위치
| 파일 | 역할 |
|------|------|
| `rag_bench/strategies/contextual_retrieval.py` | `enrich_only()` 메서드 |
| `rag_bench/combo/cache.py` | `pre_enriched` param + BM25 fit fix |
| `rag_bench/combo/builder.py` | `pre_enriched` 전달 체인 |
| `rag_bench/scripts/run_all_combos.py` | Step 2.5 + `--pass2-workers` |
| `rag_bench/strategies/colbert_rerank.py` | `_COLBERT_INFERENCE_LOCK` |
| `rag_bench/runner.py` | `parallel_eval` + ETA 로그 |
| `.gsd/PATTERNS.md` | Thread Safety / Benchmark 패턴 12개 |
