---
title: "FlashRank 영속화 + 레이어 필터 + full 벤치마크 실행"
tags:
  - rag_bench
  - flashrank
  - benchmark
  - execution-summary
type: execution-summary
created: 2026-02-24T00:00:00Z
contextual_description: "FlashRank /tmp 버그 수정, 레이어 필터 3개 추가, QA 20개 full 벤치마크 실행 중 (RAGAS 진행)"
keywords:
  - FlashRank
  - 영속화
  - 레이어필터
  - reranker-filter
  - sparse-filter
  - contextual-filter
  - append-results
  - full벤치마크
  - RAGAS
  - QA20개
related:
  - 2026-02-24_flashrank-tmp-cache-reboot-bug
  - 2026-02-23_e2e-report-phase-pct-bug
---

## FlashRank 영속화 + 레이어 필터 + full 벤치마크 실행

## 작업 개요
1. FlashRank 20개 전략 재부팅 후 실패 원인 분석 및 수정
2. 특정 전략 별도 실행 후 결과 통합을 위한 레이어 필터 기능 추가
3. QA 20개 (--max-qa-per-page 10) full 벤치마크 백그라운드 실행 중

## 1. FlashRank /tmp 버그 수정 (`combo/cache.py`)
- `MODELS_DIR` import 추가
- `get_flashrank_ranker()`에서 `cache_dir=str(MODELS_DIR / "flashrank")` 명시
- 모델 `rag_bench/_models/flashrank/ms-marco-MultiBERT-L-12/` 다운로드 완료 (98.7MB)

## 2. 레이어별 실행 필터 추가 (`scripts/run_all_combos.py`)
```
--reranker-filter   : none/colbert/flashrank 쉼표 구분
--sparse-filter     : korean_bm25/splade 쉼표 구분
--contextual-filter : none/contextual 쉼표 구분
```
기존 `--dense-filter` + `--append-results`와 조합해 전략 분리 실행 후 통합 가능:
```bash
python -m rag_bench.scripts.run_all_combos \
  --preset full --reranker-filter flashrank --append-results
```

## 3. Full 벤치마크 현황 (백그라운드 task: bca716c)
- QA 20개 (`--max-qa-per-page 10`) 재생성 완료
- Contextual 캐시 26건 히트 (LLM 비용 없음)
- FlashRank 20개 전략 실패 (이 시점은 수정 전 실행 — 다음 실행 시 해결됨)
- Pass 1 레이턴시 완료 (ColBERT ~5초/쿼리)
- Pass 2 RAGAS 진행 중: 6~7/40 완료, 전략당 ~230초, ETA ~136분
- RAGAS 모델: gpt-4o-mini + ada-002 (기본), core_only 프리셋 (4개 메트릭)

## 추가 분석
- RAGAS 비용 추정: ~$17–20 (gpt-4o-mini × 40전략 × 20쿼리 × 8호출)
- ada-002 → text-embedding-3-small 교체 시 임베딩 비용 80% 절감 가능 (미적용)

## 커밋
`7c08f31` fix+feat: FlashRank 모델 영속화 + 레이어별 실행 필터 추가
