# Session: 통합 벤치마크 스크립트 + 전체 9종 비교 + 시각화 노트북

**Date**: 2026-02-11
**Branch**: main
**Status**: Completed

## Summary
rag_bench 패키지를 독립 공유 가능하도록 구조화하고, 벤치마크 스크립트 3종 + 시각화 노트북을 구현. 전체 9종 전략(DenseSparse 4종 + ColBERT + ColBERTRerank 4종)에 대해 RAGAS 평가 포함 E2E 벤치마크 실행 완료.

## Changes Made

### New Files
- `rag_bench/scripts/generate_qa.py` — QA 데이터셋 자동 생성 (GPT-4o-mini, 해시 캐싱)
- `rag_bench/scripts/run_bench.py` — 3종 통합 벤치마크 + RAGAS
- `rag_bench/scripts/run_all_combos.py` — DenseSparse 6종 + ColBERT + ColBERTRerank 전체 비교
- `rag_bench/scripts/bench_visualize.ipynb` — 7종 시각화 차트 노트북
- `rag_bench/scripts/__init__.py`
- `rag_bench/docs/.gitkeep` + 2개 .md 문서
- `rag_bench/pyproject.toml`, `uv.lock`, `.python-version`, `docker-compose.yml` (독립 공유용 복사)

### Modified Files
- `rag_bench/config.py` — PACKAGE_ROOT, BENCH_DOCS_DIR, BENCH_DATA_DIR 추가
- `rag_bench/README.md` — 전면 개편 (4종 전략 상세, 스크립트 사용법, 독립 공유 안내)
- `.gitignore` — `rag_bench/_benchdata/` 추가
- `pyproject.toml` — matplotlib 의존성 추가

## Benchmark Results (9 strategies, 2 QA, --skip_paid)

### Speed Winner: DS4-MiniLM (107ms avg)
### Quality Winner: ColBERT (Faithfulness=1.0, Context Precision=1.0, Context Recall=1.0)
### Best Tradeoff: ColBERTRerank on E5+SPLADE (RAGAS avg 0.92, ~12s)

## Visualization Notebook Enhancements
- bench_visualize.ipynb 전체 8개 마크다운 헤더 셀에 **상세 한국어 해석 가이드** 추가:
  - 전략 분류 테이블 + RAGAS 메트릭 해설 (header)
  - 레이턴시 차트 읽는 법 + 워밍업 주의점 (latency-header)
  - RAGAS Grouped Bar 해석 포인트 — 리랭킹 효과 분석 (ragas-bar-header)
  - 레이더 차트 면적/형상 기반 품질 판단법 (radar-header)
  - 품질-속도 Scatter: 파레토 프론티어, SLA 기준 전략 선택 (tradeoff-header)
  - 히트맵 행/열 방향 분석법 + 약점 식별 (heatmap-header)
  - 쿼리별 레이턴시: 일관성/의존성 패턴 분석 (per-query-header)
  - 종합 순위표: 최종 전략 선택 기준 3가지 (indexing-header)

## Key Decisions
- 패키지 내부에 docs/, scripts/, _benchdata/ 배치하여 단독 공유 가능 구조
- run_all_combos.py에 실패 내성(try/except) 적용으로 부분 실패 시에도 성공한 전략만 벤치마크 계속
- matplotlib 의존성 추가 (시각화용)
