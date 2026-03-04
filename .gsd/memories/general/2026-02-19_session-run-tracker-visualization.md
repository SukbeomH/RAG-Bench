---
title: "RunTracker 수행 이력 추적 시스템 + 시각화 통합"
tags: [run-tracker, visualization, token-tracking, benchmarking]
type: general
created: 2026-02-19
contextual_description: "벤치마크 수행 이력 추적 모듈(run_tracker.py) 신규 구현, 토큰 사용량 추적, 시각화 4종 함수 추가, bench_visualize.ipynb 섹션 10 추가, 비중% 전면 추가"
keywords: [RunTracker, TokenUsage, track_openai_tokens, StrategyTiming, PhaseTime, BenchmarkRunRecord, plot_run_info, plot_phase_timeline, plot_build_times, plot_token_usage]
related: [run_all_combos.py, generate_qa.py, colab_visualizer.py, bench_visualize.ipynb]
---

## 세션 요약

### 구현 내용
1. **`rag_bench/run_tracker.py`** (NEW, 448줄): 벤치마크 수행 이력 추적 모듈
   - `collect_platform_info()`: OS, CPU, RAM, GPU, Apple Silicon, git commit 수집
   - `TokenUsage` + `track_openai_tokens()`: LangChain get_openai_callback 래퍼
   - `StrategyTiming`: 전략별 빌드/쿼리 타이밍 + RAGAS 점수
   - `RunTracker`: phase() 컨텍스트 매니저, JSON 저장, latest.json 심링크

2. **벤치마크 스크립트 통합**
   - `run_all_combos.py`: 모든 단계를 tracker.phase()로 래핑, 토큰 추적
   - `generate_qa.py`: QA 생성 시 RunTracker + 토큰 추적
   - e2e_report.md: 실행 환경, 단계별 시간(비중%), 토큰 테이블

3. **시각화 4종** (`colab_visualizer.py`)
   - `plot_run_info()`: 실행 정보 요약 카드
   - `plot_phase_timeline()`: 단계별 소요 시간 막대
   - `plot_build_times()`: 전략별 빌드 시간 (LLM 사용 구분)
   - `plot_token_usage()`: 토큰 파이차트 + prompt/completion 비율

4. **bench_visualize.ipynb**: 섹션 10 "수행 이력" 추가

### 핵심 패턴
- `with tracker.phase("name"):` — 단계별 시간 자동 측정
- `with track_openai_tokens() as usage:` — LLM 토큰 자동 추적
- `_benchdata/run_history/run_{YYYYMMDD_HHMMSS}.json` + `latest.json` 심링크
- 모든 출력에 전체 소요시간 대비 비중(%) 표시
