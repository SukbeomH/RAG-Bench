---
title: "Colab 동기화: MetricPreset, RunTracker, 가중 점수 통합"
tags:
  - colab
  - sync
  - metric-preset
  - run-tracker
  - weighted-score
type: execution-summary
created: 2026-02-19T04:31:15Z
contextual_description: "rag_bench_colab 최신화 — MetricPreset/ScoringProfile/RunTracker/가중점수를 Colab 환경에 통합한 실행 기록"
keywords:
  - colab sync
  - MetricPreset
  - ScoringProfile
  - RunTracker
  - weighted score
  - rag_bench_colab
related:
  - 2025-02-17_ragas-kg-qa-evaluation-report
---

## Colab 동기화: MetricPreset, RunTracker, 가중 점수 통합

## Colab 동기화: MetricPreset, RunTracker, 가중 점수 통합

### 작업 내용
rag_bench/의 최신 기능을 rag_bench_colab/에 동기화 완료.

### 변경 파일 (4건)
- **colab_runner.py**: MetricPreset/ScoringProfile 파라미터 추가, RunTracker lazy init, run_pass2() 가중 점수 출력, reports 프로퍼티, get_run_record() 메서드
- **colab_visualizer.py**: display_weighted_scores() 함수 추가, display_dashboard()에 가중 점수 섹션(#6) 통합
- **rag_benchmark.ipynb**: METRIC_PRESET/SCORING_PROFILE 설정 셀, RunTracker 시각화 셀, Layer Contribution 셀 추가
- **README.md**: Colab 뱃지 URL 수정(autorag→RAG-Bench), 메트릭 프리셋/스코어링 프로파일 테이블 문서화

### 핵심 패턴
- Colab 환경은 rag_bench 패키지를 직접 import하므로, colab_runner가 MetricPreset enum과 SCORING_PROFILES dict를 가져와서 사용
- RunTracker는 lazy init (_ensure_tracker)로 Colab 환경 정보(platform, GPU)를 자동 수집
- requirements_colab.txt는 새 의존성 불필요 (기존 ragas, pandas로 충분)

### 브랜치
feature/sync-colab-with-rag-bench (커밋: b0d78ed)
