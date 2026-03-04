---
title: "QA 스케일링 인사이트 — QA 1개당 ~120s (48전략, 순차 기준)"
tags:
  - pattern-discovery
  - benchmark
  - qa-scaling
  - timing
  - performance
type: pattern-discovery
created: 2026-02-23T10:00:00+09:00
contextual_description: "60개 조합 벤치마크에서 QA 1개당 Pass1 110.10s + Pass2 9.78s = ~120s 소요. QA 10~20개가 속도와 통계적 신뢰성의 균형점. timing_report.py로 재실행 없이 분석 가능."
keywords:
  - qa-scaling
  - 110s-per-qa
  - pass1-rate
  - pass2-rate
  - timing-report
  - benchmark-cost
related:
  - 2026-02-23_dense-filter-append-results-timing-tools
---

## QA 수별 벤치마크 소요 시간 인사이트

### 측정 기준
- **조합**: 60개 (Dense 5종 × Sparse 2종 × Mode 6종)
- **Pass 1 전략**: 48개 (Upstage 12개 제외, OpenAI 등 48개 성공)
- **Pass 2 전략**: 24개 (상위 전략 RAGAS 평가 대상)
- **실행 기준**: 순차 실행 (--pass1-workers 미사용)

### 실측 수치 (QA=20 기준)
| 항목 | 값 |
|------|-----|
| QA 1개당 Pass 1 (전 전략 합산) | 110.10s |
| 전략 평균 Pass 1 / QA | 2.29s |
| QA 1개당 Pass 2 (RAGAS 전 전략) | 9.78s |
| 전략 평균 Pass 2 / QA | 0.41s |
| **QA 1개당 총 소요 (Pass1+Pass2)** | **~120s** |
| 빌드 시간 (고정, QA 무관) | ~35s |

### QA 수별 예상 소요 시간
| QA 수 | 예상 시간 |
|-------|----------|
| 5 | 약 10분 |
| 10 | 약 20분 |
| 20 | 약 66분 (현재) |
| 50 | 약 2.7시간 |
| 100 | 약 5.3시간 |

### 권장 QA 수
- **QA 10~20개**: 속도(10~66분)와 통계적 신뢰성의 균형점
- **QA 5개 이하**: 너무 빠르지만 통계적으로 불안정
- **QA 50개 이상**: 초기 탐색보다는 최종 검증 단계에 적합

### 분석 방법 (재실행 불필요)
```bash
# 기존 결과 데이터에서 분석
python -m rag_bench.scripts.timing_report

# 특정 QA 범위 지정
python -m rag_bench.scripts.timing_report --qa-range 5,10,20,30,50
```

데이터 소스: `all_combos_latency.csv` + `run_history/latest.json` + `all_combos_ragas.csv`
