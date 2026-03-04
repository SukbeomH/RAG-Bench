---
title: "e2e_report.md 단계 비중 5824% 버그 — rec.duration_s=None fallback"
tags:
  - bug
  - root-cause
  - run_all_combos
  - e2e_report
type: root-cause
created: 2026-02-23T18:30:00Z
contextual_description: "e2e_report.md 단계별 비중이 5824%로 표시되는 버그 — rec.duration_s=None일 때 total_s=1 fallback으로 인해 발생"
keywords:
  - 비중
  - duration_s
  - total_s
  - phase_sum
  - e2e_report
  - run_all_combos
related:
  - 2026-02-23_html-report-quality-improvements.md
---

## e2e_report.md 단계 비중 5824% 버그 — rec.duration_s=None fallback

## 증상
e2e_report.md의 "단계별 소요 시간" 섹션에서 비중이 올바른 % 대신 수천 %로 표기됨:
- strategy_build_and_indexing: 58.2s | **5824.0%**
- pass1_latency: 605.2s | **60517.0%**
- pass2_ragas: 1622.4s | **162244.0%**

## 근본 원인
`run_all_combos.py` 857-858번 줄:
```python
if tracker._phases:
    total_s = rec.duration_s or 1   # ← 버그: rec.duration_s=None이면 total_s=1
    pct = p.duration_s / total_s * 100
```
`rec.duration_s`(전체 런 소요시간)가 `None`으로 설정되어 있어 fallback으로 `total_s=1`(초) 사용.
결과: `58.2 / 1 * 100 = 5820%` (실제 기록값과 1~2% 차이는 부동소수점)

## 수정
```python
phase_sum = sum(p.duration_s for p in tracker._phases if p.duration_s > 0)
total_s = rec.duration_s or phase_sum or 1
```
`rec.duration_s`가 None이면 각 phase의 duration_s 합산값으로 fallback.

## 재발 방지
- `rec.duration_s`는 런 종료 시점에 저장되는데 일부 경로에서 누락될 수 있음
- phase_sum fallback으로 항상 유효한 total 보장
