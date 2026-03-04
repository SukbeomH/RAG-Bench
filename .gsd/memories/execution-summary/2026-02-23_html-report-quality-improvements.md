---
title: "HTML 보고서 품질 개선 — 중앙값/이상치/RAGAS 탭/순수 Contextual Delta"
tags:
  - html-report
  - generate_html_report
  - rag_bench
  - bug-fix
  - data-analysis
type: execution-summary
created: 2026-02-23T18:30:00Z
contextual_description: "HTML 벤치마크 보고서 전반적 품질 개선: 레이턴시 중앙값 전환, IQR 이상치 감지, RAGAS ON/OFF 탭, 순수 Contextual 페어드 비교, NaN 과대추정 갭 표시"
keywords:
  - 중앙값
  - 이상치
  - IQR
  - RAGAS탭
  - Contextual
  - NaN
  - 가중점수
  - HTML보고서
  - 레이턴시
  - 페어드비교
related:
  - 2026-02-23_rag-bench-11-b1-b2-h-d-c.md
  - 2026-02-23_4-layer-rag-html.md
---

## HTML 보고서 품질 개선 — 중앙값/이상치/RAGAS 탭/순수 Contextual Delta

## 작업 개요
전문가 리뷰 기반으로 `generate_html_report.py` 품질 이슈 5건 해결 + e2e_report.md 생성 로직 버그 2건 수정.

## 수정 사항 (generate_html_report.py)

### 1. 레이턴시 전체 평균 → 중앙값 전환
- `_agg_latency()`: `avg_latency_ms = x.quantile(0.5)` (MEDIAN), `mean_latency_ms = "mean"` (보조 참고용)
- 차트/테이블/레이블 전체 "평균 레이턴시" → "중앙값 레이턴시"
- 이유: 5개 쿼리 기반 → 단일 이상치가 평균 크게 왜곡 (예: DS(KoSimCSE+korean_bm25) 696ms 이상치 → 평균 321ms, 중앙값 120ms)

### 2. IQR×1.5 이상치 감지 배지
- `_has_outlier()`: IQR×1.5 기준 이상치 탐지
- `_latency_table_html()`: ⚠️이상치 배지 + CV 툴팁 + 이상치 경고 박스
- 이상치 있는 전략 아래 평균 레이턴시 보조 표시

### 3. RAGAS 테이블 Contextual ON/OFF 탭 분리
- `_ragas_split_tables_html()`: Bootstrap tabs (OFF 30 / ON 30)
- NaN 값 → "N/A" 렌더링

### 4. 순수 1:1 Contextual 페어드 비교
- `_layer_contribution_html()`: regex 기반 base-key 추출 → OFF/ON 동일 조합 1:1 매칭
- 혼동 변수(FlashRank 풀링) 제거한 순수 Delta 표시
- 쌍별 승/패/무 카운트 표시

### 5. NaN 과대추정 갭 표시
- `_compute_weighted_scores_nan_penalized()`: NaN=0 보수적 점수 계산
- 각 NaN 전략에 "skipna 점수 vs NaN=0 점수" 갭 표시
- CASE2 버그 수정: 추가 컬럼(_reranker, _contextual 등) 필터링 후 scoring 함수 호출

## 수정 사항 (run_all_combos.py)

### Bug 1: 단계별 비중 계산 오류 (5824%)
- 원인: `total_s = rec.duration_s or 1` → `rec.duration_s=None` 시 총합=1초
- 결과: 58.2s / 1s × 100 = 5820% (e2e_report.md에서 5824% 표기됨)
- 수정: `phase_sum = sum(p.duration_s ...)` fallback 추가
  ```python
  phase_sum = sum(p.duration_s for p in tracker._phases if p.duration_s > 0)
  total_s = rec.duration_s or phase_sum or 1
  ```

### Bug 2: 레이턴시 표기 기준 불일치
- `_report_latency_section()`: "평균 레이턴시" → `p50_latency` 우선 사용
- `p50_latency` 없으면 `avg_latency` fallback
- 헤더 레이블도 동적으로 "중앙값 레이턴시" / "평균 레이턴시" 전환

## 검증
- 보고서 재생성 성공 (에러 없음, 2026-02-23 18:17:33)
- 전 섹션 Playwright 시각 확인: 요약/레이턴시/RAGAS탭/타이밍/결론
- 7개 엣지 케이스 테스트 통과 (이전 세션)

## 영향 파일
- `rag_bench/scripts/generate_html_report.py` — 주 변경 파일
- `rag_bench/scripts/run_all_combos.py` — 비중 계산 버그 + 레이턴시 표기 수정
