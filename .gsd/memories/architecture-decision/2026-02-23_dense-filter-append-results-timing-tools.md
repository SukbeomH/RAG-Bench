---
title: "dense-filter / append-results 옵션 + combo_timing.csv + timing_report.py 도입"
tags:
  - architecture-decision
  - benchmark
  - cli-options
  - timing-analysis
  - partial-rerun
type: architecture-decision
created: 2026-02-23T10:00:00+09:00
contextual_description: "run_all_combos.py에 --dense-filter, --append-results 옵션 추가. combo_timing.csv로 조합별 소요 시간 기록. timing_report.py 독립 스크립트로 재실행 없이 기존 데이터에서 QA 스케일링 분석."
keywords:
  - dense-filter
  - append-results
  - combo_timing.csv
  - timing_report.py
  - qa-scaling
  - partial-rerun
  - run_all_combos
related:
  - 2026-02-23_full-bench-run-upstage-retry-setup
  - 2026-02-23_qa-scaling-120s-per-qa-48strategies
---

## 타이밍 분석 도구 아키텍처 결정 (2026-02-23)

### 배경
- Upstage 12개 조합 실패 → 전체 60개 재실행 불필요, 12개만 재실행하여 병합 필요
- 조합별 전체 소요 시간(빌드/Pass1/Pass2) 가시화 요구
- QA 수 결정을 위한 스케일링 분석 요구 (재실행 없이)

### 신규 CLI 옵션 (`run_all_combos.py`)

#### `--dense-filter`
```python
parser.add_argument("--dense-filter", type=str, default=None,
    help="실행할 dense 모델 필터 (쉼표 구분). 예: --dense-filter upstage,openai-large")
# 적용 위치: combos 생성 후 필터링
if getattr(args, "dense_filter", None):
    filter_models = [m.strip() for m in args.dense_filter.split(",")]
    combos = [c for c in combos if c.dense in filter_models]
```

#### `--append-results`
```python
parser.add_argument("--append-results", action="store_true",
    help="기존 latency/RAGAS CSV에 결과를 병합(append)하여 저장")
# 병합 로직: 기존 CSV에서 현재 전략 행 제거 → 신규 행 concat → 저장
```

### 신규 파일

#### `combo_timing.csv` (자동 생성)
위치: `_benchdata/combo_timing.csv`
컬럼:
| 컬럼 | 설명 |
|------|------|
| label | 전략 레이블 |
| dense, sparse, reranker, llm_support | 레이어 값 |
| build_s | 인덱스 빌드 시간(초) |
| pass1_s | Pass 1 전체 시간(초) |
| pass1_s_per_qa | QA 1개당 Pass 1 시간(초) |
| pass2_s | Pass 2 RAGAS 시간(초) |
| pass2_s_per_qa | QA 1개당 Pass 2 시간(초) |
| total_s | 전체 소요 시간(초) |
| n_queries | 실행된 QA 수 |

#### `rag_bench/scripts/timing_report.py` (독립 스크립트)
```bash
# 사용법
python -m rag_bench.scripts.timing_report
python -m rag_bench.scripts.timing_report --latency-csv path/to/latency.csv
python -m rag_bench.scripts.timing_report --qa-range 10,20,50,100
```

데이터 소스:
- `_benchdata/all_combos_latency.csv` → Pass 1 레이턴시
- `_benchdata/run_history/latest.json` → 빌드 시간 + RAGAS 단계 시간
- `_benchdata/all_combos_ragas.csv` → RAGAS 평가 전략 수

### 수정된 파일
- `rag_bench/scripts/run_all_combos.py`: `--dense-filter`, `--append-results`, `_build_combo_timing_df()` 추가
- `rag_bench/runner.py`: `_eval_times` dict + 전략별 RAGAS 타이밍 측정 추가
- `rag_bench/utils/report.py`: `print_combo_timing_table()`, `print_qa_scaling_table()` 추가
- `rag_bench/scripts/timing_report.py`: 신규 생성
