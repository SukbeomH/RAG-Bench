# RAG Bench Local

로컬 Jupyter Notebook에서 60개 RAG 전략 조합을 벤치마크합니다.

> `rag_bench_colab`에서 마이그레이션됨. Google Colab 의존성(Drive, Secrets)을 로컬 환경으로 교체.

## Quick Start

### 1. 환경 설정

```bash
# 의존성 설치 (프로젝트 루트에서)
uv sync
# 또는
pip install -r rag_bench_local/requirements_core.txt
pip install -r rag_bench_local/requirements_viz.txt  # 시각화 (선택)

# API Key 설정 (.env 파일)
echo "OPENAI_API_KEY=sk-..." > .env
# Upstage 전략 사용 시
echo "UPSTAGE_API_KEY=up_..." >> .env
```

### 2. Jupyter 노트북 실행

```bash
jupyter lab
# rag_bench_local/rag_benchmark.ipynb 열기
```

### 3. 코드 실행 예시

```python
# 1. 환경 초기화
from rag_bench_local.local_config import init_local
init_local(qdrant_mode="local")

# 2. 러너 생성
from rag_bench_local.local_runner import LocalBenchmarkRunner
runner = LocalBenchmarkRunner(preset="quick", k=3, top_n=4)

# 3. QA 데이터셋 생성 (최초 1회 또는 force=True 시)
runner.prepare_qa()  # 기존 data/docs/*.md 사용
# runner.prepare_qa(sample_pages=True)  # docs/*.pdf 페이지 샘플링

# 4. 데이터 로드 + 청킹
child_chunks, parent_pairs, queries, ground_truths = runner.prepare_data()

# 5. 벤치마크 실행
combos = runner.generate_combos()
lat_df  = runner.run_pass1(combos, queries, child_chunks, parent_pairs)
ragas_df = runner.run_pass2(lat_df, combos, queries, ground_truths, child_chunks, parent_pairs)

# 6. 결과 저장 (로컬 + HTML 보고서)
runner.export_results(lat_df, ragas_df)
```

### 4. 시각화

```python
from rag_bench_local.visualizer import (
    plot_latency_comparison,
    plot_ragas_radar,
    plot_ragas_heatmap,
    plot_latency_vs_quality,
    plot_layer_contribution,
    plot_ablation_waterfall,
    create_summary_table,
    display_styled_table,
)

# 레이턴시 비교 차트
plot_latency_comparison(lat_df)

# RAGAS 레이더 차트
plot_ragas_radar(ragas_df, top_n=5)

# 리더보드 테이블
summary = create_summary_table(lat_df, ragas_df)
display_styled_table(summary)
```

## 실행 흐름

```
  로컬 Jupyter 세션
  │
  ├─ [Step 1] init_local(qdrant_mode="local")
  │    ├─ .env 파일에서 API Key 로드 (python-dotenv)
  │    ├─ HF_HOME 설정 (rag_bench/_models 또는 시스템 기본값)
  │    ├─ patch_rag_bench_config()  ← 경로 설정
  │    └─ _setup_korean_font()
  │
  ├─ [Step 2] LocalBenchmarkRunner(preset, k, top_n, ...)
  │
  ├─ [Step 3] runner.prepare_qa(...)
  │    ├─ 캐시 확인: _benchdata/qa_dataset.json 존재 시 스킵
  │    ├─ create_parent_child_chunks()
  │    └─ _generate_qa_ragas() → _benchdata/qa_dataset.json
  │
  ├─ [Step 4] runner.prepare_data()
  │    └─ qa_dataset.json 로드 → (child_chunks, parent_pairs, queries, gts)
  │
  ├─ [Step 5] runner.generate_combos() + run_pass1()
  │    ├─ 프리셋별 조합 수: quick=2, standard=20, full=60
  │    ├─ 체크포인트: _benchdata/checkpoints/ (중단/재개 지원)
  │    └─ → lat_df (레이턴시 결과)
  │
  ├─ [Step 6] runner.run_pass2(lat_df, ...)
  │    └─ → ragas_df (RAGAS 점수)
  │
  └─ [Step 7] runner.export_results(lat_df, ragas_df)
       ├─ _benchdata/results/latency.csv
       ├─ _benchdata/results/ragas.csv
       └─ _benchdata/results/report.html
```

## 프리셋

| 프리셋 | 조합 수 | Dense | Sparse | 예상 시간 | API 비용 |
|--------|---------|-------|--------|----------|---------|
| `quick` | 2 | bge-m3 (1종) | korean_bm25 (1종) | ~10분 | ~$0.3 |
| `standard` | 20 | HF 3종 + 유료 2종 | 2종 | ~45분 | ~$2 |
| `full` | 60 | 5종 | 2종 | ~3시간 | ~$5 |

## 주요 파라미터

### LocalBenchmarkRunner

```python
runner = LocalBenchmarkRunner(
    preset="quick",               # quick | standard | full
    k=3,                          # 검색 결과 수
    top_n=10,                     # Pass 2에서 RAGAS 평가할 상위 전략 수
    qdrant_mode="local",          # local | memory
    device=None,                  # cuda | cpu | None (자동 감지)
    parallel_queries=0,           # 쿼리 병렬화 (0=비활성)
    reindex=False,                # True: 기존 인덱스 삭제 후 재구축
    metric_preset="core_only",    # core_only (4) | comprehensive (7) | full (11+)
    scoring_profile="balanced",   # balanced | precision_critical | speed_critical
)
```

## Qdrant 모드

| 모드 | 설명 | 영속성 |
|------|------|--------|
| `local` | `_benchdata/qdrant_db_*` (로컬 파일) | 영속 |
| `memory` | 인메모리 (`:memory:`) | 프로세스 종료 시 삭제 |

## 디렉토리 구조

```
rag_bench_local/
├── README.md                  # 이 파일
├── rag_benchmark.ipynb        # 메인 Jupyter 노트북
├── local_config.py            # 로컬 환경 설정 + rag_bench 패치
├── local_runner.py            # 체크포인트 지원 벤치마크 러너
├── visualizer.py              # 시각화 유틸리티 (12+차트)
├── requirements_core.txt      # 핵심 의존성
├── requirements_colab.txt     # 전체 의존성 (시각화 포함)
├── requirements_viz.txt       # 시각화 의존성 (선택)
└── data/
    └── docs/                  # 벤치마크 대상 마크다운 문서
```

## 체크포인트

벤치마크 중단 시 로컬 파일시스템에 체크포인트가 저장됩니다.
동일 설정으로 다시 실행하면 완료된 전략은 건너뜁니다.

저장 위치: `rag_bench/_benchdata/checkpoints/`

## colab → local 변경 사항

| 항목 | Colab | Local |
|------|-------|-------|
| API Key | Colab Secrets | `.env` 파일 (python-dotenv) |
| 모델 캐시 | Google Drive | 로컬 `rag_bench/_models` |
| 체크포인트 | Google Drive | 로컬 `_benchdata/checkpoints/` |
| 결과 저장 | Google Drive | 로컬 `_benchdata/results/` |
| Qdrant 모드 | ephemeral/drive/memory | local/memory |
| 디바이스 | CUDA (T4 GPU) | 자동 감지 (CUDA/CPU) |
| 환경 초기화 | `init_colab()` | `init_local()` |
| 러너 클래스 | `ColabBenchmarkRunner` | `LocalBenchmarkRunner` |
