# RAG Bench Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SukbeomH/RAG-Bench/blob/main/rag_bench_colab/rag_benchmark.ipynb)

Google Colab T4 GPU에서 60개 RAG 전략 조합을 벤치마크합니다.

## Quick Start

1. 위 Colab 뱃지 클릭 (또는 `rag_benchmark.ipynb`를 Colab에 업로드)
2. **런타임 → 런타임 유형 변경 → GPU (T4)** 선택
3. **런타임 버전: `2026.01` 선택** (flash-attn 호환을 위해 권장 — 아래 참고)
4. Colab Secrets에 `OPENAI_API_KEY` 등록
5. 노트북 셀 순서대로 실행

### 런타임 버전 선택 (권장: 2026.01)

| 런타임 | PyTorch | flash-attn 2.8.3 | 비고 |
|--------|---------|-------------------|------|
| `2026.01` | 2.9.0 | ✅ 지원 | **권장** — flash-attn 성능 최적화 사용 가능 |
| `2026.02+` | 2.10.0+ | ⚠️ 미지원 | flash-attn 건너뜀 (정확도 무관, 속도만 영향) |
| `2025.10` | 2.8.0 | ✅ 지원 | 이전 버전 |

**런타임 버전 변경 방법:**
`런타임 → 런타임 유형 변경 → 런타임 버전` 드롭다운에서 `2026-01` 선택

> flash-attn 없이도 벤치마크는 정상 동작합니다. 단, ColBERT reranker 속도가 약간 느려질 수 있습니다.

### 실행 흐름 다이어그램

```
  Google Colab 세션
  │
  ├─ [Step 1] init_colab(qdrant_mode="ephemeral")
  │    ├─ Google Drive 마운트 (/content/drive)
  │    ├─ Colab Secrets → OPENAI_API_KEY 로드
  │    ├─ HF_HOME = Drive/models (영속 캐시)
  │    ├─ patch_rag_bench_config()  ← 경로 오버라이드
  │    │    ├─ cfg.DOCS_DIR      → /content/RAG-Bench/docs
  │    │    ├─ cfg.BENCH_DOCS_DIR → colab/data/docs
  │    │    ├─ cfg.BENCH_DATA_DIR → Drive/_benchdata
  │    │    └─ gqa.* 모듈 변수 패치 (import-time 바인딩 대응)
  │    └─ _setup_korean_font()  (device= 파라미터는 DenseSparseStrategy()에 직접 전달)
  │
  ├─ [Step 2] ColabBenchmarkRunner(preset, k, top_n, ...)
  │
  ├─ [Step 3] runner.prepare_qa(num_qa, sample_pages, ...)
  │    ├─ 캐시 확인: Drive/_benchdata/qa_dataset.json 존재 시 스킵
  │    ├─ [sample_pages=True 시]
  │    │    pdfs_to_markdowns(/content/RAG-Bench/docs/*.pdf
  │    │        → colab/data/docs/*.md, ratio=10%, max=5pages)
  │    ├─ create_parent_child_chunks(colab/data/docs)
  │    ├─ effective_num_qa = min(num_qa, chunks × max_qa_per_page)
  │    ├─ KG_SAVE_PATH 임시 오버라이드 → Drive/_benchdata/ragas_kg.json
  │    ├─ _generate_qa_ragas(parent_pairs, num_qa=effective)
  │    │    └─ RAGAS KnowledgeGraph + TestsetGenerator
  │    └─ 저장: Drive/_benchdata/qa_dataset.json
  │
  ├─ [Step 4] runner.prepare_data()
  │    └─ qa_dataset.json 로드 → (child_chunks, parent_pairs, queries, gts)
  │
  ├─ [Step 5] runner.generate_combos() + run_pass1()
  │    ├─ 프리셋별 조합 수: quick=2, standard=20, full=60
  │    ├─ 각 전략 × 쿼리 → 레이턴시 측정
  │    ├─ 체크포인트: Drive/checkpoints/ (세션 재시작 시 복구)
  │    └─ → lat_df (레이턴시 결과 DataFrame)
  │
  ├─ [Step 6] runner.run_pass2(lat_df, ...)
  │    ├─ lat_df 상위 top_n 전략 선별
  │    ├─ Pass 1 결과 inject_results()로 재사용 (검색 생략)
  │    └─ → ragas_df (RAGAS 점수 DataFrame)
  │
  └─ [Step 7] runner.export_results(lat_df, ragas_df)
       ├─ Drive/results/latency_*.csv
       ├─ Drive/results/ragas_*.csv
       └─ Drive/results/benchmark_report_*.html
```

### 코드 실행 예시

```python
# 1. 환경 초기화 (Drive 마운트 + config 패치 + CUDA 디바이스)
from rag_bench_colab.colab_config import init_colab
init_colab(qdrant_mode="ephemeral")

# 2. 러너 생성
from rag_bench_colab.colab_runner import ColabBenchmarkRunner
runner = ColabBenchmarkRunner(preset="quick", k=3, top_n=4)

# 3. QA 데이터셋 생성 (최초 1회 또는 force=True 시)
#    기본: 기존 rag_bench_colab/data/docs/*.md 사용
runner.prepare_qa(num_qa=20)

#    PDF 페이지 샘플링 적용 (docs/*.pdf → data/docs/*.md 재생성)
runner.prepare_qa(num_qa=20, sample_pages=True)

# 4. 데이터 로드 + 청킹
child_chunks, parent_pairs, queries, ground_truths = runner.prepare_data()

# 5. 벤치마크 실행
combos = runner.generate_combos()
lat_df  = runner.run_pass1(combos, queries, child_chunks, parent_pairs)
ragas_df = runner.run_pass2(lat_df, combos, queries, ground_truths, child_chunks, parent_pairs)

# 6. 결과 저장 (Drive + HTML 보고서)
runner.export_results(lat_df, ragas_df)
```

## 프리셋

| 프리셋 | 조합 수 | Dense | Sparse | 예상 시간 | API 비용 |
|--------|---------|-------|--------|----------|---------|
| `quick` | 2 | bge-m3 (1종) | korean_bm25 (1종) | ~10분 | ~$0.3 |
| `standard` | 20 | HF 3종 + 유료 2종 | 2종 | ~45분 | ~$2 |
| `full` | 60 | 5종 | 2종 | ~3시간 | ~$5 |

## 주요 파라미터

### ColabBenchmarkRunner

```python
runner = ColabBenchmarkRunner(
    preset="quick",               # quick | standard | full
    k=3,                          # 검색 결과 수
    top_n=10,                     # Pass 2에서 RAGAS 평가할 상위 전략 수
    qdrant_mode="ephemeral",      # ephemeral | drive | memory
    device=None,                  # cuda | cpu | None (자동 감지)
    parallel_queries=0,           # 쿼리 병렬화 (0=비활성, T4에서 4~8 권장)
    reindex=False,                # True: 기존 인덱스 삭제 후 재구축
    metric_preset="core_only",    # core_only (4) | comprehensive (7) | full (11+) | reference_free
    scoring_profile="balanced",   # balanced | precision_critical | speed_critical | comprehensive
)
```

### prepare_qa() — QA 데이터셋 생성

```python
runner.prepare_qa(
    num_qa=20,                # 생성할 QA 수
    sample_pages=False,       # True: docs/*.pdf를 페이지 샘플링 → data/docs/*.md 재생성
    page_sample_ratio=0.1,    # 샘플링 비율 (기본 10%)
    max_sample_pages=5,       # 최대 샘플 페이지 수
    max_qa_per_page=2,        # 청크당 최대 QA 수 (QA 상한 계산용)
    force=False,              # True: 캐시 무시하고 재생성
    reuse_kg=False,           # True: 기존 KG 파일 재사용
    build_kg_only=False,      # True: KG만 구축, QA 생성 안 함
    num_personas=3,           # RAGAS 자동 페르소나 수
    query_dist="balanced",    # single_hop | multi_hop | balanced
)
```

> **경로 패치 의존성**: `prepare_qa()`는 `init_colab()` 호출 이후에 실행해야 합니다.
> `init_colab()` 내부의 `patch_rag_bench_config()`가 `DOCS_DIR`, `BENCH_DOCS_DIR`,
> `KG_SAVE_PATH` 등을 Colab 경로로 오버라이드합니다.

### 메트릭 프리셋

| 프리셋 | 메트릭 수 | 설명 |
|--------|----------|------|
| `core_only` | 4 | faithfulness, answer_relevancy, context_precision, llm_context_recall |
| `comprehensive` | 7 | Core 4 + factual_correctness, context_entity_recall, response_relevancy |
| `full` | 11+ | 모든 RAGAS v0.4+ 메트릭 (lightweight 포함) |
| `reference_free` | 2~3 | ground truth 불필요한 메트릭만 |

### 스코어링 프로파일

| 프로파일 | 설명 |
|---------|------|
| `balanced` | 4대 Core 메트릭 균등 가중 (각 25%) |
| `precision_critical` | 정확성 중심 (faithfulness 40%, context_precision 30%) |
| `speed_critical` | 핵심 2개만 (answer_relevancy 50%, faithfulness 50%) |
| `comprehensive` | 7개 메트릭 고르게 분산 |

## 디렉토리 구조

```
rag_bench_colab/
├── README.md                  # 이 파일
├── rag_benchmark.ipynb        # 메인 Colab 노트북
├── colab_config.py            # Colab 환경 설정 + rag_bench 패치
├── colab_runner.py            # 체크포인트 지원 벤치마크 러너
├── colab_visualizer.py        # 시각화 유틸리티
├── requirements_colab.txt     # Colab 전용 의존성
└── data/
    ├── qa_dataset.json        # QA 데이터셋 (20쌍)
    └── docs/                  # 벤치마크 대상 마크다운 문서
```

## 체크포인트

세션이 끊기더라도 Google Drive에 체크포인트가 저장됩니다.
커널 재시작 후 동일 설정으로 다시 실행하면 완료된 전략은 건너뜁니다.

저장 위치: `Google Drive > MyDrive > rag_bench_colab > checkpoints/`

## Qdrant 모드

| 모드 | 설명 | 영속성 |
|------|------|--------|
| `ephemeral` | `/content/qdrant_workspace` (로컬) | 세션 종료 시 삭제 |
| `drive` | Google Drive 저장 | 영속 |
| `memory` | 인메모리 (`:memory:`) | 세션 종료 시 삭제 |

## Cell 1.2 설치 최적화 (uv + flash-attn wheel 캐시)

세션 시작 시 패키지 설치 시간을 크게 단축하는 두 가지 최적화가 적용되어 있습니다:

| 최적화 | 효과 | 상세 |
|--------|------|------|
| **uv 패키지 매니저** | requirements 설치 ~8배 빠름 | pip 대비 고속. Drive에 `UV_CACHE_DIR` 설정으로 세션 간 캐시 재사용 |
| **flash-attn 사전 빌드 wheel** | 설치 30초 (소스 빌드 30~120분 → ~30초) | CUDA/Python/torch 버전 조합으로 wheel 파일명 자동 구성, GitHub에서 다운로드 |
| **Drive wheel 캐시** | 2회차부터 ~5초 | 다운로드한 wheel을 Drive에 저장, 재시작 시 즉시 재사용 |

**설치 우선순위:** ① Drive 캐시 wheel → ② GitHub wheel 다운로드 → ③ 소스 빌드 (fallback)

## rag_bench 최적화 적용 내역

`rag_bench/scripts/run_all_combos.py`의 최적화를 Colab 환경에 맞게 반영한 사항:

| # | 심각도 | 내용 | 파일 | 효과 |
|---|--------|------|------|------|
| 1 | CRITICAL | Qdrant 경로 패치 전파 | `colab_config.py` | `run_all_combos` 모듈의 값 복사된 `BENCH_DATA_DIR`도 패치하여 Colab 경로 불일치 방지 |
| 2 | CRITICAL | Pass 2 재검색 제거 | `colab_runner.py` | Pass 1 결과를 `inject_results()`로 재사용, Pass 2 실행 시간 ~50% 단축 |
| 3 | IMPORTANT | 전략 cleanup 호출 | `colab_runner.py` | Reranker 래핑 전략의 Qdrant 파일 잠금/메모리 누수 방지 |
| 4 | MODERATE | `parallel_queries` 연결 | `colab_runner.py` | T4 GPU 쿼리 병렬화 활용 가능 |
| 5 | MODERATE | `reindex` 파라미터 노출 | `colab_runner.py` | 사용자가 인덱스 재구축 가능 (`reindex=True`) |
| 6 | LOW | ColBERT `_device` 일관성 | `colab_config.py` | `build_strategy_from_spec` 래핑으로 CUDA 디바이스 자동 적용 |
| 7 | LOW | `DENSE_DIMS` 룩업 활용 | `colab_config.py` | 알려진 Dense 모델은 test inference 생략하여 초기화 가속 |
