# Session Handoff: Google Colab RAG 벤치마크 환경 구현

## Date: 2026-02-19
## Branch: master (feat/colab-benchmark → master 머지 완료)

## What Was Done

### rag_bench_colab/ 디렉토리 전체 구현 (10개 파일, ~1,560줄)

Google Colab T4 GPU에서 72-조합 벤치마크를 실행할 수 있는 별도 환경을 구축했다.

#### 1. colab_config.py (368줄)
- `init_colab()`: Drive 마운트 + API Key + 경로 패치 + 디바이스 패치 통합 진입점
- `patch_rag_bench_config()`: BENCH_DATA_DIR, BENCH_DOCS_DIR, MODELS_DIR을 Colab 경로로 오버라이드
- `patch_dense_device()`: DenseSparseStrategy._init_dense() 패치 → CUDA 사용
- `patch_colbert_device()`: IndexCacheManager.get_colbert_model() 패치 → CUDA 사용
- `patch_qdrant_memory_mode()`: _init_qdrant() 패치 → `:memory:` 인메모리 모드 지원
- Qdrant 3가지 모드: ephemeral (로컬 /content), drive (Google Drive), memory (인메모리)

#### 2. colab_runner.py (652줄)
- `CheckpointManager`: Google Drive JSON 기반 전략별 체크포인트 (12시간 세션 제한 대응)
- `ColabBenchmarkRunner`: 2-Pass 벤치마크 래퍼
  - `prepare_data()`: QA 로드 + Parent-Child 청킹
  - `run_pass1()`: 전체 조합 레이턴시 (tqdm.notebook, 체크포인트)
  - `run_pass2()`: 상위 N개 RAGAS 평가 (ExtendedRAGEvaluator, 체크포인트)
  - `run_graphrag()`: LightRAG 별도 실행
  - `export_results()`: CSV/JSON/Markdown 리포트 Drive 저장

#### 3. colab_visualizer.py (443줄)
- 8개 시각화 함수: latency bar, RAGAS radar, heatmap, pareto scatter, layer boxplot, cost pie, leaderboard table, dashboard
- matplotlib + plotly + seaborn

#### 4. rag_benchmark.ipynb (9 섹션)
- Section 0: 개요 + 3-Layer 다이어그램
- Section 1-3: 환경설정, 사용자 파라미터, 데이터 로딩
- Section 4-6: 조합 생성, Pass 1 레이턴시, Pass 2 RAGAS
- Section 7-9: GraphRAG, 시각화 대시보드, 비용/Export

#### 5. 기타
- `requirements_colab.txt`: pyproject.toml 기반 + plotly/seaborn/kaleido
- `data/qa_dataset.json`: QA 데이터셋 복사
- `data/docs/*.md`: 마크다운 문서 2개 복사
- `README.md`: Colab 뱃지, 프리셋 테이블, 사용법

### 브랜치 작업

- `feat/colab-benchmark` 브랜치 생성 → PR #2 → master 머지 (f71ac35)
- `opt/medium-low-optimizations` 브랜치에서 colab revert 정리 (reset --soft)
- PR #1 (opt/medium-low-optimizations) 도 이미 master 머지됨 (0d3c7c2)

### 핵심 설계 결정

1. **Import 방식**: `git clone + sys.path.insert` (private repo 호환)
2. **Monkey-patch 전략**: rag_bench 코드 수정 없이 런타임 패치로 Colab 환경 대응
3. **체크포인트**: 전략별 JSON → Drive, 커널 재시작 시 완료된 전략 스킵
4. **세션 핸드오프 반영**:
   - ColBERT: Colab CUDA 허용 (MPS OOM은 macOS 전용 이슈)
   - FlashRank: CPU 유지 (ONNX 전용)
   - MPS 관련 코드: Colab에서 비활성화

## What Needs To Be Done Next

1. **Colab 실제 테스트**: T4 GPU 환경에서 quick 프리셋 E2E 실행 검증
2. **README.md의 `<user>` 플레이스홀더**: 실제 GitHub 사용자명으로 교체 (repo URL, Colab 뱃지)
3. **72개 조합 풀 벤치마크 재실행**: `--preset full --top_n 10 --layers` (로컬 macOS)
4. **QA 데이터셋 고도화**: RAGAS v2 방식 `--method ragas` 구현
5. **evaluation 메트릭 확장**: Extended 메트릭 + per-sample 점수

## Critical Notes

- `rag_benchmark.ipynb` Cell 1.2의 `REPO_URL`이 `<user>` 플레이스홀더 — 사용 전 교체 필수
- Qdrant `path` vs `location` 파라미터: `:memory:`는 반드시 `location`으로 전달해야 함 (patch_qdrant_memory_mode에서 처리)
- KoNLPy는 Java JDK 필요 → 노트북 첫 셀에서 `apt-get install default-jdk` 실행

## Key Files

- `rag_bench_colab/colab_config.py` — 환경 초기화 + rag_bench 패치 (핵심)
- `rag_bench_colab/colab_runner.py` — 체크포인트 지원 벤치마크 러너
- `rag_bench_colab/colab_visualizer.py` — 시각화 함수 8종
- `rag_bench_colab/rag_benchmark.ipynb` — 메인 Colab 노트북
- `rag_bench_colab/requirements_colab.txt` — Colab 전용 의존성

## Branch Layout (최종)

```
master (f71ac35)
├── PR #1: opt/medium-low-optimizations (324e8c6) — 머지 완료
└── PR #2: feat/colab-benchmark (338af6b) — 머지 완료
```

## Commits This Session

```
338af6b feat: Google Colab 벤치마크 환경 구축 — rag_bench_colab 패키지
```

## PRs This Session

```
#1 opt/medium-low-optimizations → master (0d3c7c2) — 이전 세션 작업
#2 feat/colab-benchmark → master (f71ac35) — 이번 세션 작업
```
