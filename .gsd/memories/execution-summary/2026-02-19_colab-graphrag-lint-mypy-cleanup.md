---
title: "Colab GraphRAG 통합 + ruff/mypy 0 errors 달성"
tags:
  - execution
  - summary
  - colab
  - graphrag
  - lint
  - mypy
  - rag-bench
type: execution-summary
created: "2026-02-19T00:00:00+09:00"
contextual_description: "rag_bench_colab에 GraphRAG ComboSpec 통합, Colab 런타임 버그 3종 수정, ruff 0 + mypy 0 달성, PR #7 master 머지"
keywords:
  - GraphRAG
  - ComboSpec
  - INCLUDE_GRAPHRAG
  - ColabBenchmarkRunner
  - colab_runner
  - _strategy_name_from_spec
  - plot_latency_vs_quality
  - ruff
  - mypy
  - Optional
  - python-dotenv
  - Colab Secrets
related:
  - 2026-02-19_pass2-combospec-matching-root-cause
  - 2026-02-19_mypy-implicit-optional-pattern
---

## Colab GraphRAG 통합 + ruff/mypy 0 errors 달성

### 브랜치 / PR
- 브랜치: `feat/graphrag-as-combo` → PR #7 → `master` 머지

### 주요 변경 내용

#### 1. GraphRAG ComboSpec 통합
- `ComboSpec.graphrag: bool = False` 필드 추가
- `ComboSpec.for_graphrag()` 팩토리 메서드
- `generate_valid_combinations(include_graphrag=False)`
- `build_strategy_from_spec()` GraphRAG 분기 추가 (GraphRAGStrategy 생성)
- `ColabBenchmarkRunner.__init__(include_graphrag=False)` 파라미터 추가
- 노트북: `RUN_GRAPHRAG` → `INCLUDE_GRAPHRAG`, Section 7 통합 안내로 교체

#### 2. API 키 로딩 개선 (Colab Secrets → dotenv 폴백)
- `colab_config.setup_colab_env()`: Colab/로컬 구분 제거
- Colab Secrets 접근 실패 시 자동으로 `.env` 파일 탐색
- 탐색 경로: COLAB_PROJECT_ROOT/.env → 프로젝트 루트/.env → ./env

#### 3. Colab 런타임 버그 3종 수정
- **Pass 2 ComboSpec 매칭 실패**: `_strategy_name_from_spec` 수정 (DENSE_MODELS 룩업 + 리랭커 prefix)
- **`faithfulness` KeyError**: `plot_latency_vs_quality` quality_metric 자동 선택 추가
- **`include_graphrag` TypeError**: 이미 코드에 반영되어 있었음 (Colab 커널 재시작으로 해소)

#### 4. ruff 0 errors
- E741: `for l in` → `for llm_sup in` / `for lbl in`
- F841: `spec_map`, `run_record`, `original_ssl_bypass` 미사용 변수 제거
- F821: `get_cache_config` 반환 타입 어노테이션 제거
- F401: `List` / `typing.List` 미사용 import 정리

#### 5. mypy 0 errors (30개 → 0개)
- `run_tracker.py`: `str = None` → `Optional[str] = None` (implicit optional)
- `dense_sparse.py`: `_combo_id: Optional[int]` 첫 할당에 어노테이션
- `evaluator.py`: `_metrics: Optional[List[Any]]`, `ground_truths is not None` 방어
- `runner.py`: TYPE_CHECKING guard, `List[Optional[dict]]`, `type: ignore[return-value/arg-type]`
- `colab_config.py`: `# type: ignore[method-assign]` (monkey-patch)
- `colab_runner.py`: `Optional[Any]` 어노테이션, `Dict[str, Any]` score_dict
- `colab_visualizer.py`: `groups: Dict[str, List[float]]`, `assert lat_df is not None`
- `config.py`: `requests` import-untyped, `dotenv` E402 noqa

### 커밋
```
4947ae4 fix: mypy 타입 에러 해소 (30개 → 0개)
fb94ebb fix: 린트 에러 제거 + Colab 실행 버그 3종 수정
9ada2f0 feat: 로컬 실행 시 python-dotenv .env 자동 로드
7ad1186 feat: GraphRAG를 ComboSpec으로 통합 — INCLUDE_GRAPHRAG 옵션
```

### 현재 상태 (master)
- ruff: 0 errors
- mypy: 0 errors (4개 대상 파일 + 의존성)
- PR #7 머지 완료, `feat/graphrag-as-combo` 브랜치 삭제
