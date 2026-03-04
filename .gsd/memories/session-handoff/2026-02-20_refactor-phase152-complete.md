# Session Handoff: 리팩토링 Phase 1 / 5 / 2 완료

## Date: 2026-02-20
## Branch: master

---

## What Was Done

### 1. Phase 1 — `rag_bench/combo/` 패키지 추출

**목표**: `run_all_combos.py` 1,348줄 God Script에서 도메인 모델 분리

**생성 파일**:
- `rag_bench/combo/__init__.py` — 공개 API re-export
- `rag_bench/combo/spec.py` — `ComboSpec`, `PRESETS`, `generate_valid_combinations()`
- `rag_bench/combo/cache.py` — `CacheConfig` (DEFAULT_CONTEXTUAL_LLM 참조), `IndexCacheManager`
- `rag_bench/combo/builder.py` — `build_strategy_from_spec()`

**수정 파일**:
- `run_all_combos.py` — 정의 290줄 제거 → `rag_bench.combo` import, `_try_build_contextual()` `"gpt-4o-mini"` → `DEFAULT_CONTEXTUAL_LLM`
- `colab_config.py` — `CacheConfig` import 경로 변경
- `colab_runner.py` — `ComboSpec` 등 import 경로 변경

**커밋**: `8dbb0ab` refactor(combo)

---

### 2. Phase 5 — 공유 유틸리티 추출

**목표**: `_load_qa_dataset()`, `_print_ragas_table()` 중복 제거

**생성 파일**:
- `rag_bench/utils/qa_loader.py` — `load_qa_dataset(data_dir: Path) -> dict`
- `rag_bench/utils/report.py` — `print_ragas_table(scores_df, scoring_profile="balanced")`

**수정 파일**:
- `rag_bench/utils/__init__.py` — re-export 추가
- `run_bench.py` — 로컬 `_load_qa_dataset`, `_print_ragas_table` 제거
- `run_all_combos.py` — 로컬 함수 제거, 공용 모듈 사용

**커밋**: `68467d6` refactor(utils)

---

### 3. Phase 2 — monkey-patch → DI 패턴

**목표**: `colab_config.py` 세 monkey-patch 함수 제거

| 제거 대상 | 대안 |
|-----------|------|
| `patch_dense_device(device)` | `DenseSparseStrategy(device=)` 파라미터 |
| `patch_qdrant_memory_mode()` | `_init_qdrant()` 내 `:memory:` 분기 내장 |
| `patch_rag_bench_config()` | 환경변수 `RAG_BENCH_DATA_DIR` 등 + 하위 호환 직접 패치 유지 |

**수정 파일**:
- `rag_bench/strategies/dense_sparse.py`:
  - `__init__(device: Optional[str] = None)` 추가
  - `_init_dense()`: `self._device` 우선, fallback `detect_device()`; `trust_remote_code=True` 추가
  - `_init_qdrant()`: `":memory:"` → `QdrantClient(location=":memory:")` 분기 내장
- `rag_bench/config.py`: `RAG_BENCH_DATA_DIR`, `RAG_BENCH_DOCS_DIR`, `RAG_BENCH_DOCS_SRC` 환경변수 오버라이드 블록 추가
- `rag_bench_colab/colab_config.py`:
  - `patch_dense_device()` → `DeprecationWarning` + no-op
  - `patch_qdrant_memory_mode()` → `DeprecationWarning` + no-op
  - `init_colab()` — 두 patch 호출 제거
  - `get_cache_config()` — `dense_device=device` 전달
- `rag_bench/combo/cache.py`:
  - `CacheConfig.dense_device: Optional[str] = None` 필드 추가
  - `get_or_build()`, `get_or_build_contextual()` — `device=self.config.dense_device` 전달

**커밋**: `97b9d38` refactor(dense), `d5258b4` refactor(colab)

---

### 4. 검증 (10/10 PASS)

| # | 항목 | 결과 |
|---|------|------|
| 1 | 패키지 import 체인 | PASS |
| 2 | monkey-patch 제거 확인 | PASS |
| 3 | 전략 모듈 5개 import | PASS |
| 4 | 스크립트 import | PASS |
| 5 | Colab 모듈 import | PASS |
| 6 | 환경변수 오버라이드 | PASS |
| 7 | ComboSpec + 126개 조합 생성 | PASS |
| 8 | CacheConfig.dense_device | PASS |
| 9 | utils 통합 import (detect_device 누락 fix) | PASS (fix 후) |
| 10 | 중복 정의 잔존 없음 AST 확인 | PASS |

**fix 커밋**: `854da96` fix(utils): detect_device `__init__.py` re-export 누락 수정

---

## Current Architecture State

### 완료된 리팩토링 전체

| Phase | 내용 | 커밋 |
|-------|------|------|
| 3 | StrategyRetriever 5개 → 1개 통합 | `7136f3e`, `ef219c2` |
| 4 | detect_device() utils/device.py 중앙화 | `7136f3e` |
| 6 | LLM 모델명 상수 config.py 중앙화 | `4af9e41` |
| 1 | combo/ 패키지 추출 | `8dbb0ab` |
| 5 | qa_loader / report 공유 모듈 추출 | `68467d6` |
| 2 | monkey-patch → DI 패턴 | `97b9d38`, `d5258b4` |

**6 Phase 전체 완료. 아키텍처 리뷰에서 식별된 CRITICAL 2건, MAJOR 6건 모두 해결.**

---

## 최종 파일 구조

```
rag_bench/
├── base.py                    ← StrategyRetriever 제네릭 클래스
├── config.py                  ← DEFAULT_*_LLM 상수 + 환경변수 오버라이드
├── combo/                     ← [NEW] Phase 1
│   ├── __init__.py
│   ├── spec.py                ← ComboSpec, PRESETS, generate_valid_combinations
│   ├── cache.py               ← CacheConfig (dense_device 필드), IndexCacheManager
│   └── builder.py             ← build_strategy_from_spec
├── utils/                     ← [EXPANDED] Phase 4/5
│   ├── __init__.py            ← detect_device, load_qa_dataset, print_ragas_table
│   ├── device.py              ← detect_device()
│   ├── qa_loader.py           ← load_qa_dataset(data_dir)
│   └── report.py              ← print_ragas_table()
├── strategies/
│   └── dense_sparse.py        ← DenseSparseStrategy(device=, :memory: 내장)
└── ...

rag_bench_colab/
├── colab_config.py            ← patch_dense_device/qdrant_memory_mode → deprecated no-op
└── colab_runner.py            ← combo 패키지 import
```

---

## Critical Notes

- `patch_dense_device()`, `patch_qdrant_memory_mode()` 는 **no-op**으로 전환됨 (DeprecationWarning 발생)
  - 기존 Colab 노트북에서 호출해도 에러 없이 경고만 출력
  - 다음 세션에서 Colab 노트북 셀 업데이트 권장
- `DenseSparseStrategy._init_dense()` — HuggingFace 분기에 `trust_remote_code=True` 추가됨
  - bge-m3 등 커스텀 코드 모델 필수 옵션
- `ARCHITECTURE.md`, `STACK.md` — GSD 에이전트 자동 생성, 미커밋 상태 (루트에 존재)

---

## What Needs To Be Done Next

### 즉시
- **Colab 노트북 셀 업데이트**: `patch_dense_device()`, `patch_qdrant_memory_mode()` 호출 제거
  → `init_colab(qdrant_mode, device)` 만으로 충분
- **126개 조합 벤치마크 실행**: Phase 1/2/5 완료 후 Colab에서 전체 벤치마크 검증

### 단기
- **시각화 Phase 2**: H-2 Violin, M-1 Pipeline, M-3 Gantt, M-4 Cost-Efficiency 구현
- **ARCHITECTURE.md, STACK.md 커밋 여부** 결정

---

## Commits This Session

```
854da96 fix(utils): detect_device __init__.py re-export 누락 수정
d5258b4 refactor(colab): patch_dense_device/patch_qdrant_memory_mode deprecated 처리 — DI 패턴 적용
97b9d38 refactor(dense): DenseSparseStrategy device/:memory: 파라미터 내장 — monkey-patch 제거
68467d6 refactor(utils): load_qa_dataset / print_ragas_table 공유 모듈 추출
8dbb0ab refactor(combo): ComboSpec/CacheConfig/IndexCacheManager → rag_bench/combo/ 패키지 추출
```
