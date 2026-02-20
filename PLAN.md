# PLAN.md — 중간 우선순위 기술 부채 해결

## 배경

현재 코드베이스에는 다음 6개의 중간 우선순위 기술 부채가 존재한다:

| 번호 | 항목 | 파일 |
|------|------|------|
| #5 | `run_all_combos.py` 레거시 `_try_build_*` 함수 제거 | `rag_bench/scripts/run_all_combos.py` |
| #6 | `DENSE_MODELS`에 유료 모델(openai-small/large, upstage) 포함 — `PRESETS`와 불일치 | `rag_bench/strategies/dense_sparse.py`, `combo/spec.py` |
| #7 | `COMBO_DEFINITIONS` 레거시 dict 제거 (하위 호환 코드) | `rag_bench/strategies/dense_sparse.py` |
| #9 | `--combos`, `--skip_*` CLI 레거시 옵션 제거 | `rag_bench/scripts/run_all_combos.py` |
| #10 | `qdrant_db_combo1~4` 레거시 인덱스 정리 스크립트 생성 | `scripts/cleanup_legacy_indexes.py` |
| #16 | `config.py`에 `QDRANT_DB_PREFIX` 상수 추가 (하드코딩된 "qdrant_db_" 대체) | `rag_bench/config.py`, `combo/cache.py` |

---

## Phase 구성

### Phase 1 — config.py 확장 + 레거시 상수 정리
- **Task 1.1**: `config.py`에 `QDRANT_DB_PREFIX` 상수 추가 (#16)
- **Task 1.2**: `combo/cache.py`에서 하드코딩된 `"qdrant_db_"` → `QDRANT_DB_PREFIX` 치환 (#16)
- **Task 1.3**: `dense_sparse.py`에서 `COMBO_DEFINITIONS` 및 `combo_id` 하위 호환 코드 제거 (#7)

### Phase 2 — 레거시 CLI 옵션 및 함수 제거
- **Task 2.1**: `run_all_combos.py`에서 `--combos`, `--skip_*` 옵션 + `_run_legacy_mode()` + `_try_build_*` 함수 제거 (#5, #9)
- **Task 2.2**: `DENSE_MODELS`에서 `openai-small`, `openai-large`, `upstage` 제거 + `DENSE_DIMS` 동기화 (#6)

### Phase 3 — spec.py 정비
- **Task 3.1**: `combo/spec.py`의 `PRESETS`에서 유료 모델 참조 확인 + 정리 (#6)
- **Task 3.2**: `generate_valid_combinations()` 유효성 검증 로직 추가 (#6)

### Phase 4 — 레거시 인덱스 정리 스크립트
- **Task 4.1**: `scripts/cleanup_legacy_indexes.py` 생성 (dry-run 기본) (#10)
- **Task 4.2**: dry-run 검증 실행 후 스크립트 확인 (실제 삭제는 사용자가 명시적으로 실행) (#10)

---

## 실행 Wave

```
W1: Task 1.1                         (config.py 상수 추가 — 선행 필수)
W2: Task 1.2, 1.3, 4.1, 4.2         (병렬 실행 가능)
W3: Task 2.1, 3.1                    (Phase 2, 3 시작)
W4: Task 2.2, 3.2                    (마무리)
```

---

## 체크리스트

### Phase 1

- [x] **Task 1.1** — `config.py`에 `QDRANT_DB_PREFIX = "qdrant_db_"` 추가
  - Done: `QDRANT_DB_PREFIX` 상수가 `config.py`에 존재하고 `from rag_bench.config import QDRANT_DB_PREFIX` 가능

- [ ] **Task 1.2** — `combo/cache.py` `"qdrant_db_"` → `QDRANT_DB_PREFIX` 치환
  - Done: `cache.py` 내 하드코딩된 `"qdrant_db_"` 문자열이 0개

- [ ] **Task 1.3** — `dense_sparse.py` `COMBO_DEFINITIONS` 및 `combo_id` 제거
  - Done: `COMBO_DEFINITIONS` 심볼이 `dense_sparse.py`에 존재하지 않음
  - Done: `DenseSparseStrategy.__init__`의 `combo_id` 파라미터가 제거됨

### Phase 2

- [ ] **Task 2.1** — `run_all_combos.py` 레거시 코드 제거
  - Done: `_run_legacy_mode()` 함수가 존재하지 않음
  - Done: `_try_build_dense_sparse()`, `_try_build_colbert()`, `_try_build_rerank()`, `_try_build_contextual()`, `_try_build_flashrank_rerank()` 함수 제거
  - Done: `--combos`, `--skip_colbert`, `--skip_rerank`, `--skip_contextual`, `--skip_flashrank`, `--contextual_base` 인수 제거
  - Done: `ALL_COMBO_IDS` 상수 제거

- [ ] **Task 2.2** — `dense_sparse.py`에서 유료 모델 제거
  - Done: `DENSE_MODELS`에 `openai-small`, `openai-large`, `upstage` 키 없음
  - Done: `DENSE_DIMS`에서 해당 모델 차원 제거

### Phase 3

- [ ] **Task 3.1** — `combo/spec.py` PRESETS 정리
  - Done: `PRESETS`의 `dense_models` 목록이 `DENSE_MODELS`에 존재하는 키만 참조

- [ ] **Task 3.2** — `generate_valid_combinations()` 유효성 검증
  - Done: 유효하지 않은 dense/sparse 키를 넘기면 `ValueError` 발생

### Phase 4

- [ ] **Task 4.1** — `scripts/cleanup_legacy_indexes.py` 생성
  - Done: 파일이 `scripts/cleanup_legacy_indexes.py`에 존재
  - Done: 기본 실행이 dry-run (실제 삭제 없음)
  - Done: `--execute` 플래그로만 실제 삭제 수행

- [ ] **Task 4.2** — dry-run 검증
  - Done: `python scripts/cleanup_legacy_indexes.py` 실행 시 오류 없이 대상 목록 출력

---

## 주의사항

1. **Breaking Change**: Task 2.1 (`--combos`, `--skip_*` 제거)은 기존 스크립트 사용자에게 영향을 준다. MEMORY.md에 기록한다.
2. **Task 4.2**: 스크립트 생성 및 dry-run 확인까지만 수행. 실제 `qdrant_db_combo1~4` 삭제는 사용자가 `--execute` 플래그로 명시적으로 실행한다.
3. **Task 1.3**: `combo_id`를 사용하는 레거시 코드(`_run_legacy_mode`)가 Task 2.1에서 함께 제거되므로, 순서에 주의한다.
