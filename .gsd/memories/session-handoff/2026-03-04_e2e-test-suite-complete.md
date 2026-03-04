---
title: "E2E 테스트 스위트 48개 작성 완료 — 핸드오프"
tags:
  - handoff
  - session
  - testing
  - e2e
  - pytest
type: session-handoff
created: 2026-03-04T16:00:00+09:00
contextual_description: "5개 Python 패키지 E2E 테스트 48개 작성. 외부 의존성(K8s/GPU/API 키) 없이 로컬 완전 실행. pytest importmode=importlib로 패키지간 tests/ 충돌 해결."
keywords:
  - pytest
  - e2e test
  - autorag_parsers
  - autorag_pdf_eval
  - autorag_retrieval
  - autorag_rag_eval
  - autorag_api
  - importlib
related:
  - 2026-03-04_monorepo-refactoring-complete
---

## E2E 테스트 스위트 48개 작성 완료 — 핸드오프

### 완료 상태
48 tests passed, 0 failed (33초). 외부 서비스 없이 로컬에서 완전 실행 가능.

### 생성 파일 (11개)
| 파일 | 테스트 수 | 검증 범위 |
|------|-----------|-----------|
| `conftest.py` (root) | fixture 4개 | benchmark_pdf_dir, gt_dir, sample_pdf, table_pdf |
| `packages/pdf-parsers/tests/test_e2e_parsers.py` | 7 | pymupdf 파싱, 청킹, provenance, registry |
| `packages/pdf-eval/tests/test_e2e_eval.py` | 11 | NED/TEDS 메트릭, evaluate_document, presets, GT 파일 |
| `packages/rag-retrieval/tests/test_e2e_retrieval.py` | 15 | ComboSpec, 조합 생성, DocType, Dense/Sparse 모델 |
| `packages/rag-eval/tests/test_e2e_rag_eval.py` | 9 | RAGAS 가중치, MetricPreset, METRIC_REGISTRY |
| `packages/rag-api/tests/test_e2e_api.py` | 6 | /health, /api/parse, schema validation |

### 설정 변경
- `pyproject.toml`: `[tool.pytest.ini_options]` 추가
  - `pythonpath = ["."]` — root conftest.py 인식
  - `addopts = "--import-mode=importlib"` — 5개 패키지의 `tests/` 이름 충돌 해결

### 실행 방법
```bash
uv run pytest packages/*/tests/ -v --tb=short   # 전체
uv run pytest packages/pdf-parsers/tests/ -v     # 패키지별
```

### 발견된 이슈
1. **API 에러 핸들링 미비**: `/api/parse`에 잘못된 backend 전달 시 unhandled `KeyError` 발생
   - 해결 방안: `parse.py`에서 `get_parser()` 호출을 try/except로 감싸고 `HTTPException(422)` 반환
   - 테스트는 `pytest.raises(Exception)`으로 현재 동작 검증

### 다음 세션 추천 작업
1. **API 에러 핸들링**: `/api/parse` 엔드포인트에 HTTPException 에러 처리 추가
2. **docling 테스트 추가**: docling 백엔드가 설치된 환경에서만 실행되는 conditional 테스트
3. **CI 파이프라인**: GitHub Actions에 `uv run pytest` 통합
4. **코드 커버리지**: `pytest-cov` 추가하여 커버리지 리포트 생성
