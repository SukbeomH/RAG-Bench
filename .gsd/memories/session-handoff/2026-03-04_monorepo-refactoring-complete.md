---
title: "모노레포 리팩토링 7-Phase 완료 — 핸드오프"
tags:
  - handoff
  - session
  - monorepo
  - refactoring
  - citation
  - uv-workspace
type: session-handoff
created: 2026-03-04T11:15:00+09:00
contextual_description: "AutoRAG 모노레포 7-Phase 리팩토링 완료 후 핸드오프. uv workspace 5패키지 + Next.js 프론트엔드. 기존 코드 참조용 보존 상태."
keywords:
  - uv workspace
  - autorag-parsers
  - autorag-retrieval
  - autorag-api
  - citation provenance
  - next.js frontend
  - 잔여 작업
related:
  - 2026-03-04_uv-workspace-monorepo-refactoring
  - 2026-03-04_monorepo-refactoring-7-phases
---

## 모노레포 리팩토링 7-Phase 완료 — 핸드오프

### 완료 상태
커밋 `6f010dc` (master, pushed). 7-Phase 모두 완료 및 E2E 검증 통과.

### 새 패키지 구조 (packages/)
| 패키지 | import | 검증 |
|---|---|---|
| `pdf-parsers` | `autorag_parsers` | `get_parser('pymupdf').convert()` → 5 pages, bbox=True |
| `pdf-eval` | `autorag_pdf_eval` | import OK |
| `rag-retrieval` | `autorag_retrieval` | `DenseSparseStrategy`, `ComboSpec` import OK |
| `rag-eval` | `autorag_rag_eval` | `MetricPreset`, `create_metrics` import OK |
| `rag-api` | `autorag_api` | FastAPI 8 routes, `/api/parse` + `/api/ask` E2E 통과 |
| `rag-frontend` | — | `next build` 성공, 4 컴포넌트 |

### 미커밋 잔여 파일 (git status)
다음 파일들이 working tree에 남아 있음 (의도적 보존 또는 추후 정리 필요):

**Modified (기존 코드, 참조용 보존 중):**
- `pdf_parser/benchmark/runner.py` — 기존 벤치마크 러너 (이전 세션 수정분)
- `pdf_parser/benchmark/spec.py` — 기존 프리셋 정의
- `pdf_parser/category3_openai.py` — 기존 OpenAI 백엔드
- `pdf_parser/category3_opensource.py` — 기존 OCR 백엔드

**Untracked (정리 대상):**
- `CLAUDE.md` — 빈 파일, 삭제 가능
- `docs/research/rag_citation_provenance_ui_research.md` — citation 리서치 문서 (커밋 추천)
- `k8s/Dockerfile.deepseek-ocr2`, `k8s/deepseek_ocr2_server.py`, `k8s/manifests/deepseek-ocr2-deployment.yaml` — 이미 `deploy/`, `servers/`로 복사됨, 원본 삭제 가능
- `packages/rag-frontend/next-env.d.ts`, `package-lock.json` — 커밋 추천
- `paddleocr_standalone/` — 계획대로 삭제 대상 (Phase 7)
- `pdf_parser/backends/` — `isolated_backends/`로 복사됨, 원본 삭제 가능
- `pdf_parser/generate_report.py`, `pdf_parser/reports/` — pdf-eval로 이동 예정 또는 삭제

### 다음 세션 추천 작업

1. **레거시 정리**: `paddleocr_standalone/`, `rag_bench_colab/`, `rag_bench_local/`, 중복 k8s 파일 삭제
2. **pdf-eval 패키지 완성**: `benchmark/evaluator.py`, `benchmark/spec.py`, `benchmark/runner.py`를 `autorag_pdf_eval`로 마이그레이션
3. **rag-api LLM 연동**: `/api/ask` 엔드포인트에 실제 LLM (Ollama/OpenAI) 호출 추가
4. **rag-api 벡터 검색**: `/api/retrieve`를 키워드→Qdrant 벡터 검색으로 전환
5. **react-pdf-highlighter-extended**: React 19 호환 시 PDFViewer에 통합 (현재 pdfjs-dist 직접 사용)
6. **CLAUDE.md 작성**: 새 패키지 구조에 맞는 프로젝트 지침서

### 주의사항
- **`uv run` 사용**: 모든 Python 실행은 `uv run python` (workspace venv 자동 활성화)
- **workspace 간 의존성**: `[tool.uv.sources]`에 `{ workspace = true }` 필수
- **root pyproject.toml**: virtual workspace — `[project]` 섹션 없음, 빌드 안 됨
- **기존 `pdf_parser/`, `rag_bench/`**: 참조용 보존 중. 새 코드는 `packages/` 아래에만 작성
