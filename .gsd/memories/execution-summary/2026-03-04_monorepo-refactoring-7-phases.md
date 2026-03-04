---
title: "모노레포 리팩토링 7-Phase 실행 완료"
tags:
  - execution
  - summary
  - monorepo
  - refactoring
  - citation
  - fastapi
  - nextjs
type: execution-summary
created: 2026-03-04T11:00:00+09:00
contextual_description: "AutoRAG 모노레포 7-Phase 리팩토링 전체 완료. uv workspace 5패키지 + Next.js 프론트엔드. E2E 파이프라인(PDF→chunking→citation→API→UI) 검증 통과."
keywords:
  - phase1 uv workspace
  - phase2 pdf-parsers protocol registry
  - phase3 citation provenance bbox
  - phase4 rag-retrieval rag-eval
  - phase5 fastapi rag-api
  - phase6 nextjs frontend
  - phase7 deploy cleanup
related:
  - 2026-03-04_uv-workspace-monorepo-refactoring
  - 2026-02-26_codebase-architecture
---

## 모노레포 리팩토링 7-Phase 실행 완료

### Phase별 결과

| Phase | 내용 | 상태 | 검증 |
|---|---|---|---|
| 1 | uv workspace 뼈대 (5패키지) | ✅ | `uv sync` 222 packages resolved |
| 2 | pdf-parsers Protocol+Registry+백엔드 | ✅ | `get_parser('pymupdf').convert()` → 5 pages |
| 3 | Citation provenance + bbox chunking | ✅ | 7 chunks with bbox=True |
| 4 | rag-retrieval + rag-eval 마이그레이션 | ✅ | `DenseSparseStrategy`, `ComboSpec` import OK |
| 5 | FastAPI rag-api 서비스 | ✅ | `/api/parse` + `/api/ask` E2E 통과, citation 포함 |
| 6 | Next.js 15 프론트엔드 | ✅ | `next build` 성공, 4개 컴포넌트 |
| 7 | 배포 정리 + 파일 이동 | ✅ | deploy/, servers/, orchestrators/ 구조 완성 |

### 주요 이슈 및 해결

1. **uv workspace root 빌드 실패**: `pyproject.toml`에 `[project]` 섹션이 있으면 빌드 시도 → virtual workspace (project 섹션 제거)로 해결
2. **workspace 간 의존성**: `autorag-parsers = { workspace = true }` in `[tool.uv.sources]` 필요
3. **pymupdf4llm words 빈 배열**: `page_chunks=True`에서 words 자동 제공 안 됨 → `fitz.open()` + `page.get_text("words")` 직접 호출로 해결
4. **react-pdf-highlighter-extended React 19 비호환**: 제거, pdfjs-dist로 직접 bbox 렌더링
5. **Next.js pdfUrl null 타입 에러**: non-null assertion (`!`) 추가

### 생성된 주요 파일
- `packages/pdf-parsers/src/autorag_parsers/` — _protocol.py, registry.py, pymupdf.py, docling.py, openai_vision.py, upstage.py, openai_compat.py, provenance.py, chunking.py
- `packages/rag-api/src/autorag_api/` — app.py, schemas.py, routers/parse.py, routers/retrieve.py
- `packages/rag-frontend/src/` — app/page.tsx, components/ChatInterface.tsx, CitationPanel.tsx, PDFViewer.tsx, SourceSidebar.tsx
