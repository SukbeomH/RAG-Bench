---
title: "uv workspace 모노레포 전환 + Citation 파이프라인"
tags:
  - arch
  - decision
  - monorepo
  - uv-workspace
  - citation
  - refactoring
type: architecture-decision
created: 2026-03-04T11:00:00+09:00
contextual_description: "AutoRAG를 uv workspace 모노레포로 전환. 5개 Python 패키지(pdf-parsers, pdf-eval, rag-retrieval, rag-eval, rag-api) + Next.js 프론트엔드. Citation provenance 파이프라인 도입."
keywords:
  - uv workspace
  - monorepo
  - autorag-parsers
  - autorag-retrieval
  - autorag-api
  - citation
  - provenance
  - bbox
  - ChunkProvenance
  - PDFParser Protocol
  - registry pattern
related:
  - 2026-02-26_codebase-architecture
  - 2026-02-20_rag-bench-service-modularization-cookbook
---

## uv workspace 모노레포 전환 + Citation 파이프라인

### 결정 배경
- 단일 pyproject.toml에 torch, langchain, docling 등 모든 의존성 집중
- pdf_parser/와 rag_bench/ 간 분리되어 파이프라인 연결 어려움
- RAG 답변에 출처(페이지, bbox) 추적 불가

### 결정 내용
**uv virtual workspace**로 전환 (root에 빌드 없음, `[tool.uv.workspace]` only).

| 패키지 | import | 역할 |
|---|---|---|
| `pdf-parsers` | `autorag_parsers` | PDFParser Protocol + Registry, 6개 백엔드, bbox 추출, chunking+provenance |
| `pdf-eval` | `autorag_pdf_eval` | NED/TEDS 평가 |
| `rag-retrieval` | `autorag_retrieval` | Dense+Sparse, ColBERT/FlashRank reranking |
| `rag-eval` | `autorag_rag_eval` | RAGAS 평가 |
| `rag-api` | `autorag_api` | FastAPI (/api/parse, /api/ask 등) |
| `rag-frontend` | — | Next.js 15 + Tailwind, Chat+PDFViewer+Citation |

### 핵심 패턴
- **PDFParser Protocol**: `name` property + `convert(pdf_path) -> ConversionResult`
- **Registry**: `@register("pymupdf")` 데코레이터, `get_parser(name)` 팩토리, lazy-loading
- **ChunkProvenance**: doc_id, page_number, chunk_id, bbox 추적
- **PyMuPDF bbox**: `page.get_text("words")` → word-level bounding box → ChunkProvenance.bbox
- **workspace 간 의존성**: `[tool.uv.sources]` + `autorag-parsers = { workspace = true }`

### 대안 비교
- Poetry workspace: uv보다 느림, virtual workspace 미지원
- pip-tools: 워크스페이스 개념 없음
- nx/turborepo: Python 생태계 아님

### 배포 구조
- `deploy/k8s/` — Dockerfiles, manifests, scripts
- `servers/` — OCR FastAPI 서버 (paddleocr_vl, got_ocr2, deepseek_ocr2)
- `orchestrators/` — K8s job 오케스트레이터
- `isolated_backends/` — Python 버전 충돌 격리 (paddleocr, deepseek_ocr2)
- `data/benchmark_pdfs/` — 벤치마크 PDF 데이터
