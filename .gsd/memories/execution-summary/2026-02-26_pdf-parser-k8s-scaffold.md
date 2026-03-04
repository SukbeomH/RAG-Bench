---
title: "PDF Parser K8s 시스템 초안 구축"
tags:
  - execution
  - summary
  - k8s
  - pdf-parser
type: execution-summary
created: 2026-02-26T10:10:00+09:00
contextual_description: "pdf_parser/k8s/ 하위에 Dockerfile, 워커, 오케스트레이터, 매니페스트 8개 파일 생성. 3백엔드(PyMuPDF/Docling/Gemini VLM) 지원, hybrid 페이지별 라우팅"
keywords:
  - pdf parser
  - k8s
  - dockerfile
  - docling
  - gemini vlm
  - hybrid routing
  - pymupdf4llm
related:
  - 2026-02-26_tei-removal-direct-embedding
---

## PDF Parser K8s 시스템 초안 구축

### 생성 파일 (8개)
- `pdf_parser/k8s/Dockerfile` — 멀티스테이지, tesseract OCR(한국어), python:3.12-slim
- `pdf_parser/k8s/requirements-worker.txt` — pymupdf4llm, docling, google-genai
- `pdf_parser/k8s/worker_entrypoint.py` — PVC 입출력, 품질 검사, 원자적 JSON 결과
- `pdf_parser/k8s/orchestrator.py` — Job CRUD, PVC 업로드(busybox), 결과 수집
- `pdf_parser/k8s/manifests/` — namespace, pvc(EFS RWX 20Gi), job-template

### 설계 결정
- 네임스페이스 분리: `pdf-parser` (rag-bench-test와 독립)
- PVC 단일: input/output 모두 pdf-storage (20Gi)
- Secret: `parser-secrets` (GEMINI_API_KEY, optional)
- ai-platform toleration 포함

### TODO
- `--split N` 배치 분할 시 파일 분배 로직 미구현
- Docling 모델 캐시 PVC 미구현
- Harbor ImagePullSecret 공유 검토 필요
