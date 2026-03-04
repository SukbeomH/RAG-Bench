# AutoRAG Codebase Map (2026-03-04)

## 규모
- Python 파일: ~80개, 총 12,507 LOC
- 5 Python 패키지 + 1 Next.js 프론트엔드 (uv workspace 모노레포)

## 패키지 의존성 그래프
```
autorag-parsers  (독립)
    ↓
autorag-pdf-eval (→ parsers)
autorag-retrieval (독립)
    ↓
autorag-rag-eval (→ retrieval)
autorag-api (→ parsers + retrieval)
```

## 핵심 아키텍처 패턴

### 1. Registry + Lazy Loading
- `@register("name")` 데코레이터로 파서 등록
- `__init__.py`에서 `_LAZY_BACKENDS` 딕셔너리로 지연 import
- pymupdf만 즉시 로드, 나머지는 `get_parser()` 첫 호출 시 import

### 2. Bridge/Worker Subprocess 격리
```
Main Python → bridge.py → subprocess.run([venv_python, worker.py, pdf_path])
                           stdout: ---OUTPUT_START---\n{JSON}\n---OUTPUT_END---
                           stderr: 진행 메시지
```
대상: docling(.venv-docling), paddleocr(PaddleOCR venv), deepseek_ocr2(.venv-deepseek)

### 3. K8s 2-Phase 벤치마크
- Phase 1 (prep): HF 데이터 → 청킹 → Contextual 보강 → PVC 직렬화
- Phase 2 (bench): 인덱스 빌드 → Pass1(레이턴시) → Pass2(RAGAS) → 결과 JSON
- 오케스트레이터: kubectl Job 생성/폴링/결과 수집

### 4. OCR 서버 공통 패턴
- FastAPI + OpenAI-compatible `/v1/chat/completions`
- 비동기 모델 로딩 (lifespan), 동기 추론은 `run_in_executor`
- `/health`: 503(로딩) → 200(정상) → 500(에러)

## 레거시 / 기술 부채

### 잔존 레거시 디렉토리
| 경로 | 상태 | 비고 |
|---|---|---|
| `rag_bench/` | Python 소스 없음 | `_benchdata/`, `_models/` 캐시만 잔존 |
| `pdf_parser/` | Python 소스 없음 | `bench_results/`, `__pycache__` 잔존, `.py` 파일은 삭제됨 |
| `pdf_parser/category*.py` | 5개 파서 코드 | Dockerfile에서 flat COPY, 아직 autorag_parsers 미통합 |

### 오케스트레이터 참조 깨짐
- `orchestrators/rag_bench/orchestrator.py`: `rag_bench.scripts.merge_service_results` 참조 — 모듈 삭제됨, 호출 시 실패

### 테스트 부족
- 전체 e2e 테스트만 존재 (단위 테스트 없음)
- 패키지당 1개 테스트 파일

### 하드코딩된 로컬 경로
- `isolated_backends/deepseek_ocr2/`: SSL cert 경로 `/Users/sukbeom/Documents/cert/combined-ca-bundle.pem`
- `isolated_backends/paddleocr/`: `uv run` + 로컬 PaddleOCR 디렉토리 참조

### Docling 의존성 충돌
- `docling>=2.75` → `transformers>=4.49` 필요
- `langchain-upstage` → `tokenizers<0.21` 요구
- 해결: pyproject.toml에서 `[docling]` extra 분리, 로컬은 subprocess 격리, Docker에서만 직접 설치

## 주요 파일 맵
| 파일 | 역할 |
|---|---|
| `packages/pdf-parsers/src/autorag_parsers/_protocol.py` | PageResult, ConversionResult, PDFParser Protocol |
| `packages/pdf-parsers/src/autorag_parsers/registry.py` | 파서 팩토리 레지스트리 |
| `packages/pdf-parsers/src/autorag_parsers/__init__.py` | 지연 로딩 monkey-patch |
| `packages/rag-api/src/autorag_api/app.py` | FastAPI 엔트리포인트 |
| `orchestrators/pdf_parser/orchestrator.py` | PDF 벤치마크 K8s 오케스트레이터 |
| `orchestrators/rag_bench/orchestrator.py` | RAG 벤치마크 K8s 오케스트레이터 |
| `k8s_results/generate_k8s_report.py` | 벤치마크 보고서 생성기 |
| `conftest.py` | 루트 공유 pytest fixtures |
