# Session Handoff: Docling 격리 Backend + 코드베이스 분석

**Date**: 2026-03-04
**Branch**: master
**Last Commit**: `4452f17` feat: docling 격리 subprocess backend + 코드베이스 맵 추가

## 완료된 작업

### 1. Docling Isolated Backend 구현 ✅
의존성 충돌 (docling `transformers>=4.49` ↔ langchain-upstage `tokenizers<0.21`) 해결을 위한 subprocess 격리 실행.

**신규 파일:**
- `isolated_backends/docling/worker.py` — docling 독립 실행, per-page JSON stdout
- `isolated_backends/docling/bridge.py` — .venv-docling venv python → subprocess 실행 → dict[int, str]
- `isolated_backends/docling/setup_venv.sh` — 격리 venv 생성 (python3.12, docling>=2.75, pymupdf)

**수정 파일 (이전 커밋 `10ce3fa`):**
- `packages/pdf-parsers/src/autorag_parsers/docling.py` — convert() → try _convert_direct() / except ImportError → _convert_subprocess() fallback

**검증 완료:**
- worker 단독: 5페이지 JSON 정상 출력
- 통합 (get_parser('docling')): subprocess fallback → 5페이지 PageResult 생성, 22.9초

### 2. 코드베이스 분석 (codebase-mapper) ✅
- `.gsd/memories/codebase-map.md` — 전체 아키텍처, 패턴, 기술 부채 문서화
- MEMORY.md 200줄 이하로 압축 (상세 → benchmark-details.md 분리)

## 기술 부채 발견
- `orchestrators/rag_bench/orchestrator.py`: 삭제된 `rag_bench.scripts.merge_service_results` 참조 (호출 시 실패)
- `pdf_parser/` 레거시 디렉토리: Python 소스 없음, __pycache__/bench_results만 잔존
- 단위 테스트 없음 (e2e만 48개)

## 다음 세션 참고
- Phase 5 MinerU Pipeline 추가 가능 (spec.py에 "mineru" 예약)
- isolated_backends/docling/.venv-docling/ 은 gitignore됨 → 새 환경에서 setup_venv.sh 재실행 필요
- paddleocr bridge는 로컬 macOS 전용 (PaddleOCR/ 디렉토리 참조)
