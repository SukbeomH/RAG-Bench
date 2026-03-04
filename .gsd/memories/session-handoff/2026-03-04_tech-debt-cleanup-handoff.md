# Session Handoff: 기술부채 일괄 정리

**Date**: 2026-03-04
**Branch**: master
**Status**: 완료, 커밋 대기

## 완료된 작업

### Phase A: 런타임 버그 수정
1. `orchestrators/rag_bench/orchestrator.py` — 삭제된 `rag_bench.scripts.merge_service_results` subprocess 호출 블록 제거 (lines 473-492)
2. `packages/rag-api/src/autorag_api/routers/parse.py` — `get_parser()` KeyError → `HTTPException(400)` + 가용 백엔드 목록 포함

### Phase B: 하드코딩 경로 → 환경변수 (`SSL_CERT_BUNDLE`)
5개 파일에서 `/Users/sukbeom/Documents/cert/combined-ca-bundle.pem` 하드코딩 제거:
- `isolated_backends/deepseek_ocr2/bridge.py`
- `isolated_backends/deepseek_ocr2/worker.py`
- `isolated_backends/paddleocr/bridge.py`
- `isolated_backends/paddleocr/worker.py`
- `packages/pdf-eval/src/autorag_pdf_eval/runner.py`

패턴: `os.environ.get("SSL_CERT_BUNDLE", "")` → 값이 있고 파일 존재 시 SSL_CERT_FILE/REQUESTS_CA_BUNDLE 설정

### Phase C: 레거시 디렉토리 삭제
- `pdf_parser/` — bench_results 캐시 + benchmark_pdfs (data/에 이미 존재)
- `rag_bench/` — _benchdata(770MB) + _models(162MB) + 빈 Python 모듈

### Phase D: 단위 테스트 77개 추가
| 파일 | 테스트 수 | 대상 |
|------|-----------|------|
| `packages/pdf-parsers/tests/test_unit_registry.py` | 13 | registry, chunking, ChunkConfig |
| `packages/pdf-eval/tests/test_unit_eval.py` | 29 | levenshtein, NED, TEDS, BenchSpec, BenchResult |
| `packages/rag-retrieval/tests/test_unit_retrieval.py` | 12 | ComboSpec, presets, validation |
| `packages/rag-eval/tests/test_unit_rag_eval.py` | 12 | RAGAS weights, MetricRegistry, presets |
| `packages/rag-api/tests/test_unit_api.py` | 7 | schemas, parse endpoint 400 |

기존 E2E 테스트 수정: `test_e2e_api.py::test_parse_invalid_backend` — `pytest.raises(Exception)` → 400 응답 검증

### Phase E: 메모리 정리
- `MEMORY.md` 기술부채 섹션: 해결된 6건 제거, Docling 충돌만 잔존
- MinerU Pipeline 참조 제거
- 테스트 현황 업데이트 (144 tests)

## 검증 결과
- **144 tests passed, 0 failed** (67.81s)
- 하드코딩 경로 grep: `packages/`, `isolated_backends/`, `orchestrators/` 모두 clean (.next 빌드 아티팩트 제외)

## 미처리 항목
- **Docling 의존성 충돌**: subprocess 격리로 현상 유지 (해결 불필요)
- `.next/` 빌드 아티팩트에 절대 경로 포함 — 무해 (빌드 시 자동 생성)

## 다음 세션 주의사항
- `SSL_CERT_BUNDLE` 환경변수가 설정되어야 SSL 인증서가 적용됨 (미설정 시 기본 CA 사용)
- 레거시 `pdf_parser/`, `rag_bench/` 디렉토리 참조가 다른 스크립트에 있으면 에러 발생 가능 (orchestrator 외에는 없음 확인 완료)
