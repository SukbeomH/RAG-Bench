# Current Session Context

## Session Narrative
> On 2026-03-04, the developer completed a bulk technical debt cleanup on the **master** branch. 7 of 8 documented tech debt items resolved: runtime bugs fixed, hardcoded paths → env vars, legacy dirs deleted, 77 unit tests added. Docling dependency conflict kept as-is (subprocess isolation).

## Context Snapshot
- **Active Task**: 기술부채 일괄 정리 완료
- **Branch**: master
- **Files Changed**: 12 modified + 8 new files
- **Last Updated**: 2026-03-04

## Working Files
```
 M .gitignore
 M isolated_backends/deepseek_ocr2/bridge.py
 M isolated_backends/deepseek_ocr2/worker.py
 M isolated_backends/paddleocr/bridge.py
 M isolated_backends/paddleocr/worker.py
 M orchestrators/rag_bench/orchestrator.py
 M packages/pdf-eval/src/autorag_pdf_eval/runner.py
 M packages/rag-api/src/autorag_api/routers/parse.py
 M packages/rag-api/tests/test_e2e_api.py
 D pdf_parser/ (legacy, deleted)
 D rag_bench/ (legacy, deleted)
?? packages/pdf-parsers/tests/test_unit_registry.py
?? packages/pdf-eval/tests/test_unit_eval.py
?? packages/rag-retrieval/tests/test_unit_retrieval.py
?? packages/rag-eval/tests/test_unit_rag_eval.py
?? packages/rag-api/tests/test_unit_api.py
```

## Recent Commits
```
4452f17 feat: docling 격리 subprocess backend + 코드베이스 맵 추가
10ce3fa fix: docling 파서 subprocess fallback + per-page 출력 정리
```

## Test Results
- 144 tests passed (67 E2E + 77 unit), 0 failed
- `uv run pytest packages/*/tests/ -v --import-mode=importlib`
