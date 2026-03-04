# Session Handoff: 레거시 코드 마이그레이션 + 삭제 완료

## Date: 2026-03-04
## Branch: master
## Status: COMPLETE

## 완료 작업

### 7-커밋 마이그레이션 계획 전체 실행 완료

1. **rag_bench 워커 모듈 → autorag_retrieval 이전**
   - `runner.py`, `run_tracker.py`, `datasets/hf_loader.py`, `document_types/types.py`
   - 모든 내부 import 경로 `rag_bench.*` → `autorag_retrieval.*` / `autorag_rag_eval.*` 변경

2. **analysis/ranker 상수 추출**
   - `RAGAS_WEIGHTS`, `RAGAS_COLS` → `autorag_rag_eval/constants.py`
   - `k8s_results/generate_k8s_report.py` import 경로 수정

3. **pdf_parser/benchmark/ → autorag_pdf_eval 이전**
   - `evaluator.py`, `spec.py`, `runner.py` 복사 + import 경로 수정

4. **K8s 엔트리포인트 import 경로 업데이트**
   - `orchestrators/rag_bench/worker_entrypoint.py` — 9개 import
   - `orchestrators/pdf_parser/entrypoint.py` — 2개 import
   - `orchestrators/pdf_parser/orchestrator.py` — 1개 import (추가 발견)

5. **Dockerfile COPY 경로 + requirements 이동**
   - `Dockerfile`: packages/에서 pip install 방식으로 전환
   - `Dockerfile.pdf-parser`: autorag-pdf-eval + autorag-parsers 패키지 설치, `data/benchmark_pdfs/` 경로
   - deploy 스크립트 4곳 Dockerfile 경로 업데이트
   - requirements 파일 `k8s/` → `deploy/k8s/` 복사

6. **레거시 디렉토리 삭제**
   - `rag_bench/` — 60+ 파일 git rm
   - `k8s/` — 27 파일 git rm
   - `pdf_parser/benchmark/` — 4 파일 git rm
   - `pdf_parser/` 미사용 파일 5개 삭제 (complex, hybrid, smart_router, quality_checker, generate_report)

7. **검증 완료**
   - 모든 패키지 import 테스트 통과 (uv run python -c ...)
   - 레거시 참조 0건 (scripts/verify_* 제외 — 별도 정리 대상)

## 변경 규모
- 112 files changed, 561 insertions(+), 23,318 deletions(-)

## 남은 작업 / 후속 세션
- `scripts/verify_*.py` — 레거시 `rag_bench.*` import 참조 존재, 검증 스크립트 업데이트 또는 삭제 필요
- Dockerfile 실제 빌드 테스트 (K8s 환경)
- 의존성 최적화 (langchain-community 등 미사용 패키지 확인)
- `pdf_parser/` 내 유지 중인 category*.py 5개 → 향후 autorag_parsers 통합 검토
