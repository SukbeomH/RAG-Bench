# Session: AutoRAG 크로스 프레임워크 벤치마크 스크립트 제작

- **Date**: 2026-02-12
- **Branch**: main
- **Type**: feature

## Summary
rag_bench와 AutoRAG를 동일 데이터로 비교하기 위한 크로스 프레임워크 벤치마크 스크립트 제작.

## Changes
| File | Action | Description |
|------|--------|-------------|
| `rag_bench/scripts/run_autorag.py` | **New** | AutoRAG 벤치마크 메인 스크립트 (~240 LOC) |
| `pyproject.toml` | Modified | `[project.optional-dependencies]` autorag 추가 |
| `MEMORY.md` | Modified | 세션 기록 추가 |

## Key Decisions
- AutoRAG는 LangChain 구버전 충돌 위험으로 optional dependency 분리
- 기존 `autorag_benchmark/data/` (100 QA) 보존, 새 데이터는 `data_ragbench/`에 저장
- 해시 기반 캐싱으로 중복 변환 방지

## Implementation Details
- 5단계 실행: Prerequisites → 데이터 변환 → 벤치마크 → 결과 분석 → 비교
- corpus.parquet: child_chunks → doc_id/contents/metadata
- qa.parquet: parent_id → child doc_ids 매핑으로 retrieval_gt 생성
- CLI: `--config dense|hybrid|PATH`, `--skip_convert`, `--compare`
