# Session Handoff Document

> 작성일: 2026-02-25 | 최종 커밋: `4397ccc` | 브랜치: `master` (clean, pushed)

---

## 1. 프로젝트 개요

**RAG Bench** — 문서 유형별 최적 Dense×Sparse 임베딩 조합을 찾기 위한 서비스 벤치마크 시스템.

- 5개 문서 카테고리: GENERAL / LEGAL / BUSINESS / MEDICAL / TECHNICAL
- 6개 조합: 3 Dense(kosimcse, e5, bge-m3) × 2 Sparse(korean_bm25, splade)
- 고정 파이프라인: ColBERT Reranker + Contextual Retrieval
- 평가: RAGAS core_only (Recall 0.35, Precision 0.30, Faithfulness 0.20, Relevancy 0.15)

---

## 2. 현재 진행 상황

### 실행 중인 프로세스
| PID | 명령 | 상태 |
|-----|------|------|
| 57323 | `run_service_bench --mode hf --max_queries 20 --categories general,legal,business` | 실행 중 (백그라운드) |

### 카테고리별 벤치마크 진행 현황
| 카테고리 | Pass 1 (Retrieval) | Pass 2 (RAGAS) | HF 데이터셋 |
|----------|-------------------|----------------|-------------|
| GENERAL | 실행 중 | 미완료 | klue/klue (mrc) |
| LEGAL | 실행 중 | 미완료 | yjoonjang/markers_bm (law) |
| BUSINESS | 실행 중 | 미완료 | yjoonjang/markers_bm (finance+public+commerce) |
| MEDICAL | **완료** | 미완료 | xhluca/publichealth-qa (korean) |
| TECHNICAL | 미실행 | 미실행 | sionic-ai/nanobeir-ko (NanoSCIDOCS) ← **신규** |

### 코드 상태
- 워킹 디렉토리: **clean** (커밋할 사항 없음)
- remote와 동기화: **완료** (origin/master = local master)
- `data/` 디렉토리: untracked (gitignore 대상, 벤치마크 결과 저장)

---

## 3. 최근 세션 작업 요약 (2026-02-24 ~ 02-25)

### 02-24: Notion 문서 체계화
- 기존 3페이지 → **5페이지**로 분할 (root: `310e4f18b43d80e983d8d1a8dc305974`)
- RAGAS 가중치 일관성 확보 (0.25 균등 → 0.35/0.30/0.20/0.15 실제값)
- snowflake-ko 제외, MIRACL→klue 변경사항 전 페이지 반영
- `contextual_llm` 기본값: `"gpt-4o-mini"` → `config.DEFAULT_CONTEXTUAL_LLM` 동적 로드

### 02-25: TECHNICAL 데이터셋 로더 구현
- `sionic-ai/nanobeir-ko` (NanoSCIDOCS) 선정 및 로더 구현
- 데이터 구조: 3 config(corpus/queries/qrels) × split='NanoSCIDOCS', streaming=True
- 5개 카테고리 전체 HF 데이터셋 확보 완료
- 영어 코드/API 데이터셋 조사 완료 (1순위: CoIR StackOverflowQA)

---

## 4. 남은 작업 (우선순위순)

### P0: 즉시 필요
1. **현재 벤치마크 완료 확인** — PID 57323 (general/legal/business) 완료 대기
   - 결과 확인: `ls rag_bench/_benchdata/` 또는 로그 확인
2. **MEDICAL Pass 2 RAGAS 평가 실행**
   ```bash
   uv run python -m rag_bench.scripts.run_service_bench --mode hf --max_queries 20 --categories medical
   ```
3. **TECHNICAL 카테고리 벤치마크 실행**
   ```bash
   uv run python -m rag_bench.scripts.run_service_bench --mode hf --max_queries 20 --categories technical
   ```

### P1: 후속 작업
4. **전체 결과 병합 및 리포트** — `rag_bench/scripts/merge_results.py` 활용
5. **Notion 문서 업데이트** — TECHNICAL 카테고리 상태: "HF 없음" → "nanobeir-ko/NanoSCIDOCS"
6. **영어 코드 데이터셋 적용 검토** — CoIR StackOverflowQA (~2K queries, ~20K corpus)

### P2: 개선 사항
7. `types.py` TECHNICAL metadata 업데이트 (`has_hf_dataset` 플래그 등)
8. 벤치마크 결과 기반 카테고리별 최적 조합 선정

---

## 5. 핵심 파일 맵

```
rag_bench/
├── datasets/
│   ├── hf_loader.py          # HF 데이터셋 로더 (5종 전체)
│   └── spec.py                # 데이터셋 스펙 정의
├── document_types/
│   └── types.py               # DocType enum + 카테고리 메타데이터
├── analysis/
│   ├── ranker.py              # RAGAS_WEIGHTS 상수 정의
│   ├── reporter.py            # HTML 리포트 생성
│   ├── reporter_exec.py       # 리포트 실행 엔트리
│   └── pipeline.py            # 분석 파이프라인 (리팩토링됨)
├── scripts/
│   └── run_service_bench.py   # 서비스 벤치마크 오케스트레이터
├── strategies/                # 검색 전략 (dense_sparse, colbert 등)
├── runner.py                  # 벤치마크 러너
├── config.py                  # 전역 설정
└── _benchdata/                # 벤치마크 결과 캐시
```

---

## 6. 주의 사항

### HF 데이터셋 로드 시
- 모든 로더는 `streaming=True` 필수 (CAS 다운로드 오류 우회)
- `trust_remote_code=True` 사용 금지 (deprecated)
- nanobeir-ko는 config ≠ split 이름임 (config='corpus', split='NanoSCIDOCS')

### 벤치마크 실행 시
- 체크포인트 시스템 있음 — 완료된 조합은 자동 스킵
- `--max_queries 20`으로 샘플링 실행 중 (전체 실행은 시간 소요 큼)
- MEDICAL은 총 77 QA pair로 소규모 → 100% 샘플링

### Notion 문서
- root: `310e4f18b43d80e983d8d1a8dc305974`
- MCP 도구로 접근 (`mcp__notion__*`)
- 5개 하위 페이지 구조 (레퍼런스/데이터셋/조합전략/실행현황/구현현황)

### 환경
- Python 3.12, uv 패키지 매니저
- Ollama LLM (로컬) — contextual_llm은 config에서 동적 로드
- RAGAS v0.4+ (`.scores` 리스트 순회 방식)

---

## 7. 참조 문서

| 문서 | 경로/URL |
|------|----------|
| 프로젝트 메모리 | `MEMORY.md` |
| 벤치마크 계획 | `PLAN_SERVICE_BENCH.md` |
| 범위 가이드라인 | `docs/benchmark_scope_guideline.md` |
| Notion root | https://www.notion.so/310e4f18b43d80e983d8d1a8dc305974 |
| nanobeir-ko | https://huggingface.co/datasets/sionic-ai/nanobeir-ko |
