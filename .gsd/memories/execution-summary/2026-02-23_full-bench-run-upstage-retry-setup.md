---
title: "2026-02-23 Full 벤치마크 실행 + Upstage 재시도 자동화 설정"
tags:
  - execution-summary
  - benchmark
  - upstage
  - auto-retry
  - full-preset
type: execution-summary
created: 2026-02-23T10:00:00+09:00
contextual_description: "60개 조합 full 벤치마크 로컬 실행. OpenAI 48개 성공, Upstage 12개 API키 누락 실패. .env에 UPSTAGE_API_KEY 추가 후 자동 후속 실행 대기 프로세스(PID 10047) 설정."
keywords:
  - full-preset
  - 60-combos
  - upstage
  - UPSTAGE_API_KEY
  - dense-filter
  - append-results
  - background-job
  - auto-wait
related:
  - 2026-02-23_dense-filter-append-results-timing-tools
---

## 2026-02-23 Full 벤치마크 실행 내역

### 실행 명령
```bash
.venv/bin/python -m rag_bench.scripts.run_all_combos \
  --preset full --k 3 --top_n 10 --layers --metric-preset core_only
```

### 구성
- **조합 수**: 60개 (Dense 5종 × Sparse 2종 × Mode 6종)
- **QA 수**: 20개 (`_benchdata/qa_dataset.json`)
- **백그라운드 Job ID**: bdb3ae0

### 실행 결과
| Dense 모델 | 조합 수 | 결과 |
|-----------|---------|------|
| text-embedding-3-small | 12 | 성공 |
| text-embedding-3-large | 12 | 성공 |
| bge-m3 | 12 | 성공 |
| intfloat/multilingual-e5-large | 12 | 성공 |
| upstage/solar-embedding-1-large | 12 | 실패 (UPSTAGE_API_KEY 미설정) |

### Upstage 실패 원인 및 해결
- **에러**: `ValueError: You must specify an api key`
- **원인**: `.env`에 `UPSTAGE_API_KEY` 미설정
- **해결**: 사용자가 `.env`에 키 추가

### Upstage 자동 재실행 설정
Main 벤치마크(bdb3ae0) 완료 감지 후 자동 실행 대기:
```bash
# 자동 대기 프로세스 PID: 10047
# 감지 문자열: "벤치마크 완료"
# 실행 예정 명령:
.venv/bin/python -m rag_bench.scripts.run_all_combos \
  --preset full --k 3 --top_n 10 --layers --metric-preset core_only \
  --dense-filter upstage --append-results
```

### 진행 상황 (세션 종료 시점)
- Pass 1: 60개 전략 완료 (OpenAI 48개 + 실패 12개 포함)
- Pass 2 RAGAS: ~21% 진행 (10개 상위 전략 평가 중, 7번째 완료)
- 현재 평가 중: `DS(text-embedding-3-large+korean_bm25)` 전략
