---
title: "벤치마크 코퍼스 변경 시 데이터 오염 위험 — 반드시 인덱스/캐시 초기화"
tags:
  - pattern-discovery
  - benchmark
  - contamination
  - qdrant
  - cache
  - sample-pages
type: pattern-discovery
created: 2026-02-23T12:00:00+09:00
contextual_description: "벤치마크 대상 문서(코퍼스)를 변경할 때 기존 Qdrant 인덱스/contextual_cache.json/qa_dataset.json을 삭제하지 않으면 구 데이터로 오염된 결과가 나온다. --reindex 없이 실행 시 기존 인덱스 자동 재사용."
keywords:
  - contamination
  - qdrant_db
  - contextual_cache.json
  - qa_dataset.json
  - sample-pages
  - reindex
  - IndexCacheManager
related:
  - 2026-02-23_dense-filter-append-results-timing-tools
  - 2026-02-23_pdf-page-sampling-run-all-combos
---

## 벤치마크 코퍼스 변경 시 데이터 오염 패턴

### 상황
`--sample-pages`로 PDF 샘플링 후 재실행 시, `rag_bench/docs/*.md`는 교체되지만
기존 Qdrant 인덱스/캐시가 그대로 남아 **구 전체 내용 기준으로 검색**이 일어남.

### 오염 경로
```
IndexCacheManager.get_or_build()
  → qdrant_dir.exists() and any(qdrant_dir.iterdir())  → True (구 인덱스 존재)
  → "기존 인덱스 재사용" 출력 후 구 데이터에 연결
  → 샘플된 child_chunks는 전달되지만 실제 검색은 구 인덱스에서 수행
```

### 삭제 대상 (코퍼스 변경 시 필수)

| 파일/디렉토리 | 이유 |
|--------------|------|
| `_benchdata/qdrant_db_*/` | 구 문서 기준 벡터 인덱스 |
| `_benchdata/contextual_cache.json` | 구 문서 기준 LLM 컨텍스트 캐시 |
| `_benchdata/qa_dataset.json` | 구 문서 기준 QA 쌍 |
| `_benchdata/parent_store/` | (선택) 구 parent 청크 — 청킹 재실행 시 자동 교체됨 |

### 클린 재실행 명령
```bash
# 오염 데이터 제거
rm -rf rag_bench/_benchdata/qdrant_db_*
rm -f rag_bench/_benchdata/contextual_cache.json
rm -f rag_bench/_benchdata/qa_dataset.json

# 샘플링 + 재실행 (인덱스 없으면 --reindex 불필요)
python -m rag_bench.scripts.run_all_combos \
  --preset full --sample-pages --regenerate-qa ...
```

### 안전한 재실행 순서
1. 오염 파일 삭제
2. `--sample-pages` (docs/*.md 교체)
3. `--regenerate-qa` (QA 새로 생성)
4. 인덱스가 없으므로 자동으로 새 인덱스 빌드
