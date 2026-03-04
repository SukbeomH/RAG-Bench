---
title: "Notion RAG Benchmark 페이지 생성 및 오류 수정"
tags:
  - session
  - notion
  - rag-benchmark
  - data-correction
type: session-summary
created: "2026-02-23T08:54:00+09:00"
contextual_description: "RAG 벤치마크 결과를 Notion 페이지에 정리하고, raw CSV 대조 검증으로 9개 셀 오류와 TOP 15 랭킹 오류(8~15위)를 발견 및 수정"
keywords:
  - notion
  - rag-benchmark
  - ragas
  - context-recall
  - data-validation
  - top15-ranking
  - replace_content
related: []
---

## Notion RAG Benchmark 페이지 생성 및 오류 수정

### 대상 페이지
- URL: https://www.notion.so/RAG-Benchmarks-310e4f18b43d80e983d8d1a8dc305974
- 부모: BE 02 홍석범

### 작업 내용
1. RAG 벤치마크 결과(61개 전략)를 Notion 페이지로 정리
2. raw CSV(`all_combos_ragas.csv`, `combo_timing.csv`)와 대조 검증
3. 오류 발견 및 수정

### 발견된 오류 (총 9건, 전부 context_recall 컬럼 or AnsRel)
| 전략 | 필드 | 잘못된 값 | 수정값 |
|---|---|---|---|
| DS(bge-m3+korean_bm25) | CtxRec | 0.797 | 0.800 |
| Contextual Retrieval (DS(bge-m3+korean_bm25)) | CtxRec | 0.813 | 0.800 |
| ColBERT Rerank (DS(bge-m3+korean_bm25)) | CtxRec | 0.840 | 0.813 |
| ColBERT Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25))) | CtxRec | 0.840 | 0.880 |
| ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25))) | CtxRec | 0.840 | 0.880 |
| Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)) | AnsRel | 0.887 | 0.890 |
| Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)) | CtxRec | 0.880 | 0.800 |
| ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25))) | CtxRec | 0.770 | 0.813 |
| Contextual Retrieval (DS(embedding-query+splade)) | CtxRec | 0.783 | 0.700 |

### 연쇄 오류: TOP 15 8~15위 전부 잘못됨
- 오류 원인: 잘못된 CtxRec 값이 평균 계산에 전파됨
- 올바른 8~15위 (수정 후):
  - 8: ColBERT Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25))) avg 0.901
  - 9: ColBERT Rerank (Contextual Retrieval (DS(embedding-query+splade))) avg 0.897
  - 10: ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25))) avg 0.896
  - 11: ColBERT Rerank (DS(KoSimCSE+splade)) avg 0.895
  - 12: ColBERT Rerank (DS(text-embedding-3-large+korean_bm25)) avg 0.894
  - 13: ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25))) avg 0.894
  - 14: ColBERT Rerank (DS(text-embedding-3-large+splade)) avg 0.892
  - 15: ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade))) avg 0.892

### 기술 트러블슈팅
- `replace_content_range` 타임아웃 → `replace_content` 전체 교체로 우회
- `replace_content_range` "Multiple occurrences found" → 동일 패턴이 TOP 15 + 전체 테이블에 중복 존재
- 해결: `replace_content`로 전체 페이지 내용 한 번에 교체 (성공)

### 최종 검증
- Notion 페이지 fetch 후 전체 테이블 육안 확인 완료
- 모든 수정값이 raw CSV와 일치함
