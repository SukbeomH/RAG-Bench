---
title: "4-Layer RAG 벤치마크 아키텍처 리팩토링 및 HTML 보고서 강화"
tags:
  - rag-bench
  - 4-layer
  - html-report
  - contextual-retrieval
  - architecture
type: execution-summary
created: 2026-02-23T05:08:18Z
contextual_description: "Contextual Retrieval을 Layer 4로 분리, 인덱싱 시간(build_s)으로 Layer 4 기여도 측정, HTML 보고서 방법론/설명 강화"
keywords:
  - 4-layer RAG benchmark Contextual Retrieval Layer4 build_s indexing HTML report shoelace radar
---

## 4-Layer RAG 벤치마크 아키텍처 리팩토링 및 HTML 보고서 강화

## 4-Layer RAG 벤치마크 아키텍처 리팩토링 및 HTML 보고서 강화

### 변경 배경
- 기존 3-Layer 구조에서 Contextual Retrieval을 Layer 3 리랭커와 같은 레이어로 묶었으나,
  Contextual은 인덱싱 단계 강화 기법으로 리랭커와 역할이 달라 Layer 4로 분리
- HTML 보고서에 설명이 부족하여 독자가 각 섹션의 의미를 파악하기 어려웠음

### 핵심 변경 사항

#### 아키텍처 (4-Layer)
- Layer 1: Dense 임베딩 모델 (5개: kosimcse/e5/bge-m3/openai-large/upstage)
- Layer 2: Sparse 검색 (2개: korean_bm25/splade)
- Layer 3: Reranker (3개: none/colbert/flashrank)
- Layer 4: Contextual Retrieval on/off (2개) ← 신규 분리
- 총 60개 = 5 × 2 × 3 × 2

#### Layer 4 메트릭 기준
- Layer 1~3: 쿼리 레이턴시 (avg_latency) 기준
- Layer 4: 인덱싱 소요시간 (build_s) 기준
- 이유: Contextual은 쿼리 레이턴시 추가 없이 인덱싱 비용만 발생

#### HTML 보고서 신규 함수
- _shorten_name(): 차트용 전략명 축약 (Contextual Retrieval → Ctx·, ColBERT → CB·, FlashRank → FR·)
- _radar_area_rank_html(): Shoelace 공식으로 레이더 면적 순위 계산
- _total_timing_table_html(): 인덱싱+검색+평가 합산 시간 순위 (Bootstrap 탭)
- _benchmark_methodology_html(): 2-Pass 방식 + 4-Layer×60개 설명 카드
- _layer4_timing_html(): Contextual OFF vs ON build_s 비교표

#### HTML 레이아웃 구조
- 벤치마크 방법론 섹션 (신규): 2-Pass 설명 + 조합표
- ① 속도 분석: 레이턴시 막대(col-7) + 순위 테이블(col-5) + 산점도(전체)
- ② RAGAS 종합 분석: 레이더+면적순위(col-5) + 테이블+지표해설(col-7)
- RAGAS 히트맵 제거 (테이블로 대체)
- 전략 유형 가이드 섹션 제거
- 전략 선택 가이드 (결론 섹션에 텍스트로 추가)

#### 파일별 변경
- combo/spec.py: 4-Layer 명세 업데이트
- scripts/run_all_combos.py: Layer 기여도 출력 함수 (Layer4=build_s 기준)
- strategies/contextual_retrieval.py: Layer 4 명시
- utils/report.py: 레이어 라벨 Layer 4 — Contextual
- README.md: 3-Layer → 4-Layer 전체 업데이트

### Contextual 캐시 공유 메커니즘
- 캐시 키 = 청크 내용 SHA-256 해시 (모델 조합과 무관)
- 동일 문서의 첫 Contextual 전략만 LLM 호출, 이후 전략은 캐시 재사용
- 실제 비용 = 고유 청크 수에만 비례 (전략 수 불관)

### 벤치마크 실행 상태
- 캐시 없이 full preset top_n=60으로 재실행 중 (PID: 79421)
- Qdrant DB 전체, contextual_cache.json, 이전 CSV 삭제 후 재시작
