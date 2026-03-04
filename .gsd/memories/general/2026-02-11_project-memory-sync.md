---
title: "Project Memory Sync: MEMORY.md → GSD"
tags:
  - project-context
  - architecture
  - decisions
  - branch:main
type: project-context
created: 2026-02-11T12:00:00Z
contextual_description: "MEMORY.md 전체 내용을 GSD 메모리 시스템에 동기화. 프로젝트 제작 의도, 아키텍처, 해결된 이슈, 리서치 현황 포함."
keywords:
  - rag_bench
  - enterprise-rag-benchmark
  - strategy-pattern
  - ragas
  - autorag
  - dense-sparse
  - colbert
  - graphrag
  - noderag
  - raghub
---

## Project Context: 엔터프라이즈 RAG 벤치마크

### 제작 의도
엔터프라이즈 레벨에서 사용할 RAG 아키텍처/구성을 테스트하고 성능을 비교하기 위한 프로젝트.
`rag_bench/` 패키지에 모델별/구성별 RAG 전략을 Strategy Pattern으로 추가하고, AutoRAG + ragas로 정량 비교.

### 아키텍처
- `BaseRAGStrategy` ABC → `index()`, `retrieve()`, `get_retriever()` 인터페이스
- `BenchmarkRunner` — 동일 쿼리 세트로 전략 간 레이턴시/품질 비교
- `RAGEvaluator` — ragas 기반 (Faithfulness, Answer Relevancy, Context Precision/Recall)
- LangGraph Agent — 전략 주입 Agentic RAG 대화

### 구현 상태 (2026-02-11 기준)
| 구성 요소 | 상태 |
|-----------|------|
| `DenseSparseStrategy` (6가지 조합) | 구현 완료 |
| `ColBERTStrategy` (RAGatouille) | 스텁 |
| `GraphRAGStrategy` (NodeRAG/LightRAG) | 스텁 |
| `RAGEvaluator` (ragas) | 구현 완료 |
| `BenchmarkRunner` | 구현 완료 |
| LangGraph Agentic RAG | 구현 완료 |
| PDF→Markdown→Parent-Child 청킹 | 구현 완료 |

### 임베딩 조합 6종
1. 한국어 최적: KoSimCSE-roberta + BM25/OKt
2. 다국어 균형: E5-large + SPLADE
3. 올인원 통합: BGE-M3 + BGE-M3
4. 경량/빠른: MiniLM-L6 + BM25
5. 고성능 API: OpenAI Large + SPLADE
6. 한국어 API: Upstage Solar + BM25/OKt

---

## 세션 1: RAG Bench 검증 및 환경 구성

### 해결된 이슈
1. **LangChain 버전 호환성**: `autorag` 패키지 의존성 충돌 → `pyproject.toml`에서 제거하고 직접 명시
2. **SSL 인증서**: 보안 네트워크 HF Hub SSL 실패 → `HF_HUB_DISABLE_SSL_VERIFY=1` 등 환경변수 추가
3. **Qdrant 파일 락**: 클라이언트 중복 초기화 → `_init_qdrant` 클라이언트 재사용 로직

### 검증 통과
- 패키지 Import, Parent-Child 청킹, 인덱싱/검색 (MiniLM+BM25), Runner, LangGraph Agent

---

## 세션 2: RAGAS 평가 통합

### 해결된 이슈
1. **OpenAI API SSL/Proxy**: `httpx.Client(verify=False)` 주입으로 우회
2. **Ragas v0.4+ 결과 객체**: `.scores` 리스트 순회 → 평균 dict 반환으로 수정

### 검증 통과
- `scripts/verify_ragas_eval.py`: Mock 전략 평가 파이프라인 정상 동작
- `scripts/verify_env.py`: `.env` OPENAI_API_KEY 로드 확인

---

## 세션 3: RAGHub 생태계 분석 및 프로젝트 컨텍스트 정립

### 주요 산출물
- `docs/research/raghub_ecosystem_research.md` — RAG 생태계 90개+ 도구 7개 카테고리 분석

### RAG 트렌드 (2024-2025)
1. GraphRAG 부상 (LightRAG, cognee, TrustGraph)
2. 멀티모달 RAG (ColPali, Chunkr)
3. RAG-as-a-Service (Dcup, Ragie.ai, 클라우드 빅3)
4. 평가 도구 성숙 (ragas, Trulens, AutoRAG)
5. DB-Native RAG (PostgresML, pgai, Korvus)

### 벤치마크 추가 후보
- 검색/리랭킹: RAGatouille(ColBERT), Flash-Rank, ZeroEntropy
- 청킹: Chonkie, zchunk
- 평가: Trulens, Vectara HHEM

### 리서치 문서 현황
| 파일 | 주제 |
|------|------|
| `docs/research/ragatouille_research.md` | ColBERT/RAGatouille Late Interaction |
| `docs/research/noderag_research.md` | NodeRAG 이질적 그래프 기반 RAG |
| `docs/research/raghub_ecosystem_research.md` | RAG 생태계 전체 조감도 |
