# RAG 벤치마크 전략 및 도구 추천 (2025)

본 문서는 `raghub_ecosystem_research.md` 분석 및 최신 웹 리서치(2025년 트렌드)를 바탕으로, `rag_bench` 라이브러리에 도입할 **검색(Retrieval) 및 인덱싱(Indexing)** 전략을 제안합니다.

특히 Python 버전 호환성을 고려하여 **ColBERT 구현체로 `pylate`를 채택**하는 내용을 포함합니다.

## 1. 요약: 도입 우선순위

| 우선순위 | 전략/도구 | 역할 | 구현 난이도 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **High** | **Hybrid Search** | BM25 + Vector | 낮음 | RAG의 기본 베이스라인. |
| **High** | **ColBERT (`pylate`)** | Late Interaction (Reranking/Retrieval) | 중간 | 한국어 성능 우수 (`jina-colbert-v2` 등 활용). |
| **Medium** | **GraphRAG (`LightRAG`)** | 지식 그래프 기반 추론 | 높음 | 복잡한 질의(Multi-hop) 처리에 필수. |
| **Medium** | **Contextual Retrieval** | 인덱싱 보강 (Anthropic 방식) | 중간 | 검색 정확도 향상 기법. |
| **Low** | **FlashRank** | 경량 리랭킹 | 낮음 | 속도 중요 시나리오에서 비교군으로 활용. |

---

## 2. 상세 전략 및 도구 분석

### 2.1 ColBERT: `pylate` 기반 구현

기존 `RAGatouille` 대신 `pylate`를 사용하여 ColBERT를 구현합니다.

*   **선정 이유:**
    *   **호환성:** `sentence-transformers` 기반으로 구축되어 Python 버전 및 의존성 충돌 이슈가 적음.
    *   **유연성:** 학습(Fine-tuning) 및 추론(Inference) 모두에 강점이 있으며, HNSW/FastPLAID 인덱싱 지원.
    *   **모델 지원:** `jina-colbert-v2` (한국어 지원 강력함) 및 `yjoonjang/colbert-ko-v1` 등 최신 모델 로드 용이.

*   **구현 방향 (`rag_bench/strategies/colbert.py`):**
    *   `pylate` 라이브러리를 사용하여 Document 인덱싱 및 검색 인터페이스 구현.
    *   **Reranker 모드:** 1차 검색(BM25/Vector) 후 상위 문서 재정렬 용도로 우선 구현 추천.
    *   **End-to-End Retriever 모드:** 추후 PLAID 인덱스 통합 시 고려.

### 2.2 Hybrid Search (BM25 + Vector)

벡터 검색의 의미적 유사성과 키워드 검색의 정확성을 결합하는 2025년 표준 전략입니다.

*   **구성:**
    *   **Sparse:** `rank_bm25` (표준, 속도 빠름) 또는 `SPLADE` (Sparse Vector, 성능 높음).
    *   **Dense:** 기존 `sentence-transformers` 임베딩 (e.g., `bge-m3`, `openai`).
*   **벤치마크 포인트:**
    *   `Alpha` 값(가중치) 조절에 따른 성능 변화 (0.0 ~ 1.0).
    *   단일 검색 대비 Hybrid 검색의 Recall@K 향상 폭.

### 2.3 GraphRAG: `LightRAG` vs `NodeRAG`

단순 검색으로 찾기 힘든 "전체적인 맥락"이나 "연결된 정보"를 찾기 위한 전략입니다.

*   **도구 비교:**
    *   **LightRAG:** 구조가 단순하고 빠름. 핵심적인 그래프 기능(Dual-level retrieval) 제공. 벤치마크 도입에 적합.
    *   **NodeRAG:** 노드 타입이 다양하고 복잡함. 연구용으로 적합하나 구현 복잡도가 높음.
*   **추천:**
    *   우선 **LightRAG** 구조를 차용하여 `rag_bench/strategies/graph_rag.py`에 구현.
    *   Graph 구축 비용(LLM 토큰 소모) 대비 검색 성능 이득 분석.

### 2.4 Advanced Indexing: Contextual Retrieval

Anthropic이 제안한 기법으로, 문서를 청킹할 때 "문맥(Context)"을 잃어버리는 문제를 해결합니다.

*   **방법:** 각 청크 앞에 "이 청크는 [문서 제목]의 [섹션]에 대한 내용으로..." 형태의 요약을 LLM으로 생성 하여 붙임.
*   **벤치마크 포인트:**
    *   일반 청킹 vs Contextual 청킹의 검색 정확도 비교.
    *   인덱싱 시간/비용 증가 대비 효과성 검증.

---

## 3. 추천 벤치마크 시나리오 (To-Do)

`rag_bench` 패키지 개발 시 다음 순서로 기능을 확장하는 것을 추천합니다.

1.  **Phase 1: Hybrid Search & Reranking**
    *   BM25 + Vector 조합 구현.
    *   `pylate` 기반 ColBERT Reranker 구현 및 `jina-colbert-v2` 모델 테스트.
2.  **Phase 2: Advanced Indexing**
    *   Contextual Retrieval 전처리 로직 추가.
3.  **Phase 3: Graph Search**
    *   LightRAG 스타일의 그래프 인덱싱 및 검색 전략 추가.

## 4. 참고 자료
*   **pylate GitHub:** [https://github.com/LightOn/pylate](https://github.com/LightOn/pylate)
*   **Contextual Retrieval:** [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval)
*   **Jina ColBERT v2:** [HuggingFace](https://huggingface.co/jinaai/jina-colbert-v2)
