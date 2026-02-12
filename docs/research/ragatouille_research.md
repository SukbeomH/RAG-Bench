# RAGatouille & ColBERT 리서치

> ColBERT 기반 Late Interaction 검색 방식 — rag_bench 구현과의 비교 분석

---

## 1. RAGatouille 개요

**RAGatouille**은 ColBERT(Contextualized Late Interaction over BERT) 기반의 **Late Interaction 검색**을 쉽게 사용할 수 있도록 만든 Python 라이브러리이다.

- **GitHub**: [bclavie/RAGatouille](https://github.com/bclavie/RAGatouille)
- **목적**: 최신 정보 검색 연구와 실제 RAG 파이프라인 사이의 간극을 줄임
- **핵심 모델**: ColBERTv2 (`colbert-ir/colbertv2.0`)

### 설치

```bash
pip install -U ragatouille
```

### 주요 API

```python
from ragatouille import RAGPretrainedModel

# 모델 로드
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# 인덱싱
RAG.index(
    collection=["문서1 텍스트...", "문서2 텍스트..."],
    index_name="my_index",
    split_documents=True  # 자동 청킹
)

# 검색
results = RAG.search(query="검색 질문", k=5)
# [{'content': '...', 'score': 0.85, 'rank': 1}, ...]

# 인덱스에 문서 추가 (기존 인덱스 유지)
RAG.add_to_index(new_documents=["추가 문서..."])
```

### LangChain 통합

```python
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
retriever = RAG.as_langchain_retriever(k=5)
# LangChain 체인에서 바로 사용 가능
```

---

## 2. 핵심 기술: Late Interaction & MaxSim

### Late Interaction이란?

기존 검색 모델들은 쿼리와 문서를 **단일 벡터**로 압축하거나, **함께 처리**하여 관련도를 판단한다. ColBERT의 **Late Interaction**은 이 두 접근법의 중간에 위치한다.

```
┌─────────────────────────────────────────────────────┐
│                  ColBERT 작동 방식                      │
│                                                     │
│  Query: "한국어 RAG"    Document: "RAG는 검색 증강..."  │
│     ↓                        ↓                      │
│  [q₁] [q₂] [q₃]      [d₁] [d₂] [d₃] [d₄] [d₅]   │
│     ↓                        ↓                      │
│  토큰별 임베딩 생성      토큰별 임베딩 생성 (사전 계산)    │
│     ↓                        ↓                      │
│           MaxSim 연산 (Late Interaction)               │
│     q₁ → max(sim(q₁, d₁), sim(q₁, d₂), ...)        │
│     q₂ → max(sim(q₂, d₁), sim(q₂, d₂), ...)        │
│     q₃ → max(sim(q₃, d₁), sim(q₃, d₂), ...)        │
│     ↓                                               │
│     Score = Σ MaxSim(qᵢ)                            │
└─────────────────────────────────────────────────────┘
```

### MaxSim 연산

각 쿼리 토큰에 대해 **가장 유사한 문서 토큰과의 유사도**를 구하고, 이를 **합산**한다:

$$\text{Score}(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} \text{sim}(q_i, d_j)$$

이 방식으로 **토큰 수준의 정밀한 매칭**이 가능하면서도, 문서 임베딩은 **사전 계산**할 수 있어 효율적이다.

---

## 3. 검색 방식 비교

### Bi-Encoder vs ColBERT(Late Interaction) vs Cross-Encoder

| 특성 | Bi-Encoder | ColBERT (Late Interaction) | Cross-Encoder |
|------|-----------|---------------------------|---------------|
| **인코딩** | 쿼리·문서 → 각 1개 벡터 | 쿼리·문서 → 각 N개 토큰 벡터 | 쿼리+문서 → 함께 처리 |
| **문서 사전 계산** | ✅ 가능 | ✅ 가능 (토큰 단위) | ❌ 불가 |
| **상호작용 시점** | 없음 (독립 인코딩) | Late (검색 시 MaxSim) | Early (인코딩 시 함께) |
| **검색 정확도** | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| **검색 속도** | ★★★★★ | ★★★★☆ | ★☆☆☆☆ |
| **확장성** | 수억 문서 OK | 수천만 문서 OK | 수천 문서 한계 |
| **저장 공간** | 작음 (문서당 1벡터) | 큼 (문서당 N벡터) | 없음 |
| **대표 모델** | E5, BGE-M3, MiniLM | ColBERTv2, Jina-ColBERT | KoReranker, MonoT5 |
| **주 용도** | 1차 검색 (Retrieval) | 검색 + 재순위화 겸용 | 재순위화 (Reranking) |

### 핵심 차이 요약

```
정확도:  Cross-Encoder > ColBERT > Bi-Encoder
속도:    Bi-Encoder > ColBERT >> Cross-Encoder
저장:    Bi-Encoder < ColBERT << Cross-Encoder (인덱스 불필요)
```

**ColBERT의 포지션**: Bi-Encoder의 **속도**와 Cross-Encoder의 **정확도** 사이에서 **최적의 균형점**을 제공한다.

---

## 4. 현재 rag_bench 구현과의 차이점

현재 `rag_bench` 패키지의 DenseSparse 파이프라인과 ColBERT 방식을 비교한다.

### 파이프라인 구조 비교

| 구분 | 현재 rag_bench 파이프라인 | ColBERT 기반 파이프라인 |
|------|----------------------|----------------------|
| **1차 검색** | Dense (E5, BGE-M3 등) + BM25 | ColBERT 단독 또는 +BM25 |
| **Fusion** | HybridRRF / HybridCC 결합 | 불필요 (ColBERT가 Dense+Sparse 효과) |
| **Reranker** | KoReranker / FlashRank | 불필요 (ColBERT 자체 정밀 매칭) |
| **파이프라인 복잡도** | 4단계 (Dense→Sparse→Fusion→Rerank) | 1~2단계 (ColBERT → [선택적 Rerank]) |

### 성능 특성 비교

| 항목 | 현재 방식 (Dense+BM25+Reranker) | ColBERT 방식 |
|------|-------------------------------|-------------|
| **검색 정확도** | Fusion + Reranker로 높음 | 단일 모델로 비슷한 수준 달성 |
| **레이턴시** | 다단계 처리로 느림 | 단일 검색으로 빠름 |
| **인덱스 크기** | 문서당 1벡터 (작음) | 문서당 N벡터 (10~50x 큼) |
| **한국어 최적화** | ✅ 전용 모델 사용 (KoSimCSE 등) | ⚠️ 한국어 ColBERT 모델 제한적 |
| **파라미터 튜닝** | 많음 (가중치, 노말라이즈 등) | 적음 (모델 선택 + top_k) |

### 구체적 비교

```
현재 파이프라인:
Query → [Dense: E5 multilingual] → top-10
      → [Sparse: BM25 ko_kiwi]  → top-10
      → [HybridCC: 가중 결합]    → top-10
      → [KoReranker: 재순위화]   → top-5
      → Generator

ColBERT 파이프라인:
Query → [ColBERT: 토큰별 MaxSim] → top-5
      → Generator
```

---

## 5. 한국어 지원 현황

### ColBERT 한국어 모델 옵션

| 모델 | 한국어 지원 | 언어 수 | 차원 | 컨텍스트 | 특징 |
|------|-----------|--------|------|---------|------|
| **colbertv2.0** | ❌ 영어 전용 | 1 | 128 | 512 | 가장 안정적, 레퍼런스 모델 |
| **Jina-ColBERT-v2** | ✅ 89개 언어 | 89 | 128 | 8192 | 다국어 최강, 긴 컨텍스트 |
| **ColBERT-XM** | ✅ XMOD 기반 | 80+ | 128 | 512 | 제로샷 다국어 전이 |
| **ColBERTforKorean** | ✅ 한국어 특화 | 1 | 128 | 512 | 커뮤니티 프로젝트 (실험적) |

### 권장 모델

1. **Jina-ColBERT-v2** (최우선 추천)
   - 89개 언어 지원, 한국어 포함
   - 8192 토큰 긴 컨텍스트 지원
   - 임베딩 차원 축소 시에도 성능 유지
   - HuggingFace: `jinaai/jina-colbert-v2`

2. **ColBERT-XM** (대안)
   - XMOD 백본으로 언어별 어댑터 사용
   - 영어 파인튜닝만으로 다국어 제로샷 검색 가능
   - 새 언어 확장이 용이한 모듈러 아키텍처

### RAGatouille에서 다국어 모델 사용

```python
from ragatouille import RAGPretrainedModel

# Jina-ColBERT-v2 로드
RAG = RAGPretrainedModel.from_pretrained("jinaai/jina-colbert-v2")

# 한국어 문서 인덱싱
RAG.index(
    collection=["한국어 문서 내용...", "다른 문서..."],
    index_name="korean_index"
)

# 한국어 쿼리 검색
results = RAG.search(query="RAG 파이프라인이란?", k=5)
```

---

## 6. 장단점 분석

### 장점

| # | 장점 | 설명 |
|---|------|------|
| 1 | **파이프라인 단순화** | Dense+Sparse+Fusion+Reranker → ColBERT 단일 모델로 대체 가능 |
| 2 | **정밀한 토큰 매칭** | 단일 벡터가 놓치는 세밀한 용어 매칭 포착 |
| 3 | **Reranker 불필요** | Late Interaction이 Reranker 역할을 겸함 |
| 4 | **직관적 API** | RAGatouille로 3줄 코드에 검색 시스템 구축 |
| 5 | **LangChain 통합** | `as_langchain_retriever()`로 기존 체인에 바로 연결 |

### 단점

| # | 단점 | 설명 |
|---|------|------|
| 1 | **인덱스 크기** | 토큰별 벡터 저장으로 Dense 대비 10~50배 큰 인덱스 |
| 2 | **한국어 모델 한정** | 영어 ColBERTv2 대비 한국어 특화 모델이 적음 |
| 3 | **~~구현 완료~~** | `rag_bench`의 `ColBERTStrategy` 및 `ColBERTRerankStrategy`로 구현됨 |
| 4 | **GPU 의존성** | 인덱싱 시 GPU 필요 (CPU에서도 가능하나 느림) |
| 5 | **스케일 한계** | Bi-Encoder 대비 수억 문서 규모에서 비용 증가 |

---

## 7. 현재 프로젝트에 적용 시 고려사항

### 적용 시나리오

#### 시나리오 A: ColBERT를 Reranker로 사용 (보수적)

```
현재: Dense → BM25 → HybridCC → KoReranker → Generator
변경: Dense → BM25 → HybridCC → ColBERT Reranker → Generator
```

- **장점**: 기존 파이프라인 유지, Reranker만 교체
- **단점**: ColBERT의 장점을 일부만 활용

#### 시나리오 B: ColBERT를 주 검색기로 사용 (적극적)

```
현재: Dense → BM25 → HybridCC → KoReranker → Generator
변경: ColBERT → [선택적 BM25 보완] → Generator
```

- **장점**: 파이프라인 대폭 단순화, 레이턴시 감소
- **단점**: 인덱스 크기 증가, 한국어 성능 검증 필요

### rag_bench 벤치마크 연동 현황

> **구현 완료**: `rag_bench/strategies/colbert.py` (ColBERTStrategy) 및 `rag_bench/strategies/colbert_rerank.py` (ColBERTRerankStrategy)로 PyLate 기반 구현 완료.

- ColBERTStrategy: Jina-ColBERT-v2 기반 brute-force + Voyager 인덱스 모드 지원
- ColBERTRerankStrategy: 임의 1차 전략 위에 ColBERT MaxSim 리랭킹

### 벤치마크 결과 요약 (20 QA 기준)

- ColBERT 단독: 품질 최상위권 (context_recall 1.0), 레이턴시 ~670ms
- ColBERTRerank + BGE-M3: 종합 최고 품질 (context_recall 1.0, precision 0.95)
- 상세 결과: `rag_bench/scripts/bench_visualize.ipynb` 참조

---

## 8. 참고 자료

| 리소스 | URL |
|--------|-----|
| RAGatouille GitHub | https://github.com/bclavie/RAGatouille |
| ColBERTv2 논문 | https://arxiv.org/abs/2112.01488 |
| ColBERT 원본 논문 | https://arxiv.org/abs/2004.12832 |
| Jina-ColBERT-v2 | https://huggingface.co/jinaai/jina-colbert-v2 |
| ColBERT-XM 논문 | https://aclanthology.org/2024.findings-naacl.283 |
| LangChain + RAGatouille | https://python.langchain.com/docs/integrations/retrievers/ragatouille |
| Weaviate ColBERT 해설 | https://weaviate.io/blog/colbert |
| ColBERTforKorean | https://github.com/teddysum/ColBERTforKorean |
