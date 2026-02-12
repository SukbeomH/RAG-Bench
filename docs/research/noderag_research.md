# NodeRAG 리서치

> 이질적 그래프 기반 RAG 프레임워크 — 실적용 분석 및 기존 구현 비교

---

## 1. NodeRAG 개요

**NodeRAG**는 이질적 그래프(Heterogeneous Graph) 구조를 활용하여 RAG의 검색 정밀도와 효율성을 높이는 **그래프 중심 RAG 프레임워크**이다.

- **논문**: "NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes" (arXiv, 2025.04)
- **GitHub**: [Terry-Xu-666/NodeRAG](https://github.com/Terry-Xu-666/NodeRAG)
- **최초 안정 버전**: v0.1.0 (2025.03.18)

### 핵심 동기

기존 그래프 기반 RAG (GraphRAG, LightRAG)는 **그래프 구조 설계를 충분히 고려하지 않아** 비효율성과 불일치가 발생한다. NodeRAG는 **이질적 노드 타입**을 도입하여 LLM 능력과 밀접하게 정렬된 그래프를 구축한다.

### 핵심 장점 요약

| 항목 | NodeRAG 성과 |
|------|-------------|
| **MuSiQue 정확도** | 46.29% (GraphRAG 41.71%, LightRAG 36.00%) |
| **HotpotQA 정확도** | 89.50% (GraphRAG 89.00%, 토큰 24% 절감) |
| **RAG-QA Arena** | 94.9% 검색 비율 (GraphRAG 86.3%) |
| **쿼리 시간** | 3.98s (GraphRAG 26.69s, LightRAG 5.58s) |

---

## 2. 7가지 이질적 노드 타입

NodeRAG의 핵심은 **7가지 노드 타입**으로 구성된 이질적 그래프(Heterograph)이다.

| 타입 | 기호 | 설명 | 역할 |
|------|------|------|------|
| **Entity** | N | 이름 있는 객체, 인물, 개념 | 지식 그래프의 기본 단위 |
| **Relationship** | R | 엔티티 간 연결 관계 | 엔티티↔의미단위 연결 |
| **Semantic Unit** | S | 지역 이벤트·아이디어의 의역 요약 | 원문의 의미 압축 |
| **Attribute** | A | 엔티티·관계의 LLM 생성 요약 | 노드 설명 강화 |
| **High-level Element** | H | 커뮤니티 수준 의미 요약 | 글로벌 컨텍스트 제공 |
| **Overview** | O | High-level Element의 키워드 제목 | 이중 검색 엔트리 포인트 |
| **Text Chunk** | T | 원본 텍스트 청크 | 전체 컨텍스트 보존 |

### 노드 타입별 관계도

```
                    ┌───────────────┐
                    │   Overview(O) │ ← 키워드 제목
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ High-level(H) │ ← 커뮤니티 요약
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
      ┌───────▼──┐   ┌─────▼─────┐   ┌──▼────────┐
      │ Entity(N)│───│Relation(R)│───│ Entity(N) │
      └───────┬──┘   └───────────┘   └──┬────────┘
              │                          │
      ┌───────▼──────┐          ┌───────▼──────┐
      │ Attribute(A) │          │ Attribute(A) │
      └──────────────┘          └──────────────┘
              │                          │
      ┌───────▼──────────────────────────▼──────┐
      │         Semantic Unit(S)                │
      └─────────────────┬──────────────────────┘
                        │
               ┌────────▼────────┐
               │  Text Chunk(T)  │ ← 원문 보존
               └─────────────────┘
```

---

## 3. 5단계 인덱싱 파이프라인

### 3단계 그래프 구축 프로세스

#### Stage 1: Graph Decomposition (그래프 분해)

LLM이 텍스트를 구조화된 요소로 분해한다:

```
원본 텍스트 → LLM 처리 → Semantic Units(S) + Entities(N) + Relationships(R)
```

- 텍스트에서 **의미 단위(S)** 추출 (이벤트, 사실 단위 의역)
- **엔티티(N)** 식별 (인물, 장소, 개념)
- **관계(R)** 추출 (엔티티 간 연결)

#### Stage 2: Graph Augmentation (그래프 증강)

그래프를 구조적으로 강화한다:

```
초기 그래프 → K-core 분해 → 커뮤니티 탐지 → Attribute(A) + High-level(H) + Overview(O) 생성
```

- **K-core 분해** + **Betweenness Centrality**로 구조적으로 중요한 엔티티 식별
- LLM 요약으로 **Attribute 노드(A)** 생성
- 커뮤니티 탐지 후 **High-level Element(H)** 와 **Overview(O)** 생성

#### Stage 3: Graph Enrichment (그래프 풍부화)

검색 성능을 위한 최종 강화:

```
이질적 그래프 → 텍스트 청크(T) 임베딩 → HNSW 유사도 엣지 추가 → 최종 그래프
```

- **Text Chunk(T)** 를 임베딩하여 전체 컨텍스트 보존
- **HNSW** 알고리즘으로 의미 유사도 기반 엣지 추가
- 그래프 연결성 강화

---

## 4. 검색 메커니즘: Dual Search & Shallow PPR

### Dual Search (이중 검색)

두 가지 검색을 **동시에** 수행하여 엔트리 포인트를 찾는다:

| 검색 방식 | 대상 노드 | 방법 | 목적 |
|-----------|----------|------|------|
| **Exact Match** | Entity(N), Overview(O) | 키워드 정확 일치 | 정확한 엔티티 매칭 |
| **Semantic Search** | Semantic Unit(S), Attribute(A), High-level(H) | 벡터 유사도 | 의미적 관련성 포착 |

```
Query → ┌─ Exact Match → N/O 노드 탐색
        └─ Semantic Search → S/A/H 노드 탐색
                          ↓
              Entry Points (시작 노드들)
```

### Shallow PPR (얕은 Personalized PageRank)

Dual Search로 찾은 **엔트리 포인트**에서 시작하여, **제한된 깊이**의 Random Walk을 수행한다:

```
Entry Points → Shallow PPR (제한적 반복) → Cross Nodes 식별 → 최종 context 구성
```

- **"Shallow"**: 적은 반복 횟수로 **다중 홉 추론 경로**를 효율적으로 탐색
- **목적**: 쿼리와 관련된 **Cross Nodes** (모든 엔트리 포인트에 근접한 노드) 식별
- **장점**: 과도한 노이즈 없이 관련 정보만 수집

### 전체 검색 플로우

```
Query
  ↓
Dual Search (Exact + Semantic)
  ↓
Entry Points 식별
  ↓
Shallow PPR (localized random walk)
  ↓
Cross Nodes 선택 (관련 서브그래프)
  ↓
Context 구성 → LLM 응답 생성
```

---

## 5. GraphRAG / LightRAG와의 비교

### 아키텍처 비교

| 특성 | GraphRAG | LightRAG | NodeRAG |
|------|----------|----------|---------|
| **그래프 구조** | 동질 (엔티티+관계) | 동질 (엔티티+관계+키워드) | **이질적 (7가지 노드)** |
| **인덱싱** | 전체 그래프 재구축 필요 | 증분 업데이트 지원 | 3단계 정교한 파이프라인 |
| **검색 방식** | 글로벌/로컬 검색 | 키워드 + 벡터 검색 | **Dual Search + Shallow PPR** |
| **커뮤니티 활용** | 커뮤니티 요약 중심 | 경량 커뮤니티 | K-core + 커뮤니티 요약 |
| **설계 철학** | 포괄적 (비용 높음) | 경량화 (비용 절감) | **구조 최적화 (정밀+효율)** |

### 벤치마크 성능 비교

#### Multi-hop QA (정확도 %)

| 데이터셋 | GraphRAG | LightRAG | NodeRAG | 우승 |
|---------|----------|----------|---------|------|
| **HotpotQA** | 89.00 | - | **89.50** | NodeRAG |
| **MuSiQue** | 41.71 | 36.00 | **46.29** | NodeRAG |
| **MultiHop-RAG** | - | - | - | (미보고) |

#### RAG-QA Arena (검색 비율 %)

| 도메인 | GraphRAG | LightRAG | NodeRAG |
|-------|----------|----------|---------|
| **Lifestyle** | 86.3 | 81.7 | **94.9** |

### 시스템 효율성 (HotpotQA 기준)

| 메트릭 | GraphRAG | LightRAG | NodeRAG | NodeRAG 이점 |
|--------|----------|----------|---------|-------------|
| **쿼리 시간** | 26.69s (global) | 5.58s | **3.98s** | 6.7x 빠름 (vs GraphRAG) |
| **검색 토큰** | 6.6k | - | **5.0k** | 24% 절감 |
| **저장 용량** | 227MB | 461MB | **214MB** | 최소 |

---

## 6. 현재 rag_bench 구현과의 비교

### 접근 방식의 근본적 차이

| 구분 | 현재 rag_bench (Dense+BM25 Hybrid) | NodeRAG (Graph-based) |
|------|--------------------------------|----------------------|
| **지식 표현** | 벡터 공간 (연속 임베딩) | 그래프 구조 (이산 노드+엣지) |
| **검색 단위** | 텍스트 청크 | 이질적 노드 (7가지 타입) |
| **다중 홉 추론** | ❌ 단일 홉 (1회 검색) | ✅ PPR로 다중 홉 경로 탐색 |
| **글로벌 컨텍스트** | ❌ 청크 수준 제한 | ✅ High-level Element로 글로벌 요약 |
| **인덱싱 비용** | 낮음 (임베딩만) | 높음 (LLM 호출 필수) |
| **유지보수** | 쉬움 (문서 추가/삭제) | 복잡 (그래프 재구축) |

### 현재 파이프라인이 더 적합한 경우

- **단일 홉 질문** (예: "A란 무엇인가?")
- **문서 수가 자주 변경**되는 환경
- **빠른 인덱싱**이 필요한 경우
- **LLM API 비용**에 제약이 있는 경우

### NodeRAG가 더 적합한 경우

- **다중 홉 추론** 필요 (예: "A가 B에 미친 영향 중 C와 관련된 것은?")
- **복잡한 관계 추론** 필요
- **글로벌 요약/개요** 질문이 많은 경우
- **정적 코퍼스**로 운영 (인덱싱 1회)

---

## 7. 실적용 시 아키텍처 설계

### 방안 A: NodeRAG 독립 파이프라인

```
문서 → NodeRAG Indexing (3-stage) → Heterograph
쿼리 → Dual Search + PPR → Context → LLM → 응답
```

- **적합**: 새로운 프로젝트, 다중 홉 QA가 핵심인 경우
- **비용**: 인덱싱 시 LLM API 호출 필요

### 방안 B: 현재 시스템 + NodeRAG 하이브리드

```
쿼리 → [라우터: 질문 유형 분류]
        ├─ 단순 질문 → 기존 Dense+BM25 파이프라인
        └─ 복잡 질문 → NodeRAG 그래프 검색
                    ↓
              Context 결합 → LLM → 응답
```

- **적합**: 기존 시스템 유지하면서 복잡 질문 대응력 강화
- **구현**: 질문 라우터 + 이중 파이프라인

### 방안 C: NodeRAG의 아이디어만 차용

기존 rag_bench 파이프라인에 NodeRAG 개념을 **부분 적용**:

- **커뮤니티 요약 (H)**: 코퍼스의 글로벌 요약을 별도 인덱싱
- **엔티티 그래프 (N+R)**: 주요 엔티티 관계 그래프 구축 → 검색 보강
- **Dual Search**: 키워드 매칭 + 벡터 검색 병행 (HybridRRF와 유사)

---

## 8. 한계점 및 향후 과제

### 현재 한계점

| # | 한계점 | 상세 |
|---|--------|------|
| 1 | **인덱싱 비용** | LLM API 호출 필수, 대용량 코퍼스에서 비용 증가 |
| 2 | **한국어 검증 부재** | 영어 위주 벤치마크, 한국어 NER/관계 추출 성능 미검증 |
| 3 | **증분 업데이트** | 그래프 구조 변경 시 전체 재구축 가능성 |
| 4 | **초기 설정 복잡도** | 7가지 노드 타입의 설계 파라미터 튜닝 필요 |
| 5 | **커뮤니티 성숙도** | v0.1.0 (2025.03), 프로덕션 레벨 미검증 |
| 6 | **rag_bench 미통합** | Strategy 모듈로 구현 예정 |

### 향후 과제

- [ ] 한국어 코퍼스에서 NodeRAG 그래프 품질 검증
- [ ] 인덱싱 비용 분석 (OpenAI vs Ollama 로컬 LLM)
- [ ] 현재 QA 데이터셋으로 다중 홉 질문 분류
- [ ] 하이브리드 아키텍처(방안 B) PoC 설계
- [ ] NodeRAG의 Graph Decomposition 단계 한국어 최적화

---

## 9. 참고 자료

| 리소스 | URL |
|--------|-----|
| NodeRAG 논문 | https://arxiv.org/abs/2504.11544 |
| NodeRAG GitHub | https://github.com/Terry-Xu-666/NodeRAG |
| NodeRAG 공식 사이트 | https://terry-xu-666.github.io/NodeRAG/ |
| GraphRAG (Microsoft) | https://github.com/microsoft/graphrag |
| LightRAG | https://github.com/HKUDS/LightRAG |
| GraphRAG 논문 | https://arxiv.org/abs/2404.16130 |
| LightRAG 논문 | https://arxiv.org/abs/2410.05779 |
| NodeRAG 비교 분석 (dev.to) | https://dev.to/noderag-analysis |
