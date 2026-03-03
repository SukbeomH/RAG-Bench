<!-- K8s 벤치마크 결과 보고서 — generate_k8s_report.py 자동 생성 -->
<!-- 생성일: 2026년 03월 03일 -->

# RAG ColBERT Rerank 벤치마크 결과 보고서
### Dense × Sparse 조합별 성능 비교 — ColBERT Rerank + Contextual Retrieval 고정

> **작성일**: 2026년 03월 03일  
> **평가 범위**: 1개 카테고리 / 10개 AI 조합 / 총 500개 질의응답 테스트  
> **파이프라인**: Dense 검색 + Sparse 검색 + ColBERT 리랭커 + Contextual 문맥 강화  
> **실행 환경**: EKS K8s 클러스터 (management 노드, CPU-only)  

---

## Executive Summary

> 이 보고서는 RAG 검색 파이프라인에서 ColBERT Rerank를 적용한 상태에서
> **어떤 Dense + Sparse 조합이 최적인지**를 K8s 클러스터 실험으로 확인한 결과입니다.

### 핵심 결론

#### GENERAL 카테고리

**종합 1위: OpenAI (API) + BM25 + ColBERT** (복합 점수 0.8501)

| 순위 | 조합 | 복합 점수 | 핵심 강점 |
|:----:|:-----|:--------:|:---------|
| **1** | **OpenAI (API) + BM25 + ColBERT** | **0.8501** | Recall 최고 |
| 2 | E5-multilingual + SPLADE + ColBERT | 0.8480 | Faithfulness 최고 |
| 3 | Upstage Solar (API) + BM25 + ColBERT | 0.8395 | 균형 성능 |

### 즉시 실행 항목

- [ ] **GENERAL**: 서비스 파이프라인에 **OpenAI (API) + BM25 + ColBERT** 조합 통합
- [ ] API 모델 사용 시 보안(데이터 외부 전송) 및 비용(건당 과금) 검토

---

## 1. 무엇을, 왜 테스트했는가

### 1-1. 비즈니스 질문

> **"ColBERT Rerank를 적용한 상태에서, 어떤 Dense + Sparse 조합이**  
> **가장 정확한 검색 결과를 제공하는가?"**

### 1-2. 테스트 설계

고객이 질문하면 RAG 시스템은 다음 4단계를 거칩니다:

```
질문 → [Dense 검색] + [Sparse 검색] → [ColBERT 리랭킹] → [LLM 답변 생성]
         ↑ 변수         ↑ 변수           ↑ 고정              ↑ 고정
```

| 방식 | 작동 원리 | 비유 | 강점 |
|------|----------|------|------|
| **Dense 검색** (의미 기반) | 질문의 '뜻'을 이해해서 유사 내용을 찾음 | 사서가 내용을 읽고 추천 | 유사 표현·동의어에 강함 |
| **Sparse 검색** (키워드 기반) | 단어 그대로 매칭하여 찾음 | 색인에서 단어 검색 | 정확한 용어·번호에 강함 |

이번 평가에서는 Dense 5종 × Sparse 2종 = **10개 조합**을 비교했습니다.

### 1-3. 고정 파이프라인 — 변수에서 제외한 항목

| 고정 요소 | 적용 이유 |
|----------|---------|
| **ColBERT 리랭커** (jina-colbert-v2) | 최종 답변 후보를 토큰 수준으로 재정렬. 오답률 25% 감소 확인 (IBM). |
| **Contextual 문맥 강화** | 검색 전 청크에 문맥을 AI로 추가. 검색 실패율 67% 감소 (Anthropic). |

### 1-4. 비교 대상 모델

| 구분 | 모델명 | 파라미터 | 특성 | 유형 |
|------|-------|---------|------|:----:|
| Dense | **Upstage Solar (API)** | — | Upstage Solar Embedding, 4096차원 | API |
| Dense | **OpenAI (API)** | — | text-embedding-3-large, 3072차원 | API |
| Dense | **BGE-M3** | 570M | 100+ 언어, MIRACL 한국어 SOTA | 로컬(HF) |
| Dense | **E5-multilingual** | 560M | 다국어 E5, 명령어 prefix 방식 | 로컬(HF) |
| Dense | **KoSimCSE** | 110M | 한국어 SimCSE 대조 학습 | 로컬(HF) |
| Sparse | **BM25** (OKt) | — | 한국어 형태소 기반 키워드 매칭 | — |
| Sparse | **SPLADE** | 110M | 학습된 확장 토큰으로 동의어 포착 | — |

> **API 모델 참고**: API 모델(OpenAI, Upstage)은 로컬 모델 대비 품질을 비교하기 위해 포함했습니다.
> 실서비스에서는 보안(데이터 외부 전송) 및 비용(건당 과금) 관점에서 별도 검토가 필요합니다.

---

## 2. 어떻게 측정했는가

### 2-1. 테스트 데이터셋

| 카테고리 | 데이터 출처 | 쿼리 수 | 특성 |
|---------|-----------|:------:|------|
| GENERAL | MIRACL(ko) + Ko-StrategyQA + Belebele + MrTiDy | 50 | 위키피디아 기반 범용 질의응답 |

### 2-2. 평가 지표 — 4가지

AI 답변 품질을 측정하는 4가지 관점을 가중 평균하여 **종합 점수**를 계산합니다:

| 지표 | 측정 내용 | 쉽게 말하면 | 가중치 |
|------|----------|-----------|:------:|
| **Context Recall** | 정답 내용을 빠뜨리지 않았는가 | '놓치지 않는 능력' | **35%** |
| **Context Precision** | 찾아온 문서가 질문과 관련 있는가 | '쓸모없는 내용 배제 능력' | 30% |
| **Faithfulness** | 답변이 문서 내용에 근거하는가 | '지어내지 않는 능력' | 20% |
| **Answer Relevancy** | 질문에 직접적으로 답하는가 | '동문서답 방지 능력' | 15% |

> **종합 점수** = Recall×0.35 + Precision×0.30 + Faithfulness×0.20 + Relevancy×0.15
> 0에서 1 사이 값이며, 1에 가까울수록 우수합니다.

---

## 3. 종합 성능 결과

### 3-1. GENERAL 카테고리

#### 종합 순위표

> 점수는 0~1 범위 (높을수록 우수). 1위 대비 차이(%p)를 병기합니다.

| 순위 | AI 조합 | 종합 점수 | 1위 대비 | Recall | Precision | Faithfulness | Relevancy |
|:----:|:--------|:--------:|:-------:|:------:|:---------:|:------------:|:---------:|
| 🥇 **1위** | OpenAI (API) + BM25 + ColBERT | **0.8501** | 기준 (1위) | 0.8472 | 0.9183 | 0.7433 | 0.8631 |
| 🥈 2위 | E5-multilingual + SPLADE + ColBERT | 0.8480 | −0.2% | 0.8126 | 0.9167 | 0.7983 | 0.8596 |
| 🥉 3위 | Upstage Solar (API) + BM25 + ColBERT | 0.8395 | −1.2% | 0.8351 | 0.8950 | 0.7433 | 0.8667 |
| 4위 | BGE-M3 + BM25 + ColBERT | 0.8378 | −1.4% | 0.8388 | 0.9117 | 0.7180 | 0.8477 |
| 5위 | OpenAI (API) + SPLADE + ColBERT | 0.8369 | −1.6% | 0.8061 | 0.9067 | 0.7687 | 0.8598 |
| 6위 | KoSimCSE + BM25 + ColBERT | 0.8357 | −1.7% | 0.8034 | 0.9233 | 0.7264 | 0.8815 |
| 7위 | E5-multilingual + BM25 + ColBERT | 0.8336 | −1.9% | 0.8249 | 0.9083 | 0.7113 | 0.8675 |
| 8위 | Upstage Solar (API) + SPLADE + ColBERT | 0.8329 | −2.0% | 0.8142 | 0.9050 | 0.7350 | 0.8629 |
| 9위 | BGE-M3 + SPLADE + ColBERT | 0.8302 | −2.3% | 0.7947 | 0.9133 | 0.7300 | 0.8805 |
| 10위 | KoSimCSE + SPLADE + ColBERT | 0.8260 | −2.8% | 0.7996 | 0.9183 | 0.7003 | 0.8709 |

#### 핵심 인사이트

**1. OpenAI (API) + BM25 + ColBERT가 종합 1위인 이유**
> Context Recall 지표에서 전 조합 중 최고점을 기록했습니다.
> 2위(E5-multilingual + SPLADE + ColBERT) 대비 0.2% 우세합니다.

**2. 1위와 2위 성능 차이가 미미**
> 1위와 2위의 종합 점수 차이가 0.0021 (0.2%)로 통계적으로 유의미하지 않을 수 있습니다.
> 두 조합 모두 실서비스 적용이 가능하며, **운영 비용·보안 요건**(API vs 로컬)에 따라 선택하는 것이 합리적입니다.

**3. API 모델이 품질 우위**
> API 최고(OpenAI (API) + BM25 + ColBERT, 0.8501)가 로컬 최고(E5-multilingual + SPLADE + ColBERT, 0.8480)보다 0.0021 우세합니다.
> 다만 API 모델은 데이터 외부 전송 및 건당 과금이 발생하므로 보안·비용 검토가 필요합니다.

---

## 4. 지표별 상세 분석

### 4-1. GENERAL 카테고리

#### Faithfulness — 지어내지 않는 능력

> 답변이 검색된 문서 내용에 근거하는지 측정합니다. 할루시네이션(거짓 정보 생성) 방지의 핵심 지표입니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | E5-multilingual + SPLADE + ColBERT | 0.7983 | **최고** |
| 2 | OpenAI (API) + SPLADE + ColBERT | 0.7687 |  |
| 3 | OpenAI (API) + BM25 + ColBERT | 0.7433 |  |
| 4 | Upstage Solar (API) + BM25 + ColBERT | 0.7433 |  |
| 5 | Upstage Solar (API) + SPLADE + ColBERT | 0.7350 |  |
| 6 | BGE-M3 + SPLADE + ColBERT | 0.7300 |  |
| 7 | KoSimCSE + BM25 + ColBERT | 0.7264 |  |
| 8 | BGE-M3 + BM25 + ColBERT | 0.7180 |  |
| 9 | E5-multilingual + BM25 + ColBERT | 0.7113 |  |
| 10 | KoSimCSE + SPLADE + ColBERT | 0.7003 | 최고 대비 −0.0980 |

> 전체 편차: 0.0980 (12.3%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Recall — 놓치지 않는 능력

> 정답에 필요한 문서를 빠짐없이 검색하는지 측정합니다. 정보 누락은 서비스 신뢰도를 직접 훼손합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | OpenAI (API) + BM25 + ColBERT | 0.8472 | **최고** |
| 2 | BGE-M3 + BM25 + ColBERT | 0.8388 |  |
| 3 | Upstage Solar (API) + BM25 + ColBERT | 0.8351 |  |
| 4 | E5-multilingual + BM25 + ColBERT | 0.8249 |  |
| 5 | Upstage Solar (API) + SPLADE + ColBERT | 0.8142 |  |
| 6 | E5-multilingual + SPLADE + ColBERT | 0.8126 |  |
| 7 | OpenAI (API) + SPLADE + ColBERT | 0.8061 |  |
| 8 | KoSimCSE + BM25 + ColBERT | 0.8034 |  |
| 9 | KoSimCSE + SPLADE + ColBERT | 0.7996 |  |
| 10 | BGE-M3 + SPLADE + ColBERT | 0.7947 | 최고 대비 −0.0525 |

> 전체 편차: 0.0525 (6.2%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Precision — 정확하게 찾는 능력

> 검색 결과 중 실제 관련 문서의 비율을 측정합니다. 불필요한 문서가 많으면 LLM 답변 품질이 저하됩니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | KoSimCSE + BM25 + ColBERT | 0.9233 | **최고** |
| 2 | OpenAI (API) + BM25 + ColBERT | 0.9183 |  |
| 3 | KoSimCSE + SPLADE + ColBERT | 0.9183 |  |
| 4 | E5-multilingual + SPLADE + ColBERT | 0.9167 |  |
| 5 | BGE-M3 + SPLADE + ColBERT | 0.9133 |  |
| 6 | BGE-M3 + BM25 + ColBERT | 0.9117 |  |
| 7 | E5-multilingual + BM25 + ColBERT | 0.9083 |  |
| 8 | OpenAI (API) + SPLADE + ColBERT | 0.9067 |  |
| 9 | Upstage Solar (API) + SPLADE + ColBERT | 0.9050 |  |
| 10 | Upstage Solar (API) + BM25 + ColBERT | 0.8950 | 최고 대비 −0.0283 |

> 전체 편차: 0.0283 (3.1%) — 조합 간 차이가 작아 이 지표만으로는 우열을 가리기 어렵습니다.

#### Answer Relevancy — 질문에 답하는 능력

> 생성된 답변이 질문에 직접적으로 대응하는지 측정합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | KoSimCSE + BM25 + ColBERT | 0.8815 | **최고** |
| 2 | BGE-M3 + SPLADE + ColBERT | 0.8805 |  |
| 3 | KoSimCSE + SPLADE + ColBERT | 0.8709 |  |
| 4 | E5-multilingual + BM25 + ColBERT | 0.8675 |  |
| 5 | Upstage Solar (API) + BM25 + ColBERT | 0.8667 |  |
| 6 | OpenAI (API) + BM25 + ColBERT | 0.8631 |  |
| 7 | Upstage Solar (API) + SPLADE + ColBERT | 0.8629 |  |
| 8 | OpenAI (API) + SPLADE + ColBERT | 0.8598 |  |
| 9 | E5-multilingual + SPLADE + ColBERT | 0.8596 |  |
| 10 | BGE-M3 + BM25 + ColBERT | 0.8477 | 최고 대비 −0.0338 |

> 전체 편차: 0.0338 (3.8%) — 조합 간 차이가 작아 이 지표만으로는 우열을 가리기 어렵습니다.

---

## 5. 레이턴시(속도) 참고

> **레이턴시는 순위 결정에 반영하지 않습니다.**
> LLM 추론 노이즈가 전략 간 차이를 압도하며, 동일 전략도 실행 시점에 따라 편차가 큽니다.
> CPU-only 환경 수치이므로 GPU 환경과 직접 비교할 수 없습니다.
> 아래는 실행 환경에서의 기준선 참고 데이터입니다.

### GENERAL

| 조합 | 평균 (s/query) | 중앙값 (s/query) |
|:-----|:--------------:|:---------------:|
| E5-multilingual + SPLADE + ColBERT | 32 | 32 |
| BGE-M3 + BM25 + ColBERT | 33 | 32 |
| BGE-M3 + SPLADE + ColBERT | 115 | 141 |
| KoSimCSE + BM25 + ColBERT | 142 | 145 |
| E5-multilingual + BM25 + ColBERT | 142 | 151 |
| OpenAI (API) + BM25 + ColBERT | 143 | 144 |
| Upstage Solar (API) + BM25 + ColBERT | 144 | 145 |
| OpenAI (API) + SPLADE + ColBERT | 151 | 151 |
| Upstage Solar (API) + SPLADE + ColBERT | 151 | 153 |
| KoSimCSE + SPLADE + ColBERT | 151 | 152 |

---

## 6. 모델 유형별 비교

### 6-1. GENERAL 카테고리

#### Dense 모델 비교 (Sparse 평균)

> 각 Dense 모델의 BM25/SPLADE 결과를 평균하여 순수 Dense 모델 성능을 비교합니다.

| Dense 모델 | 유형 | 평균 복합 점수 | 평균 Recall | 평균 Faithfulness |
|-----------|:----:|:------------:|:----------:|:----------------:|
| OpenAI (API) | API | 0.8435 | 0.8266 | 0.7560 |
| E5-multilingual | 로컬 | 0.8408 | 0.8187 | 0.7548 |
| Upstage Solar (API) | API | 0.8362 | 0.8246 | 0.7391 |
| BGE-M3 | 로컬 | 0.8340 | 0.8167 | 0.7240 |
| KoSimCSE | 로컬 | 0.8308 | 0.8015 | 0.7134 |

> **OpenAI (API)**가 평균 복합 점수 0.8435로 Dense 모델 중 1위.
> 최하위(KoSimCSE) 대비 0.0126 (1.5%) 우세.

#### Sparse 모델 비교 (Dense 평균)

> 각 Sparse 모델의 Dense 5종 결과를 평균하여 순수 Sparse 모델 효과를 비교합니다.

| Sparse 모델 | 평균 복합 점수 | 평균 Recall | 평균 Precision |
|------------|:------------:|:----------:|:-------------:|
| BM25 | 0.8393 | 0.8299 | 0.9113 |
| SPLADE | 0.8348 | 0.8054 | 0.9120 |

> BM25와 SPLADE의 평균 복합 점수 차이가 0.0045로 미미합니다.
> 특히 BM25가 Recall에서 +0.0244 우세 — 한국어 형태소 분석(OKt) 기반의 정확한 키워드 매칭이 Recall에 기여합니다.

---

## 7. 최종 모델 선정 가이드

### 7-1. GENERAL 카테고리

#### 용도별 추천

| 사용 상황 | 추천 조합 | 이유 |
|----------|---------|------|
| **품질 최우선** | OpenAI (API) + BM25 + ColBERT | 종합 점수 1위 (0.8501) |
| **할루시네이션 방지** | E5-multilingual + SPLADE + ColBERT | Faithfulness 최고 (0.7983) |
| **보안·비용 우선 (로컬)** | E5-multilingual + SPLADE + ColBERT | 로컬 모델 최고 (0.8480), 데이터 외부 전송 없음 |

#### 한 가지만 선택해야 한다면

> **E5-multilingual + SPLADE + ColBERT**
>
> 종합 점수 0.8480로 1위(OpenAI (API) + BM25 + ColBERT, 0.8501)와 0.0021 (0.2%) 차이로 동등 수준이며, 로컬 실행으로 데이터 외부 전송 없이 보안을 확보할 수 있습니다.

### 7-2. 향후 과제

- [ ] **추가 카테고리 벤치마크**: BUSINESS, LEGAL, MEDICAL, TECHNICAL 카테고리에서 동일 조합 검증
- [ ] **FlashRank 리랭커 비교**: ColBERT 대비 경량 리랭커의 품질-속도 트레이드오프 확인
- [ ] **분기별 재평가**: Dense 모델 신규 릴리스에 맞춘 정기 벤치마크 실행

---

---

## 부록: 기술 세부사항

> 이 섹션은 개발팀·데이터팀을 위한 기술 참고 정보입니다.

### A. 평가 지표 가중치 산정 근거

| 지표 | 가중치 | 산정 근거 |
|------|:------:|---------|
| Context Recall | 0.35 | 서비스에서 정보 누락은 오답보다 사용자 이탈로 직결. RAGAS 권고 가중치. |
| Context Precision | 0.30 | 불필요 컨텍스트 포함 시 LLM 답변 품질 저하 및 토큰 비용 증가. |
| Faithfulness | 0.20 | 할루시네이션 방지 — 법률·의료 도메인에서 특히 중요. |
| Answer Relevancy | 0.15 | 위 3개 지표가 높으면 자연스럽게 따라오는 경향. |

> **복합 점수** = Recall×0.35 + Precision×0.30 + Faithfulness×0.20 + Relevancy×0.15

### B. 파이프라인 구성 요소

| 요소 | 설정 | 적용 근거 |
|------|------|---------|
| ColBERT Rerank | jina-colbert-v2 | 토큰 수준 재정렬, 오답률 25% 감소 (IBM 연구) |
| FlashRank Rerank | FlashRank v2 | 경량 교차 인코더 리랭킹, 속도 대비 품질 균형 |
| Contextual Retrieval | LLM 문맥 강화 | 검색 실패율 67% 감소 (Anthropic 보고) |
| Chunking | Parent-Child (512/128 tokens) | 문맥 보존 + 세밀 검색 동시 확보 |

### C. 실행 환경

| 항목 | 값 |
|------|-----|
| 클러스터 | EKS zcp-ags-cp-eks (ap-northeast-2) |
| 노드 | management (m7i/m8i.2xlarge, 8C/32G) |
| GPU | 없음 (CPU-only) |
| Namespace | rag-bench-test |
| PVC | bench-results (EFS RWX) |
| 실행 시간 | 약 137~168분/Job (ColBERT CPU rerank 병목) |

### D. 전체 원시 데이터

**GENERAL** (질의 50개, 10개 조합)

| Rank | Strategy | Recall | Precision | Faithfulness | Relevancy | Composite | Avg Lat (s) | Med Lat (s) |
|:----:|:---------|:------:|:---------:|:------------:|:---------:|:---------:|:-----------:|:-----------:|
| 1 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.8472 | 0.9183 | 0.7433 | 0.8631 | **0.8501** | 143.0 | 144.0 |
| 2 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.8126 | 0.9167 | 0.7983 | 0.8596 | 0.8480 | 32.2 | 32.2 |
| 3 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.8351 | 0.8950 | 0.7433 | 0.8667 | 0.8395 | 143.5 | 145.2 |
| 4 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.8388 | 0.9117 | 0.7180 | 0.8477 | 0.8378 | 32.6 | 32.2 |
| 5 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.8061 | 0.9067 | 0.7687 | 0.8598 | 0.8369 | 150.7 | 151.2 |
| 6 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.8034 | 0.9233 | 0.7264 | 0.8815 | 0.8357 | 141.6 | 145.3 |
| 7 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.8249 | 0.9083 | 0.7113 | 0.8675 | 0.8336 | 142.2 | 150.9 |
| 8 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.8142 | 0.9050 | 0.7350 | 0.8629 | 0.8329 | 151.1 | 152.6 |
| 9 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.7947 | 0.9133 | 0.7300 | 0.8805 | 0.8302 | 115.2 | 140.9 |
| 10 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.7996 | 0.9183 | 0.7003 | 0.8709 | 0.8260 | 151.2 | 152.0 |

### E. 재현 방법

```bash
# K8s 벤치마크 실행
uv run python -m k8s.orchestrator \
    --categories general \
    --rerankers colbert \
    --preset service

# 결과 보고서 생성
uv run python k8s_results/generate_k8s_report.py \
    --run_dir k8s_results/20260226-0948
```
