<!-- K8s 벤치마크 결과 보고서 — generate_k8s_report.py 자동 생성 -->
<!-- 생성일: 2026년 03월 03일 -->

# RAG 리랭커 비교 벤치마크 결과 보고서
### Dense × Sparse × Reranker 조합별 성능 비교 — ColBERT + FlashRank 리랭커 비교

> **작성일**: 2026년 03월 03일  
> **평가 범위**: 5개 카테고리 / 20개 AI 조합 / 총 6,820개 질의응답 테스트  
> **파이프라인**: Dense 검색 + Sparse 검색 + 리랭커(ColBERT + FlashRank) + Contextual 문맥 강화  
> **실행 환경**: EKS K8s 클러스터 (management 노드, CPU-only)  

---

## Executive Summary

> 이 보고서는 RAG 검색 파이프라인에서 ColBERT / FlashRank 리랭커를 비교하여
> **어떤 Dense + Sparse + Reranker 조합이 최적인지**를 K8s 클러스터 실험으로 확인한 결과입니다.

### 핵심 결론

#### BUSINESS 카테고리

**종합 1위: KoSimCSE + BM25 + ColBERT** (복합 점수 0.8668)

| 순위 | 조합 | 복합 점수 | 핵심 강점 |
|:----:|:-----|:--------:|:---------|
| **1** | **KoSimCSE + BM25 + ColBERT** | **0.8668** | Faithfulness 최고 |
| 2 | OpenAI (API) + SPLADE + ColBERT | 0.8631 | 균형 성능 |
| 3 | BGE-M3 + SPLADE + ColBERT | 0.8630 | Recall 최고 |

#### GENERAL 카테고리

**종합 1위: BGE-M3 + BM25 + ColBERT** (복합 점수 0.8561)

| 순위 | 조합 | 복합 점수 | 핵심 강점 |
|:----:|:-----|:--------:|:---------|
| **1** | **BGE-M3 + BM25 + ColBERT** | **0.8561** | 균형 성능 |
| 2 | BGE-M3 + SPLADE + ColBERT | 0.8536 | Faithfulness 최고, Relevancy 최고 |
| 3 | OpenAI (API) + BM25 + ColBERT | 0.8500 | Recall 최고 |

#### LEGAL 카테고리

**종합 1위: E5-multilingual + SPLADE + ColBERT** (복합 점수 0.8565)

| 순위 | 조합 | 복합 점수 | 핵심 강점 |
|:----:|:-----|:--------:|:---------|
| **1** | **E5-multilingual + SPLADE + ColBERT** | **0.8565** | Recall 최고 |
| 2 | Upstage Solar (API) + SPLADE + ColBERT | 0.8559 | Faithfulness 최고, Precision 최고 |
| 3 | BGE-M3 + SPLADE + ColBERT | 0.8553 | Precision 최고 |

#### MEDICAL 카테고리

**종합 1위: E5-multilingual + BM25 + ColBERT** (복합 점수 0.9166)

| 순위 | 조합 | 복합 점수 | 핵심 강점 |
|:----:|:-----|:--------:|:---------|
| **1** | **E5-multilingual + BM25 + ColBERT** | **0.9166** | 균형 성능 |
| 2 | BGE-M3 + BM25 + ColBERT | 0.9146 | Relevancy 최고 |
| 3 | OpenAI (API) + BM25 + ColBERT | 0.9146 | Precision 최고 |

#### TECHNICAL 카테고리

**종합 1위: E5-multilingual + BM25 + ColBERT** (복합 점수 0.5229)

| 순위 | 조합 | 복합 점수 | 핵심 강점 |
|:----:|:-----|:--------:|:---------|
| **1** | **E5-multilingual + BM25 + ColBERT** | **0.5229** | Faithfulness 최고 |
| 2 | Upstage Solar (API) + BM25 + ColBERT | 0.5194 | Recall 최고 |
| 3 | OpenAI (API) + SPLADE + ColBERT | 0.5189 | Precision 최고 |

### 즉시 실행 항목

- [ ] **BUSINESS**: 서비스 파이프라인에 **KoSimCSE + BM25 + ColBERT** 조합 통합
- [ ] **GENERAL**: 서비스 파이프라인에 **BGE-M3 + BM25 + ColBERT** 조합 통합
- [ ] **LEGAL**: 서비스 파이프라인에 **E5-multilingual + SPLADE + ColBERT** 조합 통합
- [ ] **MEDICAL**: 서비스 파이프라인에 **E5-multilingual + BM25 + ColBERT** 조합 통합
- [ ] **TECHNICAL**: 서비스 파이프라인에 **E5-multilingual + BM25 + ColBERT** 조합 통합
- [ ] API 모델 사용 시 보안(데이터 외부 전송) 및 비용(건당 과금) 검토

---

## 1. 무엇을, 왜 테스트했는가

### 1-1. 비즈니스 질문

> **"어떤 Dense + Sparse + Reranker 조합이**  
> **가장 정확한 검색 결과를 제공하는가?"**

### 1-2. 테스트 설계

고객이 질문하면 RAG 시스템은 다음 4단계를 거칩니다:

```
질문 → [Dense 검색] + [Sparse 검색] → [리랭킹(ColBERT / FlashRank)] → [LLM 답변 생성]
         ↑ 변수         ↑ 변수           ↑ 변수                  ↑ 고정
```

| 방식 | 작동 원리 | 비유 | 강점 |
|------|----------|------|------|
| **Dense 검색** (의미 기반) | 질문의 '뜻'을 이해해서 유사 내용을 찾음 | 사서가 내용을 읽고 추천 | 유사 표현·동의어에 강함 |
| **Sparse 검색** (키워드 기반) | 단어 그대로 매칭하여 찾음 | 색인에서 단어 검색 | 정확한 용어·번호에 강함 |

이번 평가에서는 Dense 5종 × Sparse 2종 = **20개 조합**을 비교했습니다.

### 1-3. 고정 파이프라인 — 변수에서 제외한 항목

| 고정 요소 | 적용 이유 |
|----------|---------|
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
| BUSINESS | 비즈니스 QA 데이터셋 | 77 | 비즈니스 문서 질의응답 |
| GENERAL | MIRACL(ko) + Ko-StrategyQA + Belebele + MrTiDy | 100 | 위키피디아 기반 범용 질의응답 |
| LEGAL | 법률 QA 데이터셋 | 37 | 법률 문서 질의응답 |
| MEDICAL | 의료 QA 데이터셋 | 77 | 의료 문서 질의응답 |
| TECHNICAL | 기술 QA 데이터셋 | 50 | 기술 문서 질의응답 |

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

### 3-1. BUSINESS 카테고리

#### 종합 순위표

> 점수는 0~1 범위 (높을수록 우수). 1위 대비 차이(%p)를 병기합니다.

| 순위 | AI 조합 | 종합 점수 | 1위 대비 | Recall | Precision | Faithfulness | Relevancy |
|:----:|:--------|:--------:|:-------:|:------:|:---------:|:------------:|:---------:|
| 🥇 **1위** | KoSimCSE + BM25 + ColBERT | **0.8668** | 기준 (1위) | 0.7577 | 0.9946 | 0.8601 | 0.8746 |
| 🥈 2위 | OpenAI (API) + SPLADE + ColBERT | 0.8631 | −0.4% | 0.7629 | 0.9870 | 0.8531 | 0.8623 |
| 🥉 3위 | BGE-M3 + SPLADE + ColBERT | 0.8630 | −0.4% | 0.7636 | 0.9848 | 0.8410 | 0.8810 |
| 4위 | Upstage Solar (API) + SPLADE + ColBERT | 0.8627 | −0.5% | 0.7559 | 0.9870 | 0.8518 | 0.8776 |
| 5위 | E5-multilingual + SPLADE + ColBERT | 0.8613 | −0.6% | 0.7568 | 0.9848 | 0.8415 | 0.8845 |
| 6위 | OpenAI (API) + BM25 + ColBERT | 0.8595 | −0.8% | 0.7541 | 0.9870 | 0.8345 | 0.8835 |
| 7위 | BGE-M3 + BM25 + ColBERT | 0.8583 | −1.0% | 0.7571 | 0.9848 | 0.8308 | 0.8784 |
| 8위 | KoSimCSE + SPLADE + ColBERT | 0.8583 | −1.0% | 0.7392 | 0.9978 | 0.8411 | 0.8799 |
| 9위 | E5-multilingual + BM25 + ColBERT | 0.8573 | −1.1% | 0.7566 | 0.9848 | 0.8300 | 0.8736 |
| 10위 | Upstage Solar (API) + BM25 + ColBERT | 0.8535 | −1.5% | 0.7483 | 0.9870 | 0.8227 | 0.8730 |
| 11위 | E5-multilingual + BM25 + FlashRank | 0.7616 | −12.1% | 0.7135 | 0.9632 | 0.4582 | 0.8751 |
| 12위 | BGE-M3 + BM25 + FlashRank | 0.7575 | −12.6% | 0.6727 | 0.9740 | 0.4955 | 0.8719 |
| 13위 | OpenAI (API) + BM25 + FlashRank | 0.7453 | −14.0% | 0.6364 | 0.9740 | 0.4943 | 0.8766 |
| 14위 | Upstage Solar (API) + BM25 + FlashRank | 0.7443 | −14.1% | 0.6977 | 0.9762 | 0.3779 | 0.8775 |
| 15위 | KoSimCSE + BM25 + FlashRank | 0.7341 | −15.3% | 0.6419 | 0.9675 | 0.4536 | 0.8565 |
| 16위 | Upstage Solar (API) + SPLADE + FlashRank | 0.7021 | −19.0% | 0.6103 | 0.9665 | 0.3497 | 0.8574 |
| 17위 | E5-multilingual + SPLADE + FlashRank | 0.6968 | −19.6% | 0.6199 | 0.9232 | 0.3649 | 0.8660 |
| 18위 | BGE-M3 + SPLADE + FlashRank | 0.6964 | −19.7% | 0.6407 | 0.9361 | 0.3312 | 0.8336 |
| 19위 | OpenAI (API) + SPLADE + FlashRank | 0.6787 | −21.7% | 0.5789 | 0.9405 | 0.3295 | 0.8538 |
| 20위 | KoSimCSE + SPLADE + FlashRank | 0.6715 | −22.5% | 0.5707 | 0.9426 | 0.3245 | 0.8272 |

#### 핵심 인사이트

**1. KoSimCSE + BM25 + ColBERT가 종합 1위인 이유**
> Faithfulness 지표에서 전 조합 중 최고점을 기록했습니다.
> 2위(OpenAI (API) + SPLADE + ColBERT) 대비 0.4% 우세합니다.

**2. 1위와 2위 성능 차이가 미미**
> 1위와 2위의 종합 점수 차이가 0.0037 (0.4%)로 통계적으로 유의미하지 않을 수 있습니다.
> 두 조합 모두 실서비스 적용이 가능하며, **운영 비용·보안 요건**(API vs 로컬)에 따라 선택하는 것이 합리적입니다.

**3. 로컬 모델이 API 모델과 동등 이상의 성능**
> 로컬 최고(KoSimCSE + BM25 + ColBERT, 0.8668)가 API 최고(OpenAI (API) + SPLADE + ColBERT, 0.8631)보다 0.0037 우세합니다.
> API 모델이 반드시 우수하지 않으므로, 보안·비용 측면에서 로컬 모델이 유리합니다.

### 3-2. GENERAL 카테고리

#### 종합 순위표

> 점수는 0~1 범위 (높을수록 우수). 1위 대비 차이(%p)를 병기합니다.

| 순위 | AI 조합 | 종합 점수 | 1위 대비 | Recall | Precision | Faithfulness | Relevancy |
|:----:|:--------|:--------:|:-------:|:------:|:---------:|:------------:|:---------:|
| 🥇 **1위** | BGE-M3 + BM25 + ColBERT | **0.8561** | 기준 (1위) | 0.8552 | 0.9183 | 0.7557 | 0.8675 |
| 🥈 2위 | BGE-M3 + SPLADE + ColBERT | 0.8536 | −0.3% | 0.8299 | 0.9067 | 0.7922 | 0.8846 |
| 🥉 3위 | OpenAI (API) + BM25 + ColBERT | 0.8500 | −0.7% | 0.8572 | 0.9233 | 0.7067 | 0.8774 |
| 4위 | E5-multilingual + BM25 + ColBERT | 0.8489 | −0.8% | 0.8307 | 0.9233 | 0.7548 | 0.8681 |
| 5위 | Upstage Solar (API) + BM25 + ColBERT | 0.8479 | −1.0% | 0.8251 | 0.9258 | 0.7578 | 0.8652 |
| 6위 | Upstage Solar (API) + SPLADE + ColBERT | 0.8476 | −1.0% | 0.8246 | 0.9150 | 0.7725 | 0.8666 |
| 7위 | E5-multilingual + SPLADE + ColBERT | 0.8476 | −1.0% | 0.8284 | 0.9133 | 0.7672 | 0.8683 |
| 8위 | KoSimCSE + SPLADE + ColBERT | 0.8433 | −1.5% | 0.8144 | 0.9142 | 0.7685 | 0.8690 |
| 9위 | KoSimCSE + BM25 + ColBERT | 0.8432 | −1.5% | 0.8180 | 0.9275 | 0.7432 | 0.8665 |
| 10위 | OpenAI (API) + SPLADE + ColBERT | 0.8409 | −1.8% | 0.8228 | 0.9092 | 0.7522 | 0.8645 |
| 11위 | Upstage Solar (API) + SPLADE + FlashRank | 0.6010 | −29.8% | 0.5662 | 0.8333 | 0.2628 | 0.6685 |
| 12위 | KoSimCSE + SPLADE + FlashRank | 0.5979 | −30.2% | 0.5897 | 0.8000 | 0.2615 | 0.6615 |
| 13위 | OpenAI (API) + BM25 + FlashRank | 0.5963 | −30.3% | 0.5031 | 0.8258 | 0.3000 | 0.7499 |
| 14위 | E5-multilingual + SPLADE + FlashRank | 0.5908 | −31.0% | 0.5772 | 0.7850 | 0.2360 | 0.7074 |
| 15위 | BGE-M3 + SPLADE + FlashRank | 0.5825 | −32.0% | 0.5643 | 0.7825 | 0.2600 | 0.6549 |
| 16위 | KoSimCSE + BM25 + FlashRank | 0.5819 | −32.0% | 0.4969 | 0.8292 | 0.2646 | 0.7089 |
| 17위 | OpenAI (API) + SPLADE + FlashRank | 0.5818 | −32.0% | 0.5745 | 0.7942 | 0.2107 | 0.6689 |
| 18위 | Upstage Solar (API) + BM25 + FlashRank | 0.5776 | −32.5% | 0.4670 | 0.8333 | 0.2843 | 0.7152 |
| 19위 | BGE-M3 + BM25 + FlashRank | 0.5715 | −33.2% | 0.4622 | 0.8242 | 0.3115 | 0.6679 |
| 20위 | E5-multilingual + BM25 + FlashRank | 0.5701 | −33.4% | 0.4839 | 0.8167 | 0.2569 | 0.6955 |

#### 핵심 인사이트

**1. BGE-M3 + BM25 + ColBERT가 종합 1위인 이유**
> 모든 지표에서 균형 있게 높은 성능을 보여 가중 합산에서 최고 점수를 달성했습니다.
> 2위(BGE-M3 + SPLADE + ColBERT) 대비 0.3% 우세합니다.

**2. 1위와 2위 성능 차이가 미미**
> 1위와 2위의 종합 점수 차이가 0.0025 (0.3%)로 통계적으로 유의미하지 않을 수 있습니다.
> 두 조합 모두 실서비스 적용이 가능하며, **운영 비용·보안 요건**(API vs 로컬)에 따라 선택하는 것이 합리적입니다.

**3. 로컬 모델이 API 모델과 동등 이상의 성능**
> 로컬 최고(BGE-M3 + BM25 + ColBERT, 0.8561)가 API 최고(OpenAI (API) + BM25 + ColBERT, 0.8500)보다 0.0061 우세합니다.
> API 모델이 반드시 우수하지 않으므로, 보안·비용 측면에서 로컬 모델이 유리합니다.

### 3-3. LEGAL 카테고리

#### 종합 순위표

> 점수는 0~1 범위 (높을수록 우수). 1위 대비 차이(%p)를 병기합니다.

| 순위 | AI 조합 | 종합 점수 | 1위 대비 | Recall | Precision | Faithfulness | Relevancy |
|:----:|:--------|:--------:|:-------:|:------:|:---------:|:------------:|:---------:|
| 🥇 **1위** | E5-multilingual + SPLADE + ColBERT | **0.8565** | 기준 (1위) | 0.7310 | 0.9842 | 0.8685 | 0.8779 |
| 🥈 2위 | Upstage Solar (API) + SPLADE + ColBERT | 0.8559 | −0.1% | 0.7069 | 0.9955 | 0.9105 | 0.8516 |
| 🥉 3위 | BGE-M3 + SPLADE + ColBERT | 0.8553 | −0.1% | 0.7159 | 0.9955 | 0.8987 | 0.8420 |
| 4위 | OpenAI (API) + BM25 + ColBERT | 0.8427 | −1.6% | 0.6877 | 0.9842 | 0.8845 | 0.8657 |
| 5위 | KoSimCSE + BM25 + ColBERT | 0.8416 | −1.7% | 0.6850 | 0.9955 | 0.8649 | 0.8682 |
| 6위 | Upstage Solar (API) + BM25 + ColBERT | 0.8407 | −1.8% | 0.6935 | 0.9842 | 0.8571 | 0.8753 |
| 7위 | E5-multilingual + BM25 + ColBERT | 0.8379 | −2.2% | 0.7032 | 0.9842 | 0.8462 | 0.8486 |
| 8위 | OpenAI (API) + SPLADE + ColBERT | 0.8379 | −2.2% | 0.6954 | 0.9842 | 0.8466 | 0.8659 |
| 9위 | KoSimCSE + SPLADE + ColBERT | 0.8369 | −2.3% | 0.7126 | 0.9955 | 0.8067 | 0.8499 |
| 10위 | BGE-M3 + BM25 + ColBERT | 0.8194 | −4.3% | 0.6698 | 0.9730 | 0.8131 | 0.8694 |
| 11위 | Upstage Solar (API) + BM25 + FlashRank | 0.7701 | −10.1% | 0.7122 | 0.9887 | 0.4583 | 0.8836 |
| 12위 | KoSimCSE + BM25 + FlashRank | 0.7679 | −10.3% | 0.6789 | 0.9820 | 0.5540 | 0.8327 |
| 13위 | OpenAI (API) + BM25 + FlashRank | 0.7532 | −12.1% | 0.6579 | 0.9955 | 0.4995 | 0.8290 |
| 14위 | BGE-M3 + BM25 + FlashRank | 0.7485 | −12.6% | 0.6717 | 0.9910 | 0.4587 | 0.8294 |
| 15위 | E5-multilingual + BM25 + FlashRank | 0.7465 | −12.8% | 0.6250 | 0.9842 | 0.5193 | 0.8578 |
| 16위 | BGE-M3 + SPLADE + FlashRank | 0.7271 | −15.1% | 0.6789 | 0.9437 | 0.4304 | 0.8017 |
| 17위 | OpenAI (API) + SPLADE + FlashRank | 0.7169 | −16.3% | 0.6694 | 0.9797 | 0.3076 | 0.8477 |
| 18위 | KoSimCSE + SPLADE + FlashRank | 0.7099 | −17.1% | 0.6443 | 0.9707 | 0.3769 | 0.7852 |
| 19위 | E5-multilingual + SPLADE + FlashRank | 0.7084 | −17.3% | 0.6525 | 0.9572 | 0.3426 | 0.8292 |
| 20위 | Upstage Solar (API) + SPLADE + FlashRank | 0.6884 | −19.6% | 0.6134 | 0.9640 | 0.2895 | 0.8438 |

#### 핵심 인사이트

**1. E5-multilingual + SPLADE + ColBERT가 종합 1위인 이유**
> Context Recall 지표에서 전 조합 중 최고점을 기록했습니다.
> 2위(Upstage Solar (API) + SPLADE + ColBERT) 대비 0.1% 우세합니다.

**2. 1위와 2위 성능 차이가 미미**
> 1위와 2위의 종합 점수 차이가 0.0006 (0.1%)로 통계적으로 유의미하지 않을 수 있습니다.
> 두 조합 모두 실서비스 적용이 가능하며, **운영 비용·보안 요건**(API vs 로컬)에 따라 선택하는 것이 합리적입니다.

**3. 로컬 모델이 API 모델과 동등 이상의 성능**
> 로컬 최고(E5-multilingual + SPLADE + ColBERT, 0.8565)가 API 최고(Upstage Solar (API) + SPLADE + ColBERT, 0.8559)보다 0.0006 우세합니다.
> API 모델이 반드시 우수하지 않으므로, 보안·비용 측면에서 로컬 모델이 유리합니다.

### 3-4. MEDICAL 카테고리

#### 종합 순위표

> 점수는 0~1 범위 (높을수록 우수). 1위 대비 차이(%p)를 병기합니다.

| 순위 | AI 조합 | 종합 점수 | 1위 대비 | Recall | Precision | Faithfulness | Relevancy |
|:----:|:--------|:--------:|:-------:|:------:|:---------:|:------------:|:---------:|
| 🥇 **1위** | E5-multilingual + BM25 + ColBERT | **0.9166** | 기준 (1위) | 0.9111 | 0.9762 | 0.8708 | 0.8714 |
| 🥈 2위 | BGE-M3 + BM25 + ColBERT | 0.9146 | −0.2% | 0.9111 | 0.9697 | 0.8584 | 0.8878 |
| 🥉 3위 | OpenAI (API) + BM25 + ColBERT | 0.9146 | −0.2% | 0.9063 | 0.9773 | 0.8662 | 0.8730 |
| 4위 | BGE-M3 + SPLADE + ColBERT | 0.9134 | −0.3% | 0.9155 | 0.9697 | 0.8701 | 0.8538 |
| 5위 | OpenAI (API) + SPLADE + ColBERT | 0.9115 | −0.6% | 0.9104 | 0.9567 | 0.8734 | 0.8748 |
| 6위 | E5-multilingual + SPLADE + ColBERT | 0.9109 | −0.6% | 0.9125 | 0.9740 | 0.8614 | 0.8470 |
| 7위 | Upstage Solar (API) + SPLADE + ColBERT | 0.9100 | −0.7% | 0.9032 | 0.9675 | 0.8686 | 0.8663 |
| 8위 | KoSimCSE + BM25 + ColBERT | 0.9045 | −1.3% | 0.9060 | 0.9610 | 0.8514 | 0.8589 |
| 9위 | KoSimCSE + SPLADE + ColBERT | 0.8817 | −3.8% | 0.8661 | 0.9372 | 0.8414 | 0.8609 |
| 10위 | Upstage Solar (API) + BM25 + ColBERT | 0.8256 | −9.9% | 0.8445 | 0.8074 | 0.7781 | 0.8814 |
| 11위 | BGE-M3 + BM25 + FlashRank | 0.6418 | −30.0% | 0.5791 | 0.6872 | 0.5006 | 0.8857 |
| 12위 | OpenAI (API) + BM25 + FlashRank | 0.6369 | −30.5% | 0.5849 | 0.6721 | 0.4961 | 0.8757 |
| 13위 | E5-multilingual + BM25 + FlashRank | 0.6057 | −33.9% | 0.5087 | 0.6613 | 0.4905 | 0.8747 |
| 14위 | Upstage Solar (API) + BM25 + FlashRank | 0.5854 | −36.1% | 0.5033 | 0.5823 | 0.5152 | 0.8768 |
| 15위 | KoSimCSE + BM25 + FlashRank | 0.5833 | −36.4% | 0.5375 | 0.6061 | 0.4106 | 0.8748 |
| 16위 | E5-multilingual + SPLADE + FlashRank | 0.5244 | −42.8% | 0.4311 | 0.5649 | 0.3709 | 0.8661 |
| 17위 | KoSimCSE + SPLADE + FlashRank | 0.5197 | −43.3% | 0.4685 | 0.5292 | 0.3219 | 0.8840 |
| 18위 | Upstage Solar (API) + SPLADE + FlashRank | 0.5163 | −43.7% | 0.4494 | 0.5108 | 0.3633 | 0.8875 |
| 19위 | BGE-M3 + SPLADE + FlashRank | 0.5156 | −43.7% | 0.4237 | 0.5703 | 0.3239 | 0.8765 |
| 20위 | OpenAI (API) + SPLADE + FlashRank | 0.5092 | −44.4% | 0.4225 | 0.5173 | 0.3752 | 0.8741 |

#### 핵심 인사이트

**1. E5-multilingual + BM25 + ColBERT가 종합 1위인 이유**
> 모든 지표에서 균형 있게 높은 성능을 보여 가중 합산에서 최고 점수를 달성했습니다.
> 2위(BGE-M3 + BM25 + ColBERT) 대비 0.2% 우세합니다.

**2. 1위와 2위 성능 차이가 미미**
> 1위와 2위의 종합 점수 차이가 0.0020 (0.2%)로 통계적으로 유의미하지 않을 수 있습니다.
> 두 조합 모두 실서비스 적용이 가능하며, **운영 비용·보안 요건**(API vs 로컬)에 따라 선택하는 것이 합리적입니다.

**3. 로컬 모델이 API 모델과 동등 이상의 성능**
> 로컬 최고(E5-multilingual + BM25 + ColBERT, 0.9166)가 API 최고(OpenAI (API) + BM25 + ColBERT, 0.9146)보다 0.0020 우세합니다.
> API 모델이 반드시 우수하지 않으므로, 보안·비용 측면에서 로컬 모델이 유리합니다.

### 3-5. TECHNICAL 카테고리

#### 종합 순위표

> 점수는 0~1 범위 (높을수록 우수). 1위 대비 차이(%p)를 병기합니다.

| 순위 | AI 조합 | 종합 점수 | 1위 대비 | Recall | Precision | Faithfulness | Relevancy |
|:----:|:--------|:--------:|:-------:|:------:|:---------:|:------------:|:---------:|
| 🥇 **1위** | E5-multilingual + BM25 + ColBERT | **0.5229** | 기준 (1위) | 0.3429 | 0.4633 | 0.6715 | 0.8637 |
| 🥈 2위 | Upstage Solar (API) + BM25 + ColBERT | 0.5194 | −0.7% | 0.3499 | 0.4517 | 0.6671 | 0.8535 |
| 🥉 3위 | OpenAI (API) + SPLADE + ColBERT | 0.5189 | −0.8% | 0.3468 | 0.4850 | 0.6220 | 0.8505 |
| 4위 | BGE-M3 + SPLADE + ColBERT | 0.5166 | −1.2% | 0.3160 | 0.4817 | 0.6557 | 0.8687 |
| 5위 | BGE-M3 + BM25 + ColBERT | 0.5146 | −1.6% | 0.3421 | 0.4600 | 0.6490 | 0.8469 |
| 6위 | E5-multilingual + SPLADE + ColBERT | 0.5109 | −2.3% | 0.3207 | 0.4650 | 0.6580 | 0.8502 |
| 7위 | OpenAI (API) + BM25 + ColBERT | 0.5067 | −3.1% | 0.3332 | 0.4500 | 0.6303 | 0.8600 |
| 8위 | KoSimCSE + BM25 + ColBERT | 0.5000 | −4.4% | 0.2994 | 0.4483 | 0.6570 | 0.8620 |
| 9위 | Upstage Solar (API) + SPLADE + ColBERT | 0.4788 | −8.4% | 0.3124 | 0.3633 | 0.6612 | 0.8548 |
| 10위 | KoSimCSE + SPLADE + ColBERT | 0.4670 | −10.7% | 0.2724 | 0.4800 | 0.4893 | 0.8652 |
| 11위 | Upstage Solar (API) + BM25 + FlashRank | 0.4486 | −14.2% | 0.2940 | 0.4400 | 0.4242 | 0.8588 |
| 12위 | OpenAI (API) + BM25 + FlashRank | 0.3974 | −24.0% | 0.2357 | 0.3900 | 0.3602 | 0.8390 |
| 13위 | BGE-M3 + BM25 + FlashRank | 0.3882 | −25.8% | 0.1939 | 0.3350 | 0.4521 | 0.8628 |
| 14위 | E5-multilingual + BM25 + FlashRank | 0.3548 | −32.1% | 0.2222 | 0.3000 | 0.2978 | 0.8500 |
| 15위 | KoSimCSE + BM25 + FlashRank | 0.3466 | −33.7% | 0.1693 | 0.2967 | 0.3472 | 0.8590 |
| 16위 | BGE-M3 + SPLADE + FlashRank | 0.3332 | −36.3% | 0.1824 | 0.3067 | 0.2363 | 0.8675 |
| 17위 | E5-multilingual + SPLADE + FlashRank | 0.3254 | −37.8% | 0.1497 | 0.2983 | 0.2669 | 0.8675 |
| 18위 | OpenAI (API) + SPLADE + FlashRank | 0.3239 | −38.1% | 0.1568 | 0.3133 | 0.2246 | 0.8671 |
| 19위 | Upstage Solar (API) + SPLADE + FlashRank | 0.3105 | −40.6% | 0.1575 | 0.2900 | 0.1975 | 0.8595 |
| 20위 | KoSimCSE + SPLADE + FlashRank | 0.2885 | −44.8% | 0.1329 | 0.2633 | 0.1577 | 0.8762 |

#### 핵심 인사이트

**1. E5-multilingual + BM25 + ColBERT가 종합 1위인 이유**
> Faithfulness 지표에서 전 조합 중 최고점을 기록했습니다.
> 2위(Upstage Solar (API) + BM25 + ColBERT) 대비 0.7% 우세합니다.

**2. 1위와 2위 성능 차이가 미미**
> 1위와 2위의 종합 점수 차이가 0.0035 (0.7%)로 통계적으로 유의미하지 않을 수 있습니다.
> 두 조합 모두 실서비스 적용이 가능하며, **운영 비용·보안 요건**(API vs 로컬)에 따라 선택하는 것이 합리적입니다.

**3. 로컬 모델이 API 모델과 동등 이상의 성능**
> 로컬 최고(E5-multilingual + BM25 + ColBERT, 0.5229)가 API 최고(Upstage Solar (API) + BM25 + ColBERT, 0.5194)보다 0.0035 우세합니다.
> API 모델이 반드시 우수하지 않으므로, 보안·비용 측면에서 로컬 모델이 유리합니다.

---

## 4. 지표별 상세 분석

### 4-1. BUSINESS 카테고리

#### Faithfulness — 지어내지 않는 능력

> 답변이 검색된 문서 내용에 근거하는지 측정합니다. 할루시네이션(거짓 정보 생성) 방지의 핵심 지표입니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | KoSimCSE + BM25 + ColBERT | 0.8601 | **최고** |
| 2 | OpenAI (API) + SPLADE + ColBERT | 0.8531 |  |
| 3 | Upstage Solar (API) + SPLADE + ColBERT | 0.8518 |  |
| 4 | E5-multilingual + SPLADE + ColBERT | 0.8415 |  |
| 5 | KoSimCSE + SPLADE + ColBERT | 0.8411 |  |
| 6 | BGE-M3 + SPLADE + ColBERT | 0.8410 |  |
| 7 | OpenAI (API) + BM25 + ColBERT | 0.8345 |  |
| 8 | BGE-M3 + BM25 + ColBERT | 0.8308 |  |
| 9 | E5-multilingual + BM25 + ColBERT | 0.8300 |  |
| 10 | Upstage Solar (API) + BM25 + ColBERT | 0.8227 |  |
| 11 | BGE-M3 + BM25 + FlashRank | 0.4955 |  |
| 12 | OpenAI (API) + BM25 + FlashRank | 0.4943 |  |
| 13 | E5-multilingual + BM25 + FlashRank | 0.4582 |  |
| 14 | KoSimCSE + BM25 + FlashRank | 0.4536 |  |
| 15 | Upstage Solar (API) + BM25 + FlashRank | 0.3779 |  |
| 16 | E5-multilingual + SPLADE + FlashRank | 0.3649 |  |
| 17 | Upstage Solar (API) + SPLADE + FlashRank | 0.3497 |  |
| 18 | BGE-M3 + SPLADE + FlashRank | 0.3312 |  |
| 19 | OpenAI (API) + SPLADE + FlashRank | 0.3295 |  |
| 20 | KoSimCSE + SPLADE + FlashRank | 0.3245 | 최고 대비 −0.5356 |

> 전체 편차: 0.5356 (62.3%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Recall — 놓치지 않는 능력

> 정답에 필요한 문서를 빠짐없이 검색하는지 측정합니다. 정보 누락은 서비스 신뢰도를 직접 훼손합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | BGE-M3 + SPLADE + ColBERT | 0.7636 | **최고** |
| 2 | OpenAI (API) + SPLADE + ColBERT | 0.7629 |  |
| 3 | KoSimCSE + BM25 + ColBERT | 0.7577 |  |
| 4 | BGE-M3 + BM25 + ColBERT | 0.7571 |  |
| 5 | E5-multilingual + SPLADE + ColBERT | 0.7568 |  |
| 6 | E5-multilingual + BM25 + ColBERT | 0.7566 |  |
| 7 | Upstage Solar (API) + SPLADE + ColBERT | 0.7559 |  |
| 8 | OpenAI (API) + BM25 + ColBERT | 0.7541 |  |
| 9 | Upstage Solar (API) + BM25 + ColBERT | 0.7483 |  |
| 10 | KoSimCSE + SPLADE + ColBERT | 0.7392 |  |
| 11 | E5-multilingual + BM25 + FlashRank | 0.7135 |  |
| 12 | Upstage Solar (API) + BM25 + FlashRank | 0.6977 |  |
| 13 | BGE-M3 + BM25 + FlashRank | 0.6727 |  |
| 14 | KoSimCSE + BM25 + FlashRank | 0.6419 |  |
| 15 | BGE-M3 + SPLADE + FlashRank | 0.6407 |  |
| 16 | OpenAI (API) + BM25 + FlashRank | 0.6364 |  |
| 17 | E5-multilingual + SPLADE + FlashRank | 0.6199 |  |
| 18 | Upstage Solar (API) + SPLADE + FlashRank | 0.6103 |  |
| 19 | OpenAI (API) + SPLADE + FlashRank | 0.5789 |  |
| 20 | KoSimCSE + SPLADE + FlashRank | 0.5707 | 최고 대비 −0.1929 |

> 전체 편차: 0.1929 (25.3%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Precision — 정확하게 찾는 능력

> 검색 결과 중 실제 관련 문서의 비율을 측정합니다. 불필요한 문서가 많으면 LLM 답변 품질이 저하됩니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | KoSimCSE + SPLADE + ColBERT | 0.9978 | **최고** |
| 2 | KoSimCSE + BM25 + ColBERT | 0.9946 |  |
| 3 | Upstage Solar (API) + SPLADE + ColBERT | 0.9870 |  |
| 4 | OpenAI (API) + BM25 + ColBERT | 0.9870 |  |
| 5 | Upstage Solar (API) + BM25 + ColBERT | 0.9870 |  |
| 6 | OpenAI (API) + SPLADE + ColBERT | 0.9870 |  |
| 7 | BGE-M3 + SPLADE + ColBERT | 0.9848 |  |
| 8 | E5-multilingual + SPLADE + ColBERT | 0.9848 |  |
| 9 | BGE-M3 + BM25 + ColBERT | 0.9848 |  |
| 10 | E5-multilingual + BM25 + ColBERT | 0.9848 |  |
| 11 | Upstage Solar (API) + BM25 + FlashRank | 0.9762 |  |
| 12 | BGE-M3 + BM25 + FlashRank | 0.9740 |  |
| 13 | OpenAI (API) + BM25 + FlashRank | 0.9740 |  |
| 14 | KoSimCSE + BM25 + FlashRank | 0.9675 |  |
| 15 | Upstage Solar (API) + SPLADE + FlashRank | 0.9665 |  |
| 16 | E5-multilingual + BM25 + FlashRank | 0.9632 |  |
| 17 | KoSimCSE + SPLADE + FlashRank | 0.9426 |  |
| 18 | OpenAI (API) + SPLADE + FlashRank | 0.9405 |  |
| 19 | BGE-M3 + SPLADE + FlashRank | 0.9361 |  |
| 20 | E5-multilingual + SPLADE + FlashRank | 0.9232 | 최고 대비 −0.0746 |

> 전체 편차: 0.0746 (7.5%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Answer Relevancy — 질문에 답하는 능력

> 생성된 답변이 질문에 직접적으로 대응하는지 측정합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | E5-multilingual + SPLADE + ColBERT | 0.8845 | **최고** |
| 2 | OpenAI (API) + BM25 + ColBERT | 0.8835 |  |
| 3 | BGE-M3 + SPLADE + ColBERT | 0.8810 |  |
| 4 | KoSimCSE + SPLADE + ColBERT | 0.8799 |  |
| 5 | BGE-M3 + BM25 + ColBERT | 0.8784 |  |
| 6 | Upstage Solar (API) + SPLADE + ColBERT | 0.8776 |  |
| 7 | Upstage Solar (API) + BM25 + FlashRank | 0.8775 |  |
| 8 | OpenAI (API) + BM25 + FlashRank | 0.8766 |  |
| 9 | E5-multilingual + BM25 + FlashRank | 0.8751 |  |
| 10 | KoSimCSE + BM25 + ColBERT | 0.8746 |  |
| 11 | E5-multilingual + BM25 + ColBERT | 0.8736 |  |
| 12 | Upstage Solar (API) + BM25 + ColBERT | 0.8730 |  |
| 13 | BGE-M3 + BM25 + FlashRank | 0.8719 |  |
| 14 | E5-multilingual + SPLADE + FlashRank | 0.8660 |  |
| 15 | OpenAI (API) + SPLADE + ColBERT | 0.8623 |  |
| 16 | Upstage Solar (API) + SPLADE + FlashRank | 0.8574 |  |
| 17 | KoSimCSE + BM25 + FlashRank | 0.8565 |  |
| 18 | OpenAI (API) + SPLADE + FlashRank | 0.8538 |  |
| 19 | BGE-M3 + SPLADE + FlashRank | 0.8336 |  |
| 20 | KoSimCSE + SPLADE + FlashRank | 0.8272 | 최고 대비 −0.0573 |

> 전체 편차: 0.0573 (6.5%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

### 4-2. GENERAL 카테고리

#### Faithfulness — 지어내지 않는 능력

> 답변이 검색된 문서 내용에 근거하는지 측정합니다. 할루시네이션(거짓 정보 생성) 방지의 핵심 지표입니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | BGE-M3 + SPLADE + ColBERT | 0.7922 | **최고** |
| 2 | Upstage Solar (API) + SPLADE + ColBERT | 0.7725 |  |
| 3 | KoSimCSE + SPLADE + ColBERT | 0.7685 |  |
| 4 | E5-multilingual + SPLADE + ColBERT | 0.7672 |  |
| 5 | Upstage Solar (API) + BM25 + ColBERT | 0.7578 |  |
| 6 | BGE-M3 + BM25 + ColBERT | 0.7557 |  |
| 7 | E5-multilingual + BM25 + ColBERT | 0.7548 |  |
| 8 | OpenAI (API) + SPLADE + ColBERT | 0.7522 |  |
| 9 | KoSimCSE + BM25 + ColBERT | 0.7432 |  |
| 10 | OpenAI (API) + BM25 + ColBERT | 0.7067 |  |
| 11 | BGE-M3 + BM25 + FlashRank | 0.3115 |  |
| 12 | OpenAI (API) + BM25 + FlashRank | 0.3000 |  |
| 13 | Upstage Solar (API) + BM25 + FlashRank | 0.2843 |  |
| 14 | KoSimCSE + BM25 + FlashRank | 0.2646 |  |
| 15 | Upstage Solar (API) + SPLADE + FlashRank | 0.2628 |  |
| 16 | KoSimCSE + SPLADE + FlashRank | 0.2615 |  |
| 17 | BGE-M3 + SPLADE + FlashRank | 0.2600 |  |
| 18 | E5-multilingual + BM25 + FlashRank | 0.2569 |  |
| 19 | E5-multilingual + SPLADE + FlashRank | 0.2360 |  |
| 20 | OpenAI (API) + SPLADE + FlashRank | 0.2107 | 최고 대비 −0.5815 |

> 전체 편차: 0.5815 (73.4%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Recall — 놓치지 않는 능력

> 정답에 필요한 문서를 빠짐없이 검색하는지 측정합니다. 정보 누락은 서비스 신뢰도를 직접 훼손합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | OpenAI (API) + BM25 + ColBERT | 0.8572 | **최고** |
| 2 | BGE-M3 + BM25 + ColBERT | 0.8552 |  |
| 3 | E5-multilingual + BM25 + ColBERT | 0.8307 |  |
| 4 | BGE-M3 + SPLADE + ColBERT | 0.8299 |  |
| 5 | E5-multilingual + SPLADE + ColBERT | 0.8284 |  |
| 6 | Upstage Solar (API) + BM25 + ColBERT | 0.8251 |  |
| 7 | Upstage Solar (API) + SPLADE + ColBERT | 0.8246 |  |
| 8 | OpenAI (API) + SPLADE + ColBERT | 0.8228 |  |
| 9 | KoSimCSE + BM25 + ColBERT | 0.8180 |  |
| 10 | KoSimCSE + SPLADE + ColBERT | 0.8144 |  |
| 11 | KoSimCSE + SPLADE + FlashRank | 0.5897 |  |
| 12 | E5-multilingual + SPLADE + FlashRank | 0.5772 |  |
| 13 | OpenAI (API) + SPLADE + FlashRank | 0.5745 |  |
| 14 | Upstage Solar (API) + SPLADE + FlashRank | 0.5662 |  |
| 15 | BGE-M3 + SPLADE + FlashRank | 0.5643 |  |
| 16 | OpenAI (API) + BM25 + FlashRank | 0.5031 |  |
| 17 | KoSimCSE + BM25 + FlashRank | 0.4969 |  |
| 18 | E5-multilingual + BM25 + FlashRank | 0.4839 |  |
| 19 | Upstage Solar (API) + BM25 + FlashRank | 0.4670 |  |
| 20 | BGE-M3 + BM25 + FlashRank | 0.4622 | 최고 대비 −0.3950 |

> 전체 편차: 0.3950 (46.1%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Precision — 정확하게 찾는 능력

> 검색 결과 중 실제 관련 문서의 비율을 측정합니다. 불필요한 문서가 많으면 LLM 답변 품질이 저하됩니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | KoSimCSE + BM25 + ColBERT | 0.9275 | **최고** |
| 2 | Upstage Solar (API) + BM25 + ColBERT | 0.9258 |  |
| 3 | OpenAI (API) + BM25 + ColBERT | 0.9233 |  |
| 4 | E5-multilingual + BM25 + ColBERT | 0.9233 |  |
| 5 | BGE-M3 + BM25 + ColBERT | 0.9183 |  |
| 6 | Upstage Solar (API) + SPLADE + ColBERT | 0.9150 |  |
| 7 | KoSimCSE + SPLADE + ColBERT | 0.9142 |  |
| 8 | E5-multilingual + SPLADE + ColBERT | 0.9133 |  |
| 9 | OpenAI (API) + SPLADE + ColBERT | 0.9092 |  |
| 10 | BGE-M3 + SPLADE + ColBERT | 0.9067 |  |
| 11 | Upstage Solar (API) + BM25 + FlashRank | 0.8333 |  |
| 12 | Upstage Solar (API) + SPLADE + FlashRank | 0.8333 |  |
| 13 | KoSimCSE + BM25 + FlashRank | 0.8292 |  |
| 14 | OpenAI (API) + BM25 + FlashRank | 0.8258 |  |
| 15 | BGE-M3 + BM25 + FlashRank | 0.8242 |  |
| 16 | E5-multilingual + BM25 + FlashRank | 0.8167 |  |
| 17 | KoSimCSE + SPLADE + FlashRank | 0.8000 |  |
| 18 | OpenAI (API) + SPLADE + FlashRank | 0.7942 |  |
| 19 | E5-multilingual + SPLADE + FlashRank | 0.7850 |  |
| 20 | BGE-M3 + SPLADE + FlashRank | 0.7825 | 최고 대비 −0.1450 |

> 전체 편차: 0.1450 (15.6%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Answer Relevancy — 질문에 답하는 능력

> 생성된 답변이 질문에 직접적으로 대응하는지 측정합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | BGE-M3 + SPLADE + ColBERT | 0.8846 | **최고** |
| 2 | OpenAI (API) + BM25 + ColBERT | 0.8774 |  |
| 3 | KoSimCSE + SPLADE + ColBERT | 0.8690 |  |
| 4 | E5-multilingual + SPLADE + ColBERT | 0.8683 |  |
| 5 | E5-multilingual + BM25 + ColBERT | 0.8681 |  |
| 6 | BGE-M3 + BM25 + ColBERT | 0.8675 |  |
| 7 | Upstage Solar (API) + SPLADE + ColBERT | 0.8666 |  |
| 8 | KoSimCSE + BM25 + ColBERT | 0.8665 |  |
| 9 | Upstage Solar (API) + BM25 + ColBERT | 0.8652 |  |
| 10 | OpenAI (API) + SPLADE + ColBERT | 0.8645 |  |
| 11 | OpenAI (API) + BM25 + FlashRank | 0.7499 |  |
| 12 | Upstage Solar (API) + BM25 + FlashRank | 0.7152 |  |
| 13 | KoSimCSE + BM25 + FlashRank | 0.7089 |  |
| 14 | E5-multilingual + SPLADE + FlashRank | 0.7074 |  |
| 15 | E5-multilingual + BM25 + FlashRank | 0.6955 |  |
| 16 | OpenAI (API) + SPLADE + FlashRank | 0.6689 |  |
| 17 | Upstage Solar (API) + SPLADE + FlashRank | 0.6685 |  |
| 18 | BGE-M3 + BM25 + FlashRank | 0.6679 |  |
| 19 | KoSimCSE + SPLADE + FlashRank | 0.6615 |  |
| 20 | BGE-M3 + SPLADE + FlashRank | 0.6549 | 최고 대비 −0.2297 |

> 전체 편차: 0.2297 (26.0%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

### 4-3. LEGAL 카테고리

#### Faithfulness — 지어내지 않는 능력

> 답변이 검색된 문서 내용에 근거하는지 측정합니다. 할루시네이션(거짓 정보 생성) 방지의 핵심 지표입니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | Upstage Solar (API) + SPLADE + ColBERT | 0.9105 | **최고** |
| 2 | BGE-M3 + SPLADE + ColBERT | 0.8987 |  |
| 3 | OpenAI (API) + BM25 + ColBERT | 0.8845 |  |
| 4 | E5-multilingual + SPLADE + ColBERT | 0.8685 |  |
| 5 | KoSimCSE + BM25 + ColBERT | 0.8649 |  |
| 6 | Upstage Solar (API) + BM25 + ColBERT | 0.8571 |  |
| 7 | OpenAI (API) + SPLADE + ColBERT | 0.8466 |  |
| 8 | E5-multilingual + BM25 + ColBERT | 0.8462 |  |
| 9 | BGE-M3 + BM25 + ColBERT | 0.8131 |  |
| 10 | KoSimCSE + SPLADE + ColBERT | 0.8067 |  |
| 11 | KoSimCSE + BM25 + FlashRank | 0.5540 |  |
| 12 | E5-multilingual + BM25 + FlashRank | 0.5193 |  |
| 13 | OpenAI (API) + BM25 + FlashRank | 0.4995 |  |
| 14 | BGE-M3 + BM25 + FlashRank | 0.4587 |  |
| 15 | Upstage Solar (API) + BM25 + FlashRank | 0.4583 |  |
| 16 | BGE-M3 + SPLADE + FlashRank | 0.4304 |  |
| 17 | KoSimCSE + SPLADE + FlashRank | 0.3769 |  |
| 18 | E5-multilingual + SPLADE + FlashRank | 0.3426 |  |
| 19 | OpenAI (API) + SPLADE + FlashRank | 0.3076 |  |
| 20 | Upstage Solar (API) + SPLADE + FlashRank | 0.2895 | 최고 대비 −0.6210 |

> 전체 편차: 0.6210 (68.2%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Recall — 놓치지 않는 능력

> 정답에 필요한 문서를 빠짐없이 검색하는지 측정합니다. 정보 누락은 서비스 신뢰도를 직접 훼손합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | E5-multilingual + SPLADE + ColBERT | 0.7310 | **최고** |
| 2 | BGE-M3 + SPLADE + ColBERT | 0.7159 |  |
| 3 | KoSimCSE + SPLADE + ColBERT | 0.7126 |  |
| 4 | Upstage Solar (API) + BM25 + FlashRank | 0.7122 |  |
| 5 | Upstage Solar (API) + SPLADE + ColBERT | 0.7069 |  |
| 6 | E5-multilingual + BM25 + ColBERT | 0.7032 |  |
| 7 | OpenAI (API) + SPLADE + ColBERT | 0.6954 |  |
| 8 | Upstage Solar (API) + BM25 + ColBERT | 0.6935 |  |
| 9 | OpenAI (API) + BM25 + ColBERT | 0.6877 |  |
| 10 | KoSimCSE + BM25 + ColBERT | 0.6850 |  |
| 11 | KoSimCSE + BM25 + FlashRank | 0.6789 |  |
| 12 | BGE-M3 + SPLADE + FlashRank | 0.6789 |  |
| 13 | BGE-M3 + BM25 + FlashRank | 0.6717 |  |
| 14 | BGE-M3 + BM25 + ColBERT | 0.6698 |  |
| 15 | OpenAI (API) + SPLADE + FlashRank | 0.6694 |  |
| 16 | OpenAI (API) + BM25 + FlashRank | 0.6579 |  |
| 17 | E5-multilingual + SPLADE + FlashRank | 0.6525 |  |
| 18 | KoSimCSE + SPLADE + FlashRank | 0.6443 |  |
| 19 | E5-multilingual + BM25 + FlashRank | 0.6250 |  |
| 20 | Upstage Solar (API) + SPLADE + FlashRank | 0.6134 | 최고 대비 −0.1176 |

> 전체 편차: 0.1176 (16.1%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Precision — 정확하게 찾는 능력

> 검색 결과 중 실제 관련 문서의 비율을 측정합니다. 불필요한 문서가 많으면 LLM 답변 품질이 저하됩니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | KoSimCSE + SPLADE + ColBERT | 0.9955 | **최고** |
| 2 | BGE-M3 + SPLADE + ColBERT | 0.9955 |  |
| 3 | KoSimCSE + BM25 + ColBERT | 0.9955 |  |
| 4 | OpenAI (API) + BM25 + FlashRank | 0.9955 |  |
| 5 | Upstage Solar (API) + SPLADE + ColBERT | 0.9955 |  |
| 6 | BGE-M3 + BM25 + FlashRank | 0.9910 |  |
| 7 | Upstage Solar (API) + BM25 + FlashRank | 0.9887 |  |
| 8 | E5-multilingual + BM25 + FlashRank | 0.9842 |  |
| 9 | E5-multilingual + SPLADE + ColBERT | 0.9842 |  |
| 10 | OpenAI (API) + SPLADE + ColBERT | 0.9842 |  |
| 11 | E5-multilingual + BM25 + ColBERT | 0.9842 |  |
| 12 | Upstage Solar (API) + BM25 + ColBERT | 0.9842 |  |
| 13 | OpenAI (API) + BM25 + ColBERT | 0.9842 |  |
| 14 | KoSimCSE + BM25 + FlashRank | 0.9820 |  |
| 15 | OpenAI (API) + SPLADE + FlashRank | 0.9797 |  |
| 16 | BGE-M3 + BM25 + ColBERT | 0.9730 |  |
| 17 | KoSimCSE + SPLADE + FlashRank | 0.9707 |  |
| 18 | Upstage Solar (API) + SPLADE + FlashRank | 0.9640 |  |
| 19 | E5-multilingual + SPLADE + FlashRank | 0.9572 |  |
| 20 | BGE-M3 + SPLADE + FlashRank | 0.9437 | 최고 대비 −0.0518 |

> 전체 편차: 0.0518 (5.2%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Answer Relevancy — 질문에 답하는 능력

> 생성된 답변이 질문에 직접적으로 대응하는지 측정합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | Upstage Solar (API) + BM25 + FlashRank | 0.8836 | **최고** |
| 2 | E5-multilingual + SPLADE + ColBERT | 0.8779 |  |
| 3 | Upstage Solar (API) + BM25 + ColBERT | 0.8753 |  |
| 4 | BGE-M3 + BM25 + ColBERT | 0.8694 |  |
| 5 | KoSimCSE + BM25 + ColBERT | 0.8682 |  |
| 6 | OpenAI (API) + SPLADE + ColBERT | 0.8659 |  |
| 7 | OpenAI (API) + BM25 + ColBERT | 0.8657 |  |
| 8 | E5-multilingual + BM25 + FlashRank | 0.8578 |  |
| 9 | Upstage Solar (API) + SPLADE + ColBERT | 0.8516 |  |
| 10 | KoSimCSE + SPLADE + ColBERT | 0.8499 |  |
| 11 | E5-multilingual + BM25 + ColBERT | 0.8486 |  |
| 12 | OpenAI (API) + SPLADE + FlashRank | 0.8477 |  |
| 13 | Upstage Solar (API) + SPLADE + FlashRank | 0.8438 |  |
| 14 | BGE-M3 + SPLADE + ColBERT | 0.8420 |  |
| 15 | KoSimCSE + BM25 + FlashRank | 0.8327 |  |
| 16 | BGE-M3 + BM25 + FlashRank | 0.8294 |  |
| 17 | E5-multilingual + SPLADE + FlashRank | 0.8292 |  |
| 18 | OpenAI (API) + BM25 + FlashRank | 0.8290 |  |
| 19 | BGE-M3 + SPLADE + FlashRank | 0.8017 |  |
| 20 | KoSimCSE + SPLADE + FlashRank | 0.7852 | 최고 대비 −0.0984 |

> 전체 편차: 0.0984 (11.1%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

### 4-4. MEDICAL 카테고리

#### Faithfulness — 지어내지 않는 능력

> 답변이 검색된 문서 내용에 근거하는지 측정합니다. 할루시네이션(거짓 정보 생성) 방지의 핵심 지표입니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | OpenAI (API) + SPLADE + ColBERT | 0.8734 | **최고** |
| 2 | E5-multilingual + BM25 + ColBERT | 0.8708 |  |
| 3 | BGE-M3 + SPLADE + ColBERT | 0.8701 |  |
| 4 | Upstage Solar (API) + SPLADE + ColBERT | 0.8686 |  |
| 5 | OpenAI (API) + BM25 + ColBERT | 0.8662 |  |
| 6 | E5-multilingual + SPLADE + ColBERT | 0.8614 |  |
| 7 | BGE-M3 + BM25 + ColBERT | 0.8584 |  |
| 8 | KoSimCSE + BM25 + ColBERT | 0.8514 |  |
| 9 | KoSimCSE + SPLADE + ColBERT | 0.8414 |  |
| 10 | Upstage Solar (API) + BM25 + ColBERT | 0.7781 |  |
| 11 | Upstage Solar (API) + BM25 + FlashRank | 0.5152 |  |
| 12 | BGE-M3 + BM25 + FlashRank | 0.5006 |  |
| 13 | OpenAI (API) + BM25 + FlashRank | 0.4961 |  |
| 14 | E5-multilingual + BM25 + FlashRank | 0.4905 |  |
| 15 | KoSimCSE + BM25 + FlashRank | 0.4106 |  |
| 16 | OpenAI (API) + SPLADE + FlashRank | 0.3752 |  |
| 17 | E5-multilingual + SPLADE + FlashRank | 0.3709 |  |
| 18 | Upstage Solar (API) + SPLADE + FlashRank | 0.3633 |  |
| 19 | BGE-M3 + SPLADE + FlashRank | 0.3239 |  |
| 20 | KoSimCSE + SPLADE + FlashRank | 0.3219 | 최고 대비 −0.5515 |

> 전체 편차: 0.5515 (63.1%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Recall — 놓치지 않는 능력

> 정답에 필요한 문서를 빠짐없이 검색하는지 측정합니다. 정보 누락은 서비스 신뢰도를 직접 훼손합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | BGE-M3 + SPLADE + ColBERT | 0.9155 | **최고** |
| 2 | E5-multilingual + SPLADE + ColBERT | 0.9125 |  |
| 3 | E5-multilingual + BM25 + ColBERT | 0.9111 |  |
| 4 | BGE-M3 + BM25 + ColBERT | 0.9111 |  |
| 5 | OpenAI (API) + SPLADE + ColBERT | 0.9104 |  |
| 6 | OpenAI (API) + BM25 + ColBERT | 0.9063 |  |
| 7 | KoSimCSE + BM25 + ColBERT | 0.9060 |  |
| 8 | Upstage Solar (API) + SPLADE + ColBERT | 0.9032 |  |
| 9 | KoSimCSE + SPLADE + ColBERT | 0.8661 |  |
| 10 | Upstage Solar (API) + BM25 + ColBERT | 0.8445 |  |
| 11 | OpenAI (API) + BM25 + FlashRank | 0.5849 |  |
| 12 | BGE-M3 + BM25 + FlashRank | 0.5791 |  |
| 13 | KoSimCSE + BM25 + FlashRank | 0.5375 |  |
| 14 | E5-multilingual + BM25 + FlashRank | 0.5087 |  |
| 15 | Upstage Solar (API) + BM25 + FlashRank | 0.5033 |  |
| 16 | KoSimCSE + SPLADE + FlashRank | 0.4685 |  |
| 17 | Upstage Solar (API) + SPLADE + FlashRank | 0.4494 |  |
| 18 | E5-multilingual + SPLADE + FlashRank | 0.4311 |  |
| 19 | BGE-M3 + SPLADE + FlashRank | 0.4237 |  |
| 20 | OpenAI (API) + SPLADE + FlashRank | 0.4225 | 최고 대비 −0.4930 |

> 전체 편차: 0.4930 (53.9%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Precision — 정확하게 찾는 능력

> 검색 결과 중 실제 관련 문서의 비율을 측정합니다. 불필요한 문서가 많으면 LLM 답변 품질이 저하됩니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | OpenAI (API) + BM25 + ColBERT | 0.9773 | **최고** |
| 2 | E5-multilingual + BM25 + ColBERT | 0.9762 |  |
| 3 | E5-multilingual + SPLADE + ColBERT | 0.9740 |  |
| 4 | BGE-M3 + BM25 + ColBERT | 0.9697 |  |
| 5 | BGE-M3 + SPLADE + ColBERT | 0.9697 |  |
| 6 | Upstage Solar (API) + SPLADE + ColBERT | 0.9675 |  |
| 7 | KoSimCSE + BM25 + ColBERT | 0.9610 |  |
| 8 | OpenAI (API) + SPLADE + ColBERT | 0.9567 |  |
| 9 | KoSimCSE + SPLADE + ColBERT | 0.9372 |  |
| 10 | Upstage Solar (API) + BM25 + ColBERT | 0.8074 |  |
| 11 | BGE-M3 + BM25 + FlashRank | 0.6872 |  |
| 12 | OpenAI (API) + BM25 + FlashRank | 0.6721 |  |
| 13 | E5-multilingual + BM25 + FlashRank | 0.6613 |  |
| 14 | KoSimCSE + BM25 + FlashRank | 0.6061 |  |
| 15 | Upstage Solar (API) + BM25 + FlashRank | 0.5823 |  |
| 16 | BGE-M3 + SPLADE + FlashRank | 0.5703 |  |
| 17 | E5-multilingual + SPLADE + FlashRank | 0.5649 |  |
| 18 | KoSimCSE + SPLADE + FlashRank | 0.5292 |  |
| 19 | OpenAI (API) + SPLADE + FlashRank | 0.5173 |  |
| 20 | Upstage Solar (API) + SPLADE + FlashRank | 0.5108 | 최고 대비 −0.4665 |

> 전체 편차: 0.4665 (47.7%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Answer Relevancy — 질문에 답하는 능력

> 생성된 답변이 질문에 직접적으로 대응하는지 측정합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | BGE-M3 + BM25 + ColBERT | 0.8878 | **최고** |
| 2 | Upstage Solar (API) + SPLADE + FlashRank | 0.8875 |  |
| 3 | BGE-M3 + BM25 + FlashRank | 0.8857 |  |
| 4 | KoSimCSE + SPLADE + FlashRank | 0.8840 |  |
| 5 | Upstage Solar (API) + BM25 + ColBERT | 0.8814 |  |
| 6 | Upstage Solar (API) + BM25 + FlashRank | 0.8768 |  |
| 7 | BGE-M3 + SPLADE + FlashRank | 0.8765 |  |
| 8 | OpenAI (API) + BM25 + FlashRank | 0.8757 |  |
| 9 | OpenAI (API) + SPLADE + ColBERT | 0.8748 |  |
| 10 | KoSimCSE + BM25 + FlashRank | 0.8748 |  |
| 11 | E5-multilingual + BM25 + FlashRank | 0.8747 |  |
| 12 | OpenAI (API) + SPLADE + FlashRank | 0.8741 |  |
| 13 | OpenAI (API) + BM25 + ColBERT | 0.8730 |  |
| 14 | E5-multilingual + BM25 + ColBERT | 0.8714 |  |
| 15 | Upstage Solar (API) + SPLADE + ColBERT | 0.8663 |  |
| 16 | E5-multilingual + SPLADE + FlashRank | 0.8661 |  |
| 17 | KoSimCSE + SPLADE + ColBERT | 0.8609 |  |
| 18 | KoSimCSE + BM25 + ColBERT | 0.8589 |  |
| 19 | BGE-M3 + SPLADE + ColBERT | 0.8538 |  |
| 20 | E5-multilingual + SPLADE + ColBERT | 0.8470 | 최고 대비 −0.0408 |

> 전체 편차: 0.0408 (4.6%) — 조합 간 차이가 작아 이 지표만으로는 우열을 가리기 어렵습니다.

### 4-5. TECHNICAL 카테고리

#### Faithfulness — 지어내지 않는 능력

> 답변이 검색된 문서 내용에 근거하는지 측정합니다. 할루시네이션(거짓 정보 생성) 방지의 핵심 지표입니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | E5-multilingual + BM25 + ColBERT | 0.6715 | **최고** |
| 2 | Upstage Solar (API) + BM25 + ColBERT | 0.6671 |  |
| 3 | Upstage Solar (API) + SPLADE + ColBERT | 0.6612 |  |
| 4 | E5-multilingual + SPLADE + ColBERT | 0.6580 |  |
| 5 | KoSimCSE + BM25 + ColBERT | 0.6570 |  |
| 6 | BGE-M3 + SPLADE + ColBERT | 0.6557 |  |
| 7 | BGE-M3 + BM25 + ColBERT | 0.6490 |  |
| 8 | OpenAI (API) + BM25 + ColBERT | 0.6303 |  |
| 9 | OpenAI (API) + SPLADE + ColBERT | 0.6220 |  |
| 10 | KoSimCSE + SPLADE + ColBERT | 0.4893 |  |
| 11 | BGE-M3 + BM25 + FlashRank | 0.4521 |  |
| 12 | Upstage Solar (API) + BM25 + FlashRank | 0.4242 |  |
| 13 | OpenAI (API) + BM25 + FlashRank | 0.3602 |  |
| 14 | KoSimCSE + BM25 + FlashRank | 0.3472 |  |
| 15 | E5-multilingual + BM25 + FlashRank | 0.2978 |  |
| 16 | E5-multilingual + SPLADE + FlashRank | 0.2669 |  |
| 17 | BGE-M3 + SPLADE + FlashRank | 0.2363 |  |
| 18 | OpenAI (API) + SPLADE + FlashRank | 0.2246 |  |
| 19 | Upstage Solar (API) + SPLADE + FlashRank | 0.1975 |  |
| 20 | KoSimCSE + SPLADE + FlashRank | 0.1577 | 최고 대비 −0.5138 |

> 전체 편차: 0.5138 (76.5%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Recall — 놓치지 않는 능력

> 정답에 필요한 문서를 빠짐없이 검색하는지 측정합니다. 정보 누락은 서비스 신뢰도를 직접 훼손합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | Upstage Solar (API) + BM25 + ColBERT | 0.3499 | **최고** |
| 2 | OpenAI (API) + SPLADE + ColBERT | 0.3468 |  |
| 3 | E5-multilingual + BM25 + ColBERT | 0.3429 |  |
| 4 | BGE-M3 + BM25 + ColBERT | 0.3421 |  |
| 5 | OpenAI (API) + BM25 + ColBERT | 0.3332 |  |
| 6 | E5-multilingual + SPLADE + ColBERT | 0.3207 |  |
| 7 | BGE-M3 + SPLADE + ColBERT | 0.3160 |  |
| 8 | Upstage Solar (API) + SPLADE + ColBERT | 0.3124 |  |
| 9 | KoSimCSE + BM25 + ColBERT | 0.2994 |  |
| 10 | Upstage Solar (API) + BM25 + FlashRank | 0.2940 |  |
| 11 | KoSimCSE + SPLADE + ColBERT | 0.2724 |  |
| 12 | OpenAI (API) + BM25 + FlashRank | 0.2357 |  |
| 13 | E5-multilingual + BM25 + FlashRank | 0.2222 |  |
| 14 | BGE-M3 + BM25 + FlashRank | 0.1939 |  |
| 15 | BGE-M3 + SPLADE + FlashRank | 0.1824 |  |
| 16 | KoSimCSE + BM25 + FlashRank | 0.1693 |  |
| 17 | Upstage Solar (API) + SPLADE + FlashRank | 0.1575 |  |
| 18 | OpenAI (API) + SPLADE + FlashRank | 0.1568 |  |
| 19 | E5-multilingual + SPLADE + FlashRank | 0.1497 |  |
| 20 | KoSimCSE + SPLADE + FlashRank | 0.1329 | 최고 대비 −0.2170 |

> 전체 편차: 0.2170 (62.0%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Context Precision — 정확하게 찾는 능력

> 검색 결과 중 실제 관련 문서의 비율을 측정합니다. 불필요한 문서가 많으면 LLM 답변 품질이 저하됩니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | OpenAI (API) + SPLADE + ColBERT | 0.4850 | **최고** |
| 2 | BGE-M3 + SPLADE + ColBERT | 0.4817 |  |
| 3 | KoSimCSE + SPLADE + ColBERT | 0.4800 |  |
| 4 | E5-multilingual + SPLADE + ColBERT | 0.4650 |  |
| 5 | E5-multilingual + BM25 + ColBERT | 0.4633 |  |
| 6 | BGE-M3 + BM25 + ColBERT | 0.4600 |  |
| 7 | Upstage Solar (API) + BM25 + ColBERT | 0.4517 |  |
| 8 | OpenAI (API) + BM25 + ColBERT | 0.4500 |  |
| 9 | KoSimCSE + BM25 + ColBERT | 0.4483 |  |
| 10 | Upstage Solar (API) + BM25 + FlashRank | 0.4400 |  |
| 11 | OpenAI (API) + BM25 + FlashRank | 0.3900 |  |
| 12 | Upstage Solar (API) + SPLADE + ColBERT | 0.3633 |  |
| 13 | BGE-M3 + BM25 + FlashRank | 0.3350 |  |
| 14 | OpenAI (API) + SPLADE + FlashRank | 0.3133 |  |
| 15 | BGE-M3 + SPLADE + FlashRank | 0.3067 |  |
| 16 | E5-multilingual + BM25 + FlashRank | 0.3000 |  |
| 17 | E5-multilingual + SPLADE + FlashRank | 0.2983 |  |
| 18 | KoSimCSE + BM25 + FlashRank | 0.2967 |  |
| 19 | Upstage Solar (API) + SPLADE + FlashRank | 0.2900 |  |
| 20 | KoSimCSE + SPLADE + FlashRank | 0.2633 | 최고 대비 −0.2217 |

> 전체 편차: 0.2217 (45.7%) — 조합 간 차이가 크므로 모델 선택이 중요합니다.

#### Answer Relevancy — 질문에 답하는 능력

> 생성된 답변이 질문에 직접적으로 대응하는지 측정합니다.

| 순위 | 조합 | 점수 | 비고 |
|:----:|:-----|:----:|:-----|
| 1 | KoSimCSE + SPLADE + FlashRank | 0.8762 | **최고** |
| 2 | BGE-M3 + SPLADE + ColBERT | 0.8687 |  |
| 3 | E5-multilingual + SPLADE + FlashRank | 0.8675 |  |
| 4 | BGE-M3 + SPLADE + FlashRank | 0.8675 |  |
| 5 | OpenAI (API) + SPLADE + FlashRank | 0.8671 |  |
| 6 | KoSimCSE + SPLADE + ColBERT | 0.8652 |  |
| 7 | E5-multilingual + BM25 + ColBERT | 0.8637 |  |
| 8 | BGE-M3 + BM25 + FlashRank | 0.8628 |  |
| 9 | KoSimCSE + BM25 + ColBERT | 0.8620 |  |
| 10 | OpenAI (API) + BM25 + ColBERT | 0.8600 |  |
| 11 | Upstage Solar (API) + SPLADE + FlashRank | 0.8595 |  |
| 12 | KoSimCSE + BM25 + FlashRank | 0.8590 |  |
| 13 | Upstage Solar (API) + BM25 + FlashRank | 0.8588 |  |
| 14 | Upstage Solar (API) + SPLADE + ColBERT | 0.8548 |  |
| 15 | Upstage Solar (API) + BM25 + ColBERT | 0.8535 |  |
| 16 | OpenAI (API) + SPLADE + ColBERT | 0.8505 |  |
| 17 | E5-multilingual + SPLADE + ColBERT | 0.8502 |  |
| 18 | E5-multilingual + BM25 + FlashRank | 0.8500 |  |
| 19 | BGE-M3 + BM25 + ColBERT | 0.8469 |  |
| 20 | OpenAI (API) + BM25 + FlashRank | 0.8390 | 최고 대비 −0.0372 |

> 전체 편차: 0.0372 (4.2%) — 조합 간 차이가 작아 이 지표만으로는 우열을 가리기 어렵습니다.

---

## 5. 레이턴시(속도) 참고

> **레이턴시는 순위 결정에 반영하지 않습니다.**
> LLM 추론 노이즈가 전략 간 차이를 압도하며, 동일 전략도 실행 시점에 따라 편차가 큽니다.
> CPU-only 환경 수치이므로 GPU 환경과 직접 비교할 수 없습니다.
> 아래는 실행 환경에서의 기준선 참고 데이터입니다.

### BUSINESS

| 조합 | 평균 (s/query) | 중앙값 (s/query) |
|:-----|:--------------:|:---------------:|
| BGE-M3 + BM25 + FlashRank | 33 | 33 |
| Upstage Solar (API) + SPLADE + FlashRank | 33 | 33 |
| Upstage Solar (API) + BM25 + FlashRank | 35 | 35 |
| OpenAI (API) + BM25 + FlashRank | 37 | 36 |
| KoSimCSE + SPLADE + FlashRank | 38 | 38 |
| OpenAI (API) + SPLADE + FlashRank | 39 | 39 |
| KoSimCSE + BM25 + FlashRank | 40 | 39 |
| E5-multilingual + BM25 + FlashRank | 45 | 44 |
| BGE-M3 + SPLADE + FlashRank | 47 | 47 |
| E5-multilingual + SPLADE + FlashRank | 49 | 49 |
| OpenAI (API) + BM25 + ColBERT | 52 | 52 |
| E5-multilingual + SPLADE + ColBERT | 56 | 56 |
| Upstage Solar (API) + BM25 + ColBERT | 77 | 77 |
| OpenAI (API) + SPLADE + ColBERT | 80 | 80 |
| KoSimCSE + BM25 + ColBERT | 83 | 83 |
| BGE-M3 + BM25 + ColBERT | 84 | 84 |
| E5-multilingual + BM25 + ColBERT | 85 | 85 |
| KoSimCSE + SPLADE + ColBERT | 89 | 90 |
| Upstage Solar (API) + SPLADE + ColBERT | 91 | 91 |
| BGE-M3 + SPLADE + ColBERT | 92 | 93 |

---

### GENERAL

| 조합 | 평균 (s/query) | 중앙값 (s/query) |
|:-----|:--------------:|:---------------:|
| Upstage Solar (API) + BM25 + FlashRank | 34 | 33 |
| OpenAI (API) + BM25 + FlashRank | 35 | 34 |
| KoSimCSE + BM25 + FlashRank | 37 | 36 |
| OpenAI (API) + SPLADE + FlashRank | 37 | 36 |
| KoSimCSE + SPLADE + FlashRank | 39 | 38 |
| Upstage Solar (API) + SPLADE + FlashRank | 40 | 40 |
| E5-multilingual + BM25 + FlashRank | 40 | 38 |
| E5-multilingual + SPLADE + FlashRank | 42 | 40 |
| BGE-M3 + BM25 + FlashRank | 44 | 43 |
| BGE-M3 + SPLADE + FlashRank | 44 | 42 |
| Upstage Solar (API) + BM25 + ColBERT | 76 | 77 |
| KoSimCSE + BM25 + ColBERT | 77 | 75 |
| KoSimCSE + SPLADE + ColBERT | 80 | 79 |
| Upstage Solar (API) + SPLADE + ColBERT | 82 | 82 |
| E5-multilingual + BM25 + ColBERT | 82 | 80 |
| OpenAI (API) + BM25 + ColBERT | 85 | 84 |
| OpenAI (API) + SPLADE + ColBERT | 87 | 87 |
| E5-multilingual + SPLADE + ColBERT | 91 | 88 |
| BGE-M3 + BM25 + ColBERT | 94 | 93 |
| BGE-M3 + SPLADE + ColBERT | 95 | 95 |

---

### LEGAL

| 조합 | 평균 (s/query) | 중앙값 (s/query) |
|:-----|:--------------:|:---------------:|
| KoSimCSE + SPLADE + FlashRank | 32 | 32 |
| BGE-M3 + SPLADE + FlashRank | 33 | 33 |
| OpenAI (API) + BM25 + FlashRank | 34 | 31 |
| Upstage Solar (API) + BM25 + FlashRank | 35 | 35 |
| OpenAI (API) + SPLADE + FlashRank | 37 | 37 |
| KoSimCSE + BM25 + FlashRank | 38 | 38 |
| Upstage Solar (API) + SPLADE + FlashRank | 39 | 39 |
| E5-multilingual + BM25 + FlashRank | 42 | 42 |
| E5-multilingual + SPLADE + FlashRank | 44 | 44 |
| BGE-M3 + BM25 + FlashRank | 49 | 47 |
| OpenAI (API) + SPLADE + ColBERT | 66 | 66 |
| BGE-M3 + SPLADE + ColBERT | 80 | 83 |
| OpenAI (API) + BM25 + ColBERT | 80 | 79 |
| Upstage Solar (API) + SPLADE + ColBERT | 81 | 81 |
| Upstage Solar (API) + BM25 + ColBERT | 81 | 81 |
| KoSimCSE + BM25 + ColBERT | 82 | 81 |
| KoSimCSE + SPLADE + ColBERT | 92 | 93 |
| E5-multilingual + BM25 + ColBERT | 94 | 92 |
| BGE-M3 + BM25 + ColBERT | 96 | 95 |
| E5-multilingual + SPLADE + ColBERT | 98 | 97 |

---

### MEDICAL

| 조합 | 평균 (s/query) | 중앙값 (s/query) |
|:-----|:--------------:|:---------------:|
| Upstage Solar (API) + BM25 + FlashRank | 28 | 32 |
| OpenAI (API) + BM25 + FlashRank | 34 | 34 |
| OpenAI (API) + SPLADE + FlashRank | 36 | 36 |
| Upstage Solar (API) + SPLADE + FlashRank | 37 | 40 |
| E5-multilingual + SPLADE + FlashRank | 39 | 39 |
| KoSimCSE + SPLADE + FlashRank | 39 | 39 |
| E5-multilingual + BM25 + FlashRank | 39 | 39 |
| KoSimCSE + BM25 + FlashRank | 40 | 39 |
| BGE-M3 + SPLADE + FlashRank | 41 | 42 |
| BGE-M3 + BM25 + FlashRank | 42 | 41 |
| E5-multilingual + BM25 + ColBERT | 45 | 45 |
| Upstage Solar (API) + BM25 + ColBERT | 60 | 72 |
| Upstage Solar (API) + SPLADE + ColBERT | 66 | 65 |
| OpenAI (API) + SPLADE + ColBERT | 69 | 70 |
| KoSimCSE + SPLADE + ColBERT | 69 | 69 |
| KoSimCSE + BM25 + ColBERT | 70 | 71 |
| OpenAI (API) + BM25 + ColBERT | 71 | 71 |
| BGE-M3 + SPLADE + ColBERT | 76 | 75 |
| BGE-M3 + BM25 + ColBERT | 81 | 81 |
| E5-multilingual + SPLADE + ColBERT | 87 | 86 |

---

### TECHNICAL

| 조합 | 평균 (s/query) | 중앙값 (s/query) |
|:-----|:--------------:|:---------------:|
| OpenAI (API) + SPLADE + FlashRank | 30 | 30 |
| E5-multilingual + SPLADE + FlashRank | 33 | 33 |
| OpenAI (API) + BM25 + FlashRank | 36 | 36 |
| Upstage Solar (API) + BM25 + FlashRank | 36 | 36 |
| KoSimCSE + BM25 + FlashRank | 37 | 37 |
| BGE-M3 + BM25 + FlashRank | 39 | 39 |
| Upstage Solar (API) + SPLADE + FlashRank | 41 | 40 |
| E5-multilingual + BM25 + FlashRank | 44 | 43 |
| KoSimCSE + SPLADE + FlashRank | 44 | 44 |
| BGE-M3 + SPLADE + FlashRank | 46 | 45 |
| BGE-M3 + SPLADE + ColBERT | 48 | 48 |
| OpenAI (API) + SPLADE + ColBERT | 64 | 63 |
| Upstage Solar (API) + SPLADE + ColBERT | 71 | 82 |
| KoSimCSE + BM25 + ColBERT | 72 | 73 |
| BGE-M3 + BM25 + ColBERT | 74 | 74 |
| Upstage Solar (API) + BM25 + ColBERT | 79 | 79 |
| KoSimCSE + SPLADE + ColBERT | 80 | 79 |
| OpenAI (API) + BM25 + ColBERT | 80 | 80 |
| E5-multilingual + SPLADE + ColBERT | 87 | 87 |
| E5-multilingual + BM25 + ColBERT | 89 | 88 |

---

## 6. 모델 유형별 비교

### 6-1. BUSINESS 카테고리

#### Dense 모델 비교 (Sparse 평균)

> 각 Dense 모델의 BM25/SPLADE 결과를 평균하여 순수 Dense 모델 성능을 비교합니다.

| Dense 모델 | 유형 | 평균 복합 점수 | 평균 Recall | 평균 Faithfulness |
|-----------|:----:|:------------:|:----------:|:----------------:|
| E5-multilingual | 로컬 | 0.7943 | 0.7117 | 0.6237 |
| BGE-M3 | 로컬 | 0.7938 | 0.7085 | 0.6246 |
| Upstage Solar (API) | API | 0.7907 | 0.7031 | 0.6005 |
| OpenAI (API) | API | 0.7866 | 0.6831 | 0.6279 |
| KoSimCSE | 로컬 | 0.7827 | 0.6774 | 0.6198 |

> **E5-multilingual**가 평균 복합 점수 0.7943로 Dense 모델 중 1위.
> 최하위(KoSimCSE) 대비 0.0116 (1.5%) 우세.

#### Sparse 모델 비교 (Dense 평균)

> 각 Sparse 모델의 Dense 5종 결과를 평균하여 순수 Sparse 모델 효과를 비교합니다.

| Sparse 모델 | 평균 복합 점수 | 평균 Recall | 평균 Precision |
|------------|:------------:|:----------:|:-------------:|
| BM25 | 0.8038 | 0.7136 | 0.9793 |
| SPLADE | 0.7754 | 0.6799 | 0.9650 |

> **BM25**가 SPLADE 대비 복합 점수 +0.0284 우세합니다.
> 특히 BM25가 Recall에서 +0.0337 우세 — 한국어 형태소 분석(OKt) 기반의 정확한 키워드 매칭이 Recall에 기여합니다.

### 6-2. GENERAL 카테고리

#### Dense 모델 비교 (Sparse 평균)

> 각 Dense 모델의 BM25/SPLADE 결과를 평균하여 순수 Dense 모델 성능을 비교합니다.

| Dense 모델 | 유형 | 평균 복합 점수 | 평균 Recall | 평균 Faithfulness |
|-----------|:----:|:------------:|:----------:|:----------------:|
| Upstage Solar (API) | API | 0.7185 | 0.6707 | 0.5193 |
| OpenAI (API) | API | 0.7173 | 0.6894 | 0.4924 |
| KoSimCSE | 로컬 | 0.7166 | 0.6798 | 0.5094 |
| BGE-M3 | 로컬 | 0.7159 | 0.6779 | 0.5299 |
| E5-multilingual | 로컬 | 0.7144 | 0.6801 | 0.5037 |

> **Upstage Solar (API)**가 평균 복합 점수 0.7185로 Dense 모델 중 1위.
> 최하위(E5-multilingual) 대비 0.0042 (0.6%) 우세.

#### Sparse 모델 비교 (Dense 평균)

> 각 Sparse 모델의 Dense 5종 결과를 평균하여 순수 Sparse 모델 효과를 비교합니다.

| Sparse 모델 | 평균 복합 점수 | 평균 Recall | 평균 Precision |
|------------|:------------:|:----------:|:-------------:|
| SPLADE | 0.7187 | 0.6992 | 0.8553 |
| BM25 | 0.7144 | 0.6599 | 0.8747 |

> BM25와 SPLADE의 평균 복합 점수 차이가 0.0043로 미미합니다.
> 특히 SPLADE가 Recall에서 +0.0393 우세 — 학습된 확장 토큰이 동의어·유사 표현까지 포착합니다.

### 6-3. LEGAL 카테고리

#### Dense 모델 비교 (Sparse 평균)

> 각 Dense 모델의 BM25/SPLADE 결과를 평균하여 순수 Dense 모델 성능을 비교합니다.

| Dense 모델 | 유형 | 평균 복합 점수 | 평균 Recall | 평균 Faithfulness |
|-----------|:----:|:------------:|:----------:|:----------------:|
| KoSimCSE | 로컬 | 0.7891 | 0.6802 | 0.6506 |
| Upstage Solar (API) | API | 0.7888 | 0.6815 | 0.6288 |
| OpenAI (API) | API | 0.7877 | 0.6776 | 0.6345 |
| BGE-M3 | 로컬 | 0.7876 | 0.6841 | 0.6502 |
| E5-multilingual | 로컬 | 0.7873 | 0.6779 | 0.6442 |

> **KoSimCSE**가 평균 복합 점수 0.7891로 Dense 모델 중 1위.
> 최하위(E5-multilingual) 대비 0.0017 (0.2%) 우세.

#### Sparse 모델 비교 (Dense 평균)

> 각 Sparse 모델의 Dense 5종 결과를 평균하여 순수 Sparse 모델 효과를 비교합니다.

| Sparse 모델 | 평균 복합 점수 | 평균 Recall | 평균 Precision |
|------------|:------------:|:----------:|:-------------:|
| BM25 | 0.7968 | 0.6785 | 0.9862 |
| SPLADE | 0.7793 | 0.6820 | 0.9770 |

> **BM25**가 SPLADE 대비 복합 점수 +0.0175 우세합니다.

### 6-4. MEDICAL 카테고리

#### Dense 모델 비교 (Sparse 평균)

> 각 Dense 모델의 BM25/SPLADE 결과를 평균하여 순수 Dense 모델 성능을 비교합니다.

| Dense 모델 | 유형 | 평균 복합 점수 | 평균 Recall | 평균 Faithfulness |
|-----------|:----:|:------------:|:----------:|:----------------:|
| BGE-M3 | 로컬 | 0.7463 | 0.7074 | 0.6382 |
| OpenAI (API) | API | 0.7430 | 0.7060 | 0.6527 |
| E5-multilingual | 로컬 | 0.7394 | 0.6908 | 0.6484 |
| KoSimCSE | 로컬 | 0.7223 | 0.6945 | 0.6063 |
| Upstage Solar (API) | API | 0.7093 | 0.6751 | 0.6313 |

> **BGE-M3**가 평균 복합 점수 0.7463로 Dense 모델 중 1위.
> 최하위(Upstage Solar (API)) 대비 0.0370 (5.0%) 우세.

#### Sparse 모델 비교 (Dense 평균)

> 각 Sparse 모델의 Dense 5종 결과를 평균하여 순수 Sparse 모델 효과를 비교합니다.

| Sparse 모델 | 평균 복합 점수 | 평균 Recall | 평균 Precision |
|------------|:------------:|:----------:|:-------------:|
| BM25 | 0.7529 | 0.7192 | 0.7901 |
| SPLADE | 0.7113 | 0.6703 | 0.7498 |

> **BM25**가 SPLADE 대비 복합 점수 +0.0416 우세합니다.
> 특히 BM25가 Recall에서 +0.0490 우세 — 한국어 형태소 분석(OKt) 기반의 정확한 키워드 매칭이 Recall에 기여합니다.

### 6-5. TECHNICAL 카테고리

#### Dense 모델 비교 (Sparse 평균)

> 각 Dense 모델의 BM25/SPLADE 결과를 평균하여 순수 Dense 모델 성능을 비교합니다.

| Dense 모델 | 유형 | 평균 복합 점수 | 평균 Recall | 평균 Faithfulness |
|-----------|:----:|:------------:|:----------:|:----------------:|
| Upstage Solar (API) | API | 0.4393 | 0.2784 | 0.4875 |
| BGE-M3 | 로컬 | 0.4381 | 0.2586 | 0.4983 |
| OpenAI (API) | API | 0.4367 | 0.2681 | 0.4593 |
| E5-multilingual | 로컬 | 0.4285 | 0.2589 | 0.4736 |
| KoSimCSE | 로컬 | 0.4005 | 0.2185 | 0.4128 |

> **Upstage Solar (API)**가 평균 복합 점수 0.4393로 Dense 모델 중 1위.
> 최하위(KoSimCSE) 대비 0.0388 (8.8%) 우세.

#### Sparse 모델 비교 (Dense 평균)

> 각 Sparse 모델의 Dense 5종 결과를 평균하여 순수 Sparse 모델 효과를 비교합니다.

| Sparse 모델 | 평균 복합 점수 | 평균 Recall | 평균 Precision |
|------------|:------------:|:----------:|:-------------:|
| BM25 | 0.4499 | 0.2783 | 0.4035 |
| SPLADE | 0.4074 | 0.2348 | 0.3747 |

> **BM25**가 SPLADE 대비 복합 점수 +0.0426 우세합니다.
> 특히 BM25가 Recall에서 +0.0435 우세 — 한국어 형태소 분석(OKt) 기반의 정확한 키워드 매칭이 Recall에 기여합니다.

---

## 7. 최종 모델 선정 가이드

### 7-1. BUSINESS 카테고리

#### 용도별 추천

| 사용 상황 | 추천 조합 | 이유 |
|----------|---------|------|
| **품질 최우선** | KoSimCSE + BM25 + ColBERT | 종합 점수 1위 (0.8668) |
| **정보 누락 방지** | BGE-M3 + SPLADE + ColBERT | Recall 최고 (0.7636) |

#### 한 가지만 선택해야 한다면

> **KoSimCSE + BM25 + ColBERT**
>
> 종합 점수 0.8668로 1위이며, 2위 대비 0.0037 (0.4%) 우세합니다.

### 7-2. GENERAL 카테고리

#### 용도별 추천

| 사용 상황 | 추천 조합 | 이유 |
|----------|---------|------|
| **품질 최우선** | BGE-M3 + BM25 + ColBERT | 종합 점수 1위 (0.8561) |
| **정보 누락 방지** | OpenAI (API) + BM25 + ColBERT | Recall 최고 (0.8572) |
| **할루시네이션 방지** | BGE-M3 + SPLADE + ColBERT | Faithfulness 최고 (0.7922) |

#### 한 가지만 선택해야 한다면

> **BGE-M3 + BM25 + ColBERT**
>
> 종합 점수 0.8561로 1위이며, 2위 대비 0.0025 (0.3%) 우세합니다.

### 7-3. LEGAL 카테고리

#### 용도별 추천

| 사용 상황 | 추천 조합 | 이유 |
|----------|---------|------|
| **품질 최우선** | E5-multilingual + SPLADE + ColBERT | 종합 점수 1위 (0.8565) |
| **할루시네이션 방지** | Upstage Solar (API) + SPLADE + ColBERT | Faithfulness 최고 (0.9105) |

#### 한 가지만 선택해야 한다면

> **E5-multilingual + SPLADE + ColBERT**
>
> 종합 점수 0.8565로 1위이며, 2위 대비 0.0006 (0.1%) 우세합니다.

### 7-4. MEDICAL 카테고리

#### 용도별 추천

| 사용 상황 | 추천 조합 | 이유 |
|----------|---------|------|
| **품질 최우선** | E5-multilingual + BM25 + ColBERT | 종합 점수 1위 (0.9166) |
| **정보 누락 방지** | BGE-M3 + SPLADE + ColBERT | Recall 최고 (0.9155) |
| **할루시네이션 방지** | OpenAI (API) + SPLADE + ColBERT | Faithfulness 최고 (0.8734) |

#### 한 가지만 선택해야 한다면

> **E5-multilingual + BM25 + ColBERT**
>
> 종합 점수 0.9166로 1위이며, 2위 대비 0.0020 (0.2%) 우세합니다.

### 7-5. TECHNICAL 카테고리

#### 용도별 추천

| 사용 상황 | 추천 조합 | 이유 |
|----------|---------|------|
| **품질 최우선** | E5-multilingual + BM25 + ColBERT | 종합 점수 1위 (0.5229) |
| **정보 누락 방지** | Upstage Solar (API) + BM25 + ColBERT | Recall 최고 (0.3499) |

#### 한 가지만 선택해야 한다면

> **E5-multilingual + BM25 + ColBERT**
>
> 종합 점수 0.5229로 1위이며, 2위 대비 0.0035 (0.7%) 우세합니다.

### 7-6. 향후 과제

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

**BUSINESS** (질의 77개, 20개 조합)

| Rank | Strategy | Recall | Precision | Faithfulness | Relevancy | Composite | Avg Lat (s) | Med Lat (s) |
|:----:|:---------|:------:|:---------:|:------------:|:---------:|:---------:|:-----------:|:-----------:|
| 1 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.7577 | 0.9946 | 0.8601 | 0.8746 | **0.8668** | 82.6 | 82.8 |
| 2 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.7629 | 0.9870 | 0.8531 | 0.8623 | 0.8631 | 79.7 | 80.5 |
| 3 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.7636 | 0.9848 | 0.8410 | 0.8810 | 0.8630 | 92.1 | 93.0 |
| 4 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.7559 | 0.9870 | 0.8518 | 0.8776 | 0.8627 | 90.7 | 90.9 |
| 5 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.7568 | 0.9848 | 0.8415 | 0.8845 | 0.8613 | 55.6 | 55.9 |
| 6 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.7541 | 0.9870 | 0.8345 | 0.8835 | 0.8595 | 51.8 | 51.7 |
| 7 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.7571 | 0.9848 | 0.8308 | 0.8784 | 0.8583 | 83.8 | 84.4 |
| 8 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.7392 | 0.9978 | 0.8411 | 0.8799 | 0.8583 | 89.2 | 90.2 |
| 9 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.7566 | 0.9848 | 0.8300 | 0.8736 | 0.8573 | 85.4 | 85.5 |
| 10 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.7483 | 0.9870 | 0.8227 | 0.8730 | 0.8535 | 77.2 | 77.3 |
| 11 | `FlashRank Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.7135 | 0.9632 | 0.4582 | 0.8751 | 0.7616 | 44.9 | 44.1 |
| 12 | `FlashRank Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.6727 | 0.9740 | 0.4955 | 0.8719 | 0.7575 | 32.6 | 32.6 |
| 13 | `FlashRank Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.6364 | 0.9740 | 0.4943 | 0.8766 | 0.7453 | 36.5 | 36.3 |
| 14 | `FlashRank Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.6977 | 0.9762 | 0.3779 | 0.8775 | 0.7443 | 35.4 | 35.3 |
| 15 | `FlashRank Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.6419 | 0.9675 | 0.4536 | 0.8565 | 0.7341 | 39.8 | 39.2 |
| 16 | `FlashRank Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.6103 | 0.9665 | 0.3497 | 0.8574 | 0.7021 | 33.4 | 32.9 |
| 17 | `FlashRank Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.6199 | 0.9232 | 0.3649 | 0.8660 | 0.6968 | 48.5 | 48.5 |
| 18 | `FlashRank Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.6407 | 0.9361 | 0.3312 | 0.8336 | 0.6964 | 46.9 | 47.3 |
| 19 | `FlashRank Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.5789 | 0.9405 | 0.3295 | 0.8538 | 0.6787 | 39.2 | 39.0 |
| 20 | `FlashRank Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.5707 | 0.9426 | 0.3245 | 0.8272 | 0.6715 | 38.0 | 38.0 |

**GENERAL** (질의 100개, 20개 조합)

| Rank | Strategy | Recall | Precision | Faithfulness | Relevancy | Composite | Avg Lat (s) | Med Lat (s) |
|:----:|:---------|:------:|:---------:|:------------:|:---------:|:---------:|:-----------:|:-----------:|
| 1 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.8552 | 0.9183 | 0.7557 | 0.8675 | **0.8561** | 93.9 | 93.5 |
| 2 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.8299 | 0.9067 | 0.7922 | 0.8846 | 0.8536 | 95.4 | 95.1 |
| 3 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.8572 | 0.9233 | 0.7067 | 0.8774 | 0.8500 | 85.1 | 83.9 |
| 4 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.8307 | 0.9233 | 0.7548 | 0.8681 | 0.8489 | 81.9 | 80.4 |
| 5 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.8251 | 0.9258 | 0.7578 | 0.8652 | 0.8479 | 76.3 | 77.0 |
| 6 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.8246 | 0.9150 | 0.7725 | 0.8666 | 0.8476 | 81.6 | 82.0 |
| 7 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.8284 | 0.9133 | 0.7672 | 0.8683 | 0.8476 | 91.2 | 87.8 |
| 8 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.8144 | 0.9142 | 0.7685 | 0.8690 | 0.8433 | 80.2 | 78.7 |
| 9 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.8180 | 0.9275 | 0.7432 | 0.8665 | 0.8432 | 77.4 | 75.2 |
| 10 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.8228 | 0.9092 | 0.7522 | 0.8645 | 0.8409 | 87.2 | 87.1 |
| 11 | `FlashRank Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.5662 | 0.8333 | 0.2628 | 0.6685 | 0.6010 | 40.3 | 39.7 |
| 12 | `FlashRank Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.5897 | 0.8000 | 0.2615 | 0.6615 | 0.5979 | 39.1 | 37.7 |
| 13 | `FlashRank Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.5031 | 0.8258 | 0.3000 | 0.7499 | 0.5963 | 34.6 | 34.0 |
| 14 | `FlashRank Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.5772 | 0.7850 | 0.2360 | 0.7074 | 0.5908 | 41.9 | 39.8 |
| 15 | `FlashRank Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.5643 | 0.7825 | 0.2600 | 0.6549 | 0.5825 | 43.9 | 41.6 |
| 16 | `FlashRank Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.4969 | 0.8292 | 0.2646 | 0.7089 | 0.5819 | 36.8 | 35.9 |
| 17 | `FlashRank Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.5745 | 0.7942 | 0.2107 | 0.6689 | 0.5818 | 37.0 | 36.4 |
| 18 | `FlashRank Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.4670 | 0.8333 | 0.2843 | 0.7152 | 0.5776 | 34.2 | 32.6 |
| 19 | `FlashRank Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.4622 | 0.8242 | 0.3115 | 0.6679 | 0.5715 | 43.7 | 42.8 |
| 20 | `FlashRank Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.4839 | 0.8167 | 0.2569 | 0.6955 | 0.5701 | 40.5 | 38.5 |

**LEGAL** (질의 37개, 20개 조합)

| Rank | Strategy | Recall | Precision | Faithfulness | Relevancy | Composite | Avg Lat (s) | Med Lat (s) |
|:----:|:---------|:------:|:---------:|:------------:|:---------:|:---------:|:-----------:|:-----------:|
| 1 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.7310 | 0.9842 | 0.8685 | 0.8779 | **0.8565** | 97.6 | 97.5 |
| 2 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.7069 | 0.9955 | 0.9105 | 0.8516 | 0.8559 | 81.1 | 80.9 |
| 3 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.7159 | 0.9955 | 0.8987 | 0.8420 | 0.8553 | 79.7 | 82.7 |
| 4 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.6877 | 0.9842 | 0.8845 | 0.8657 | 0.8427 | 80.1 | 79.4 |
| 5 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.6850 | 0.9955 | 0.8649 | 0.8682 | 0.8416 | 81.5 | 81.0 |
| 6 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.6935 | 0.9842 | 0.8571 | 0.8753 | 0.8407 | 81.3 | 81.4 |
| 7 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.7032 | 0.9842 | 0.8462 | 0.8486 | 0.8379 | 94.4 | 92.3 |
| 8 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.6954 | 0.9842 | 0.8466 | 0.8659 | 0.8379 | 66.3 | 65.8 |
| 9 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.7126 | 0.9955 | 0.8067 | 0.8499 | 0.8369 | 91.6 | 92.9 |
| 10 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.6698 | 0.9730 | 0.8131 | 0.8694 | 0.8194 | 96.3 | 95.4 |
| 11 | `FlashRank Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.7122 | 0.9887 | 0.4583 | 0.8836 | 0.7701 | 35.0 | 34.9 |
| 12 | `FlashRank Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.6789 | 0.9820 | 0.5540 | 0.8327 | 0.7679 | 38.0 | 37.6 |
| 13 | `FlashRank Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.6579 | 0.9955 | 0.4995 | 0.8290 | 0.7532 | 34.1 | 31.4 |
| 14 | `FlashRank Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.6717 | 0.9910 | 0.4587 | 0.8294 | 0.7485 | 48.8 | 46.8 |
| 15 | `FlashRank Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.6250 | 0.9842 | 0.5193 | 0.8578 | 0.7465 | 42.2 | 42.3 |
| 16 | `FlashRank Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.6789 | 0.9437 | 0.4304 | 0.8017 | 0.7271 | 32.9 | 32.7 |
| 17 | `FlashRank Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.6694 | 0.9797 | 0.3076 | 0.8477 | 0.7169 | 36.8 | 36.8 |
| 18 | `FlashRank Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.6443 | 0.9707 | 0.3769 | 0.7852 | 0.7099 | 32.4 | 32.3 |
| 19 | `FlashRank Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.6525 | 0.9572 | 0.3426 | 0.8292 | 0.7084 | 44.3 | 44.0 |
| 20 | `FlashRank Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.6134 | 0.9640 | 0.2895 | 0.8438 | 0.6884 | 39.1 | 38.8 |

**MEDICAL** (질의 77개, 20개 조합)

| Rank | Strategy | Recall | Precision | Faithfulness | Relevancy | Composite | Avg Lat (s) | Med Lat (s) |
|:----:|:---------|:------:|:---------:|:------------:|:---------:|:---------:|:-----------:|:-----------:|
| 1 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.9111 | 0.9762 | 0.8708 | 0.8714 | **0.9166** | 44.9 | 45.0 |
| 2 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.9111 | 0.9697 | 0.8584 | 0.8878 | 0.9146 | 81.3 | 80.7 |
| 3 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.9063 | 0.9773 | 0.8662 | 0.8730 | 0.9146 | 71.0 | 71.3 |
| 4 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.9155 | 0.9697 | 0.8701 | 0.8538 | 0.9134 | 76.2 | 75.3 |
| 5 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.9104 | 0.9567 | 0.8734 | 0.8748 | 0.9115 | 69.1 | 69.6 |
| 6 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.9125 | 0.9740 | 0.8614 | 0.8470 | 0.9109 | 87.3 | 86.3 |
| 7 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.9032 | 0.9675 | 0.8686 | 0.8663 | 0.9100 | 65.8 | 64.8 |
| 8 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.9060 | 0.9610 | 0.8514 | 0.8589 | 0.9045 | 70.2 | 70.8 |
| 9 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.8661 | 0.9372 | 0.8414 | 0.8609 | 0.8817 | 69.2 | 69.1 |
| 10 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.8445 | 0.8074 | 0.7781 | 0.8814 | 0.8256 | 59.5 | 71.9 |
| 11 | `FlashRank Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.5791 | 0.6872 | 0.5006 | 0.8857 | 0.6418 | 42.5 | 40.7 |
| 12 | `FlashRank Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.5849 | 0.6721 | 0.4961 | 0.8757 | 0.6369 | 34.0 | 34.0 |
| 13 | `FlashRank Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.5087 | 0.6613 | 0.4905 | 0.8747 | 0.6057 | 39.4 | 39.2 |
| 14 | `FlashRank Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.5033 | 0.5823 | 0.5152 | 0.8768 | 0.5854 | 28.3 | 32.2 |
| 15 | `FlashRank Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.5375 | 0.6061 | 0.4106 | 0.8748 | 0.5833 | 40.3 | 39.1 |
| 16 | `FlashRank Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.4311 | 0.5649 | 0.3709 | 0.8661 | 0.5244 | 38.7 | 38.6 |
| 17 | `FlashRank Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.4685 | 0.5292 | 0.3219 | 0.8840 | 0.5197 | 38.8 | 38.7 |
| 18 | `FlashRank Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.4494 | 0.5108 | 0.3633 | 0.8875 | 0.5163 | 37.5 | 40.4 |
| 19 | `FlashRank Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.4237 | 0.5703 | 0.3239 | 0.8765 | 0.5156 | 41.4 | 41.6 |
| 20 | `FlashRank Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.4225 | 0.5173 | 0.3752 | 0.8741 | 0.5092 | 36.2 | 36.3 |

**TECHNICAL** (질의 50개, 20개 조합)

| Rank | Strategy | Recall | Precision | Faithfulness | Relevancy | Composite | Avg Lat (s) | Med Lat (s) |
|:----:|:---------|:------:|:---------:|:------------:|:---------:|:---------:|:-----------:|:-----------:|
| 1 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.3429 | 0.4633 | 0.6715 | 0.8637 | **0.5229** | 89.0 | 88.0 |
| 2 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.3499 | 0.4517 | 0.6671 | 0.8535 | 0.5194 | 78.7 | 78.5 |
| 3 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.3468 | 0.4850 | 0.6220 | 0.8505 | 0.5189 | 64.1 | 63.1 |
| 4 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.3160 | 0.4817 | 0.6557 | 0.8687 | 0.5166 | 48.1 | 47.7 |
| 5 | `ColBERT Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.3421 | 0.4600 | 0.6490 | 0.8469 | 0.5146 | 73.8 | 74.0 |
| 6 | `ColBERT Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.3207 | 0.4650 | 0.6580 | 0.8502 | 0.5109 | 86.6 | 86.6 |
| 7 | `ColBERT Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.3332 | 0.4500 | 0.6303 | 0.8600 | 0.5067 | 79.9 | 79.6 |
| 8 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.2994 | 0.4483 | 0.6570 | 0.8620 | 0.5000 | 72.2 | 73.1 |
| 9 | `ColBERT Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.3124 | 0.3633 | 0.6612 | 0.8548 | 0.4788 | 70.7 | 81.9 |
| 10 | `ColBERT Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.2724 | 0.4800 | 0.4893 | 0.8652 | 0.4670 | 79.6 | 78.8 |
| 11 | `FlashRank Rerank (Contextual Retrieval (DS(embedding-query+korean_bm25)))` | 0.2940 | 0.4400 | 0.4242 | 0.8588 | 0.4486 | 36.3 | 36.3 |
| 12 | `FlashRank Rerank (Contextual Retrieval (DS(text-embedding-3-large+korean_bm25)))` | 0.2357 | 0.3900 | 0.3602 | 0.8390 | 0.3974 | 35.7 | 35.8 |
| 13 | `FlashRank Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))` | 0.1939 | 0.3350 | 0.4521 | 0.8628 | 0.3882 | 39.3 | 39.3 |
| 14 | `FlashRank Rerank (Contextual Retrieval (DS(multilingual-e5-large+korean_bm25)))` | 0.2222 | 0.3000 | 0.2978 | 0.8500 | 0.3548 | 43.5 | 43.2 |
| 15 | `FlashRank Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+korean_bm25)))` | 0.1693 | 0.2967 | 0.3472 | 0.8590 | 0.3466 | 36.6 | 36.7 |
| 16 | `FlashRank Rerank (Contextual Retrieval (DS(bge-m3+splade)))` | 0.1824 | 0.3067 | 0.2363 | 0.8675 | 0.3332 | 46.0 | 45.3 |
| 17 | `FlashRank Rerank (Contextual Retrieval (DS(multilingual-e5-large+splade)))` | 0.1497 | 0.2983 | 0.2669 | 0.8675 | 0.3254 | 33.3 | 33.2 |
| 18 | `FlashRank Rerank (Contextual Retrieval (DS(text-embedding-3-large+splade)))` | 0.1568 | 0.3133 | 0.2246 | 0.8671 | 0.3239 | 30.2 | 30.1 |
| 19 | `FlashRank Rerank (Contextual Retrieval (DS(embedding-query+splade)))` | 0.1575 | 0.2900 | 0.1975 | 0.8595 | 0.3105 | 40.6 | 40.3 |
| 20 | `FlashRank Rerank (Contextual Retrieval (DS(KoSimCSE-roberta-multitask+splade)))` | 0.1329 | 0.2633 | 0.1577 | 0.8762 | 0.2885 | 43.7 | 43.6 |

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
