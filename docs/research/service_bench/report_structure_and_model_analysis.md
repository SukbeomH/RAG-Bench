# 보고서 구성 및 신규 모델 분석

> 작성일: 2026-02-24
> 참조: W&B Korean LLM Leaderboard v3, W&B LLM Evaluation Framework, snowflake-arctic-embed-l-v2.0-ko

---

## Part 1: W&B Korean LLM Leaderboard v3 — 보고서 구성 및 논리 전개

### 1.1 리포트 개요

**공식명**: Horangi: W&B Korean LLM Leaderboard 3
- URL: https://wandb.ai/wandb-korea/llm-leaderboard3/reports/Horangi-W-B-Korean-LLM-Leaderboard-3--Vmlldzo5NTM4MjU0
- 목적: "최신 LLM 모델의 한국어 능력을 언어이해능력과 응용능력, AI Alignment의 넓은 관점에서 평가"
- 평가 대상: 40+ 모델 (OpenAI, Anthropic API + 한국/해외 오픈소스)

### 1.2 평가 섹션 구조 (논리 전개 순서)

```
[1] 평가 개요
    └─ 평가 목적, 사용 데이터셋, 실행 환경

[2] 종합 성능 순위
    └─ 전체 평균 점수 기반 리더보드 (높은 것에서 낮은 것 순)

[3] 언어 이해 능력 (Language Understanding)
    ├─ HAERAE_BENCH_V1 (한국 문화/지식)
    ├─ KMMLU (한국 지식 이해)
    └─ KoBBQ (편견 측정)

[4] 응용 능력 (Applied Capabilities)
    ├─ 정보 검색 (Information Retrieval)
    ├─ 의미 해석 (Semantic Interpretation)
    └─ 출력 제어 (Output Control)

[5] AI Alignment (안전성 평가)
    ├─ 제어성 (Controllability)
    ├─ 유해성 (Toxicity) — Korean Hate Speech 데이터
    └─ 편견 (Bias) — KoBBQ, AI HUB 윤리검증 데이터

[6] 모델별 강점/약점 심층 분석
    └─ Tables 기능으로 문항별 1:1 비교

[7] 종합 결론
    └─ 모델 선정 가이드 (사용 목적별 추천)
```

### 1.3 평가 방법론

**이중 평가 방식**:
```
최종 점수 = (Zero-shot 평가 + Few-shot 평가) / 2
```
- Zero-shot: 프롬프트 없이 직접 평가 → 모델의 고유 능력 측정
- Few-shot: 예시 제공 후 평가 → 학습 적응력 측정

**추론 환경**:
- vLLM 기반 고속 추론
- Chat template으로 모델별 최적화된 프롬프트 관리

**데이터셋 구성**:
| 데이터셋 | 카테고리 | 특성 |
|---------|---------|------|
| HAERAE_BENCH_V1 | 언어이해 | 한국 문화/상식 |
| KMMLU | 언어이해 | 한국 지식 멀티태스크 |
| KoBBQ | AI Alignment | 한국어 편견 측정 |
| Korean Hate Speech | AI Alignment | 유해 콘텐츠 판별 |
| AI HUB 텍스트윤리검증 | AI Alignment | 윤리성 평가 |

### 1.4 시각화 방식

**W&B 리포트에서 사용한 시각화 패턴**:

1. **종합 리더보드 테이블**
   - 행: 모델명, 열: 카테고리별 점수 + 평균
   - 정렬: 평균 점수 내림차순
   - 강조: 각 열 1위 색상 표시

2. **인터랙티브 모델 비교 차트**
   - 사용자가 비교할 모델 선택
   - 레이더(Spider) 차트: 카테고리별 강점/약점 한눈에 확인

3. **문항별 심층 분석 (Tables 기능)**
   - 종합 점수가 아닌 개별 문항 단위 비교
   - "왜 이 모델이 이 점수인가"를 설명하는 증거

4. **카테고리별 Bar Chart**
   - 언어이해 / 응용능력 / Alignment 각각 별도 시각화
   - 동일 카테고리 내 모델 간 성능 격차 명확화

### 1.5 결론 도출 방식

```
데이터 → 패턴 → 가이드
─────────────────────────
종합 점수 + 카테고리별 점수
        ↓
강점/약점 패턴 도출
("A 모델은 언어이해 강하나 Alignment 약함")
        ↓
사용 목적별 추천
("안전성 중요한 서비스 → B 모델 / 지식 검색 → A 모델")
```

---

## Part 2: W&B LLM Evaluation 프레임워크 원칙

### 2.1 평가 설계 원칙 (Evaluation-Driven Development)

W&B의 LLM 평가 접근법:

1. **다차원 평가**: 단일 지표가 아닌 **복수의 독립적 지표** 조합
   - 정확도 (Accuracy)
   - 지연시간 (Latency)
   - 비용 (Cost)
   - 사용자 경험 (User Experience)

2. **전문가 분리 평가**:
   - NLP Engineers: 의미론적/구문론적 결함 평가
   - Subject Matter Experts: 도메인별 출력 품질 평가

3. **자동 평가 + 인간 주석 정렬**:
   - LLM-as-Judge 자동 평가
   - 인간 주석과의 상관관계 검증으로 자동 평가 신뢰도 확보

4. **점진적 평가 파이프라인**:
   ```
   Pass 1 (레이턴시 전수 평가)
       ↓ 상위 N개 선별
   Pass 2 (품질 심층 평가)
       ↓ 결과 집계
   최종 추천
   ```

### 2.2 벤치마크 설계 원칙

- **표준 데이터셋 우선**: 커뮤니티 검증된 데이터셋 사용으로 재현성 보장
- **다양한 쿼리 유형**: 단순 사실 / 다중 홉 / 추론 필요 쿼리 혼합
- **도메인 분리 평가**: 전체 평균만으로는 도메인별 강약점 파악 불가 → 도메인별 분리 필수

### 2.3 우리 프로젝트에 적용할 보고서 구성

W&B Horangi v3 구조를 RAG Retrieval 벤치마크에 맞게 적용:

```
[Section 1] 평가 개요
    ├─ 고정 파이프라인: ColBERT + Contextual 설명
    ├─ 비교 변수: Dense × Sparse 6개 조합
    ├─ 평가 데이터셋: 5개 카테고리 × N 데이터셋
    └─ 평가 지표: NDCG@10 (주), Context Recall, Precision, Faithfulness

[Section 2] 종합 성능 리더보드
    └─ 카테고리별 평균 NDCG@10 × 조합별 → 정렬된 테이블

[Section 3] 카테고리별 상세 비교
    ├─ GENERAL: MIRACL + Ko-StrategyQA + Belebele + MrTiDy
    ├─ LEGAL: markers_bm (law)
    ├─ BUSINESS: markers_bm (finance+public+commerce)
    ├─ MEDICAL: publichealth-qa
    └─ TECHNICAL: 사용자 문서

[Section 4] 조합별 강점/약점 분석
    ├─ 레이더 차트: 조합별 카테고리 점수 프로파일
    ├─ 히트맵: 조합 × 카테고리 성능 격자
    └─ "이 조합은 왜 이 카테고리에서 강한가" 텍스트 해석

[Section 5] 중복 결과 압축
    └─ 점수 차 5% 이내 조합 → 동점 그룹으로 통합

[Section 6] 최종 선정 보고서
    ├─ 카테고리별 1위 조합 + 선정 이유
    ├─ 공통 강자 조합 (여러 카테고리에서 상위)
    └─ "모르면 이걸 써라" 기본 추천 1개
```

---

## Part 3: snowflake-arctic-embed-l-v2.0-ko — 모델 분석

### 3.1 기본 정보

| 항목 | 내용 |
|------|------|
| **모델명** | dragonkue/snowflake-arctic-embed-l-v2.0-ko |
| **베이스 모델** | Snowflake/snowflake-arctic-embed-l-v2.0 |
| **백본** | bge-m3-retromae (BGE-M3와 동일한 백본!) |
| **파라미터** | 0.6B (600M) |
| **임베딩 차원** | 1024 (256으로 압축 가능) |
| **최대 시퀀스** | 8192 토큰 |
| **한계** | 1300 토큰 이상 긴 문서에서는 다른 모델 권장 |

### 3.2 한국어 파인튜닝 상세

**학습 방법**: CachedGISTEmbedLoss (교사 모델 기반 대조 학습)

**학습 데이터 (AI Hub)**:
| 데이터셋 | 도메인 |
|---------|--------|
| 행정 문서 대상 기계 독해 | 공공/행정 |
| 기계 독해 | 일반 |
| 뉴스 기사 기계독해 | 뉴스/미디어 |
| 도서 자료 기계독해 | 학술/일반 |
| 숫자 연산 기계독해 | 수치/수학 |
| **금융 법률 문서 기계독해** | **금융 + 법률** |

> 학습 데이터의 금융+법률 도메인은 우리 LEGAL/BUSINESS 카테고리와 직접 매핑됨

### 3.3 한국어 Retrieval 벤치마크 성능 (NDCG@10)

> 이 벤치마크는 사용자가 제공한 6개 데이터셋과 **완벽히 일치**합니다.

| 모델 | **평균** | MrTiDy | MIRACL | XPQA | Belebele | PublicHealth | AutoRAG | Ko-StrategyQA |
|------|---------|--------|--------|------|----------|-------------|---------|----------------|
| **snowflake-ko** ⭐ | **0.7404** | 0.5712 | 0.6685 | **0.4436** | **0.9518** | 0.8337 | **0.9093** | *0.8050* |
| **BGE-m3-ko** | *0.7300* | 0.6099 | 0.6833 | 0.3813 | *0.9503* | 0.8155 | *0.8738* | 0.7959 |
| KURE-v1 | 0.7277 | 0.5909 | 0.6816 | 0.3816 | 0.9502 | 0.8193 | 0.8708 | 0.7999 |
| **BAAI/bge-m3** | 0.7242 | **0.6471** | *0.7015* | 0.3608 | 0.9316 | 0.8041 | 0.8301 | 0.7941 |
| snowflake-l-v2.0 | 0.7241 | 0.5907 | 0.6608 | *0.4302* | 0.9271 | 0.8168 | 0.8386 | **0.8046** |

> **AutoRAGRetrieval**: finance, public, medical, legal, commerce PDF → markers_bm과 동일 도메인
> **PublicHealthQA**: publichealth-qa와 동일 데이터셋
> **MrTiDy, MIRACL, Belebele, Ko-StrategyQA**: 사용자 제공 데이터셋과 동일

### 3.4 데이터셋-카테고리 매핑 (완벽한 정렬 확인)

| 우리 카테고리 | 사용 데이터셋 | snowflake-ko 평가 데이터셋 |
|-------------|------------|------------------------|
| GENERAL | MIRACL, MrTiDy, Belebele, Ko-StrategyQA | ✅ MIRACLRetrieval, MrTidyRetrieval, BelebeleRetrieval, Ko-StrategyQA |
| LEGAL | markers_bm (law) | ✅ AutoRAGRetrieval (legal subset) |
| BUSINESS | markers_bm (finance+public+commerce) | ✅ AutoRAGRetrieval (finance+public+commerce subsets) |
| MEDICAL | publichealth-qa | ✅ PublicHealthQARetrieval |
| TECHNICAL | 사용자 문서 | ❌ (없음) |

**결론**: 우리 벤치마크 데이터셋 선택이 현재 한국어 retrieval 평가 표준과 완벽히 일치함

### 3.5 snowflake-ko vs. bge-m3 강/약점

| 카테고리 | snowflake-ko | BAAI/bge-m3 | 우위 |
|---------|-------------|------------|------|
| 전체 평균 | **0.7404** | 0.7242 | snowflake-ko (+2.2%) |
| MrTiDy (일반 위키) | 0.5712 | **0.6471** | bge-m3 (+13.3%) |
| MIRACL (다국어 위키) | 0.6685 | **0.7015** | bge-m3 (+4.9%) |
| Belebele (MRC) | **0.9518** | 0.9316 | snowflake-ko (+2.2%) |
| PublicHealth (의료) | **0.8337** | 0.8041 | snowflake-ko (+3.7%) |
| AutoRAG (실무 문서) | **0.9093** | 0.8301 | snowflake-ko (+9.5%) |
| Ko-StrategyQA (Multi-hop) | **0.8050** | 0.7941 | snowflake-ko (+1.4%) |

**패턴**:
- **snowflake-ko 강점**: 실무 문서(AutoRAG), 의료 FAQ, MRC → **LEGAL/BUSINESS/MEDICAL** 카테고리
- **bge-m3 강점**: 대용량 Wikipedia 검색(MrTiDy, MIRACL) → **GENERAL** 카테고리 (특히 대규모)

---

## Part 4: 계획 업데이트 사항

### 4.1 Dense 모델 목록 업데이트

기존 3종 → **4종으로 확대** (여전히 ≤10 제약 충족):

| # | 모델 키 | 실제 모델명 | 특성 |
|---|--------|-----------|------|
| 1 | `kosimcse` | snunlp/KR-ELECTRA-discriminator 기반 | 한국어 특화 |
| 2 | `e5` | intfloat/multilingual-e5-large | 다국어 경량 |
| 3 | `bge-m3` | BAAI/bge-m3 | 3중 검색, 대규모 강점 |
| **4** | **`snowflake-ko`** | **dragonkue/snowflake-arctic-embed-l-v2.0-ko** | **실무문서/의료 SOTA** |

### 4.2 업데이트된 조합 수

```
4 HF Dense × 2 Sparse = 8 기본 조합  (≤10 제약 충족 ✅)
+ API 2종 포함 시: 6 × 2 = 12 → API는 선택적으로 2개만 추가 → 최대 10개
```

### 4.3 업데이트된 카테고리별 예상 강자

| 카테고리 | 예상 1위 조합 | 근거 |
|---------|------------|------|
| GENERAL | `bge-m3 + splade` | MIRACL(0.7015), MrTiDy(0.6471) 최강 |
| LEGAL | `snowflake-ko + korean_bm25` | AutoRAG 법률 SOTA(0.9093), BM25 전문 용어 강점 |
| BUSINESS | `snowflake-ko + korean_bm25` | AutoRAG 금융/공공 SOTA, 금융법률 학습 데이터 |
| MEDICAL | `snowflake-ko + splade` | PublicHealth 최강(0.8337), SPLADE 의미 확장 |
| TECHNICAL | `e5 + korean_bm25` | 기술 용어 정확 매칭 (가설) |

### 4.4 보고서 섹션 구조 (W&B Horangi 패턴 적용)

최종 선정 보고서는 다음 순서로 구성:

```
1. 평가 개요 (고정 파이프라인 + 데이터셋 + 방법론)
2. 종합 리더보드 (카테고리별 평균 → 정렬)
3. 카테고리별 상세 비교 (GENERAL/LEGAL/BUSINESS/MEDICAL/TECHNICAL)
4. 조합별 강점/약점 분석 (레이더 + 히트맵)
5. 중복 압축 (동점 그룹 통합)
6. 최종 선정 가이드 (카테고리별 추천 + 공통 추천)
```

---

## 참고 자료

- [Horangi: W&B Korean LLM Leaderboard 3](https://wandb.ai/wandb-korea/llm-leaderboard3/reports/Horangi-W-B-Korean-LLM-Leaderboard-3--Vmlldzo5NTM4MjU0)
- [W&B LLM 평가 방법론 (metrics, frameworks, best practices)](https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluations-Metrics-frameworks-and-best-practices--VmlldzoxMTMxNjQ4NA)
- [dragonkue/snowflake-arctic-embed-l-v2.0-ko · HuggingFace](https://huggingface.co/dragonkue/snowflake-arctic-embed-l-v2.0-ko)
- [Arctic-Embed 2.0 논문 (arXiv:2412.04506)](https://arxiv.org/html/2412.04506v2)
- [Snowflake Arctic Embed 2.0 공식 블로그](https://www.snowflake.com/en/engineering-blog/snowflake-arctic-embed-2-multilingual/)
- [Open Ko-LLM Leaderboard (Upstage)](https://huggingface.co/blog/leaderboard-upstage)
- [W&B Evaluation 공식 사이트](https://wandb.ai/site/evaluations/)
