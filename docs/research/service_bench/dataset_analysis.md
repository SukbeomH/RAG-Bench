# 벤치마크 데이터셋 분석 보고서

> 작성일: 2026-02-24
> 목적: 6개 HuggingFace 데이터셋의 카테고리 분석 및 PLAN_SERVICE_BENCH.md 반영

---

## 데이터셋 총람

| # | 데이터셋 | 언어 | 도메인 | 크기 (corpus) | 한국어 |
|---|---------|------|--------|-------------|--------|
| 1 | taeminlee/Ko-StrategyQA | 한국어 | 일반 백과사전 (Multi-hop) | 27.8k docs | ✅ 전용 |
| 2 | yjoonjang/markers_bm | 한국어 | 금융+공공+법률+상업 (실무 문서) | 720 docs | ✅ 전용 |
| 3 | miracl/miracl | 18개 언어 | 일반 (Wikipedia 기반) | 1.5M (ko) | ✅ ko subset |
| 4 | xhluca/publichealth-qa | 8개 언어 | 의료/공중보건 FAQ | 77 QA (ko) | ✅ ko subset |
| 5 | facebook/belebele | 99개 언어 | 일반 MRC (짧은 지문) | ~1,400 (ko) | ✅ kor_Hang |
| 6 | mteb/mrtidy | 11개 언어 | 일반 백과사전 (대규모) | 1.5M (ko) | ✅ ko subset |

---

## 데이터셋 상세 분석

---

### 1. Ko-StrategyQA
**URL**: https://huggingface.co/datasets/taeminlee/Ko-StrategyQA

| 항목 | 내용 |
|------|------|
| **원본** | NomaDamas/Ko-StrategyQA → BeIR 포맷 변환 |
| **언어** | 한국어 100% |
| **형식** | BeIR (corpus + queries + qrels) |
| **Corpus** | 27,800 docs (Wikipedia 기반 다양한 주제) |
| **Queries** | 8,500개 (Train 4.38k + Dev 1.15k) |
| **QA 유형** | **Multi-hop** 추론 (여러 문서를 연결해야 답변 가능) |
| **주제** | Julius Caesar, Mount Fuji, George Washington 등 역사/과학/문화/지리 혼합 |
| **MTEB 호환** | ✅ |

**카테고리 분석**:
- 일반 지식 기반 백과사전형 문서
- 단순 키워드 매칭이 아닌 **다중 홉 추론**이 필요 → 검색 모델의 의미론적 이해 능력 평가
- 한국어 전용 → 한국어 특화 Dense 모델(KoSimCSE) 유리 여부 검증 가능

**→ 카테고리 배정: `GENERAL` (한국어 위키 Multi-hop)**

---

### 2. markers_bm
**URL**: https://huggingface.co/datasets/yjoonjang/markers_bm

| 항목 | 내용 |
|------|------|
| **원본** | 한국 실제 PDF 문서에서 추출 |
| **언어** | 한국어 100% |
| **형식** | BeIR (corpus + queries + qrels) |
| **Corpus** | 720 docs (PDF 페이지 단위 분할) |
| **Queries** | 114개 (Test split) |
| **도메인** | Finance(금융) + Public(공공) + Law(법률) + Commerce(상업) |

**도메인 세부 분류**:
| 서브도메인 | 예시 문서 | 예상 비율 |
|-----------|---------|---------|
| **Finance** | 지방은행 전환 가이드, 통화신용정책, 핀테크 생태계 | ~25% |
| **Public** | 국가재정운용계획, 국립대학 육성사업, 행정안전부 업무계획 | ~30% |
| **Law** | 민사/행정/형사/특허 판례, 세금제도 | ~25% |
| **Commerce** | 이커머스 시장 보고서, 산업 분석 | ~20% |

**카테고리 분석**:
- 한국 **실무 문서** (정부 정책서, 법률 판례, 금융 보고서)
- 전문 용어가 밀집하여 **BM25 강점** 예상
- 도메인 혼합으로 단일 카테고리로 처리하기 어려움 → 서브도메인 분리 필요

**→ 카테고리 배정: `LEGAL` (Law 서브셋) + `BUSINESS` (Finance+Public+Commerce 서브셋)**

---

### 3. MIRACL (miracl/miracl)
**URL**: https://huggingface.co/datasets/miracl/miracl

| 항목 | 내용 |
|------|------|
| **원본** | Making a MIRACL (arXiv:2210.09984, TACL 2023) |
| **언어** | 18개 언어 (한국어 포함) |
| **형식** | BeIR 유사 (corpus + queries + qrels) |
| **Corpus (ko)** | 1,500,000 docs (한국어 Wikipedia) |
| **Queries (ko)** | Train 868개 + Dev 213개 |
| **QA 유형** | 원어민 주석, 단일 홉 |
| **라이선스** | Apache 2.0 |

**카테고리 분석**:
- 다국어 검색 표준 벤치마크 (MTEB 공식 채택)
- 한국어 원어민 주석 → **고품질 레이블**
- Wikipedia 기반 → 일반 지식 도메인
- BGE-M3가 이 벤치마크에서 nDCG@10 = 70.0으로 최고 성능 기록

**→ 카테고리 배정: `GENERAL` (다국어 Wikipedia, 품질 기준)**

---

### 4. publichealth-qa
**URL**: https://huggingface.co/datasets/xhluca/publichealth-qa

| 항목 | 내용 |
|------|------|
| **원본** | CDC + WHO FAQ 페이지 스크래핑 (2019.12~2020.04) |
| **언어** | 8개 언어 (한국어 77개 QA 포함) |
| **형식** | Q&A 쌍 (질문 + 답변 + 출처 URL + 섹션) |
| **크기 (ko)** | 77개 QA 쌍 |
| **QA 유형** | 전문 FAQ (공식 기관 문서 기반) |
| **라이선스** | CC-BY-NC-SA-3.0 |

**카테고리 분석**:
- 공중보건/의료 전문 도메인
- **짧은 질문 + 긴 상세 답변** (69~3,900자) → Dense 검색 유리
- 공식 기관(CDC/WHO) 문서 → 전문 용어 밀도 높음
- 77개로 소규모 → 샘플 전체 사용

**→ 카테고리 배정: `MEDICAL` (의료/공중보건, 신규 카테고리)**

---

### 5. Belebele (facebook/belebele)
**URL**: https://huggingface.co/datasets/facebook/belebele

| 항목 | 내용 |
|------|------|
| **원본** | BELEBELE 논문 (Meta AI), FLORES 프로젝트 지문 기반 |
| **언어** | 99개 언어 (한국어: `kor_Hang`) |
| **형식** | 지문 + 질문 + 4지선다 (MRC) |
| **크기 (ko)** | ~1,400개 (test split only) |
| **QA 유형** | 독해력 평가 (Machine Reading Comprehension) |
| **지문 길이** | 150-300 단어 (짧은 산문) |

**카테고리 분석**:
- 짧은 산문형 지문 (뉴스/위키 스타일)
- 4지선다 → Passage Retrieval 평가에 적합
- 다국어 일관성 평가 가능 (99개 언어 동일 포맷)
- Ko-StrategyQA (Multi-hop)와 대조적 → **단순 사실 검색** 능력 평가

**→ 카테고리 배정: `GENERAL` (짧은 지문 MRC, 단순 사실 검색)**

---

### 6. MrTiDy (mteb/mrtidy)
**URL**: https://huggingface.co/datasets/mteb/mrtidy

| 항목 | 내용 |
|------|------|
| **원본** | Mr-TyDi (arXiv:2108.08787), MTEB 통합 |
| **언어** | 11개 언어 (한국어 포함) |
| **형식** | BeIR (corpus + queries + qrels) |
| **Corpus (ko)** | 1,500,000 docs (백과사전 스타일) |
| **Queries (ko)** | Train+Dev+Test 2,020개 |
| **QA 유형** | 단일 홉, 사실 기반 |
| **라이선스** | CC-BY-SA-3.0 |

**카테고리 분석**:
- MIRACL과 유사한 백과사전 스타일
- MTEB 표준 벤치마크 → 글로벌 비교 가능
- 코퍼스 규모 1.5M → **대규모 검색 성능** 평가
- 한국어 코퍼스: 40,300개 관련성 판단 레이블

**→ 카테고리 배정: `GENERAL` (백과사전 대규모, MTEB 표준)**

---

## 카테고리 매핑 결과

### 기존 5개 카테고리 → 데이터셋 기반 수정

| 기존 | 수정 후 | 데이터셋 | 변경 이유 |
|------|---------|---------|---------|
| TECHNICAL | **TECHNICAL** (유지) | 사용자 업로드 문서 | 데이터셋 없음, 사용자 문서 기반 |
| LEGAL | **LEGAL** (유지) | markers_bm (law 서브셋) | ✅ 데이터셋 확보 |
| ACADEMIC | ~~ACADEMIC~~ → **MEDICAL** | publichealth-qa | 의료 FAQ 데이터셋으로 대체 |
| BUSINESS | **BUSINESS** (유지) | markers_bm (finance+public+commerce) | ✅ 데이터셋 확보 |
| GENERAL | **GENERAL** (강화) | Ko-StrategyQA + MIRACL + Belebele + MrTiDy | 4개 데이터셋 통합 |

### 확정 카테고리 5종

```
┌─────────────────────────────────────────────────────────────────┐
│                    벤치마크 카테고리 5종                         │
├──────────────┬──────────────────────────────────────────────────┤
│ GENERAL      │ Ko-StrategyQA + MIRACL(ko) + Belebele(ko) +     │
│              │ MrTiDy(ko)                                       │
│              │ → 백과사전/Wikipedia 기반 일반 지식              │
├──────────────┬──────────────────────────────────────────────────┤
│ LEGAL        │ markers_bm (law 서브셋 ~180 docs)               │
│              │ → 민사/행정/형사/특허 판례                       │
├──────────────┬──────────────────────────────────────────────────┤
│ BUSINESS     │ markers_bm (finance+public+commerce ~540 docs)   │
│              │ → 금융정책, 정부문서, 산업분석                   │
├──────────────┬──────────────────────────────────────────────────┤
│ MEDICAL      │ publichealth-qa (korean, 77 QA)                  │
│              │ → CDC/WHO 공중보건 FAQ (의료 전문 도메인)         │
├──────────────┬──────────────────────────────────────────────────┤
│ TECHNICAL    │ 사용자 업로드 문서 (데이터셋 없음)               │
│              │ → API 문서, 기술 매뉴얼, 개발 가이드             │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## 벤치마크 실행 방식 (데이터셋 통합)

### 기존 방식 (LLM 생성 QA)
```
사용자 문서 → 청킹 → LLM으로 QA 생성 → 벤치마크
```

### 신규 방식 (HuggingFace 데이터셋 기반)
```
HF 데이터셋 로드 → 코퍼스 인덱싱 → 기존 queries/qrels 사용 → 벤치마크
```

### 혼합 방식 (권장)
```
HF 표준 데이터셋 (GENERAL/LEGAL/BUSINESS/MEDICAL) + 사용자 문서 (TECHNICAL)
→ 각 카테고리별 인덱싱 → 벤치마크 → 카테고리별 순위 + 선정 근거
```

---

## GENERAL 카테고리 세분화 전략

GENERAL 카테고리에 4개 데이터셋이 집중되므로, 다음과 같이 특성별로 활용:

| 데이터셋 | 특성 | 평가 관점 |
|---------|------|---------|
| Ko-StrategyQA | Multi-hop, 한국어 | 복잡한 추론이 필요한 검색 |
| MIRACL (ko) | 원어민 주석, 고품질 | 검색 품질 기준점 (gold standard) |
| Belebele (ko) | 짧은 지문, MRC | 단순 사실 검색 속도 |
| MrTiDy (ko) | 대규모 (1.5M) | 대용량 코퍼스 검색 확장성 |

**권장 실행 방식**: MIRACL(ko)을 GENERAL의 **기준(primary)** 데이터셋으로 사용, 나머지는 검증용(secondary)으로 사용.

---

## 데이터셋 규모별 샘플링 전략

| 카테고리 | 데이터셋 | 코퍼스 크기 | 쿼리 수 | 샘플링 방식 |
|---------|---------|-----------|--------|-----------|
| GENERAL | MIRACL(ko) | 1,500,000 | 1,081 | 코퍼스 5만 샘플 + 쿼리 전체 |
| GENERAL | Ko-StrategyQA | 27,800 | 8,500 | 코퍼스 전체 + 쿼리 500 샘플 |
| GENERAL | Belebele(ko) | N/A | 1,400 | test split 전체 사용 |
| GENERAL | MrTiDy(ko) | 1,500,000 | 2,020 | 코퍼스 5만 샘플 + 쿼리 200 샘플 |
| LEGAL | markers_bm (law) | ~180 | ~30 | 전체 사용 |
| BUSINESS | markers_bm (other) | ~540 | ~84 | 전체 사용 |
| MEDICAL | publichealth-qa(ko) | N/A | 77 | 전체 사용 |
| TECHNICAL | 사용자 업로드 | 가변 | LLM 생성 | 문서당 10-20 QA |

---

## 모델별 예상 강점 (가설)

데이터셋 특성 기반 사전 가설:

| 카테고리 | 예상 1위 조합 | 근거 |
|---------|------------|------|
| GENERAL | bge-m3 + splade | MIRACL 벤치마크에서 BGE-M3 최고 성능 (nDCG 70.0) |
| GENERAL (Multi-hop) | bge-m3 + korean_bm25 | 한국어 정확 매칭 + 의미 검색 조합 |
| LEGAL | kosimcse + korean_bm25 | 법률 전문 용어 한국어 형태소 정밀 매칭 |
| BUSINESS | e5 + korean_bm25 | 다국어 비즈니스 용어 + 정확 키워드 검색 |
| MEDICAL | bge-m3 + splade | FAQ 형식의 의미론적 검색, SPLADE 어휘 확장 |
| TECHNICAL | e5 + korean_bm25 | API/코드 정확 매칭 (BM25 강점) |

> **이 가설은 실제 벤치마크 결과로 검증/반박됩니다.**

---

## 계획 변경사항 요약

`PLAN_SERVICE_BENCH.md`에 반영될 변경:

1. **카테고리 변경**: `ACADEMIC` → `MEDICAL` (publichealth-qa 데이터셋 기반)
2. **데이터셋 소스 추가**: HuggingFace 데이터셋 6종을 표준 벤치마크 소스로 통합
3. **GENERAL 강화**: 4개 데이터셋 통합 (Ko-StrategyQA + MIRACL + Belebele + MrTiDy)
4. **markers_bm 분리**: LEGAL/BUSINESS 서브도메인으로 분리 처리
5. **혼합 실행 모드**: HF 데이터셋 모드 + 사용자 문서 모드

---

*참조: PLAN_SERVICE_BENCH.md, rag_benchmark_references.md*
