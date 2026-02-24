# Plan: 문서 종류별 RAG 모델 선정 벤치마크 시스템

> 작성일: 2026-02-24
> 목표: RAG 서비스에서 사용자 문서 종류에 따라 최적 Dense×Sparse 조합을 선정하는 근거를 제공하는 벤치마크 시스템 구축

---

## 골 분석 (Goal-Backward Verification)

```
[최종 출력]
"PDF 기술 문서 → bge-m3 + splade 권장 (Context Recall 0.94 1위, 이유: 밀집 기술 용어 처리 강점)"
"법률 계약서 → kosimcse + korean_bm25 권장 (한국어 형태소 기반 정확도 강점)"
...
↑
[필요] 문서 타입별 선정 근거 보고서 (Markdown + JSON)
↑
[필요] 분석: 타입별 모델 순위 + 강점/약점 + 중복 압축
↑
[필요] 타입별 RAGAS 점수 집계
↑
[필요] 타입별 벤치마크 실행 (고정 파이프라인: ColBERT + Contextual)
↑
[필요] 타입별 QA 생성 + 청킹
↑
[필요] 타입 분류 + 샘플링된 문서 콘텐츠
↑
[필요] 사용자 문서 입력 → 멀티포맷 파서 → 타입 감지
```

---

## 현재 상태 요약

| 항목 | 현재 |
|------|------|
| Dense 모델 | kosimcse, e5, bge-m3 (HF 로컬), openai-large, upstage (API) |
| Sparse 모델 | korean_bm25, splade |
| 리랭커 | none / colbert / flashrank (선택) |
| Contextual | none / contextual (선택) |
| 문서 포맷 | PDF → Markdown (pdf_converter.py) |
| 문서 타입 구분 | 없음 (단일 스택으로 처리) |
| 비교 분석 | 없음 (점수 출력만) |

## 목표 상태

| 항목 | 목표 |
|------|------|
| Dense 모델 | 현재 유지 (HF 3종 + API 2종) |
| Sparse 모델 | 현재 유지 (korean_bm25, splade) |
| **리랭커** | **ColBERT 고정** (서비스 파이프라인에서 항상 적용) |
| **Contextual** | **고정 적용** (서비스 파이프라인에서 항상 적용) |
| 비교 조합 수 | 3(HF Dense) × 2 Sparse = **6 기본 조합** (API 2종 포함 시 최대 10개, **≤10 제약 충족**) |
| 문서 포맷 | PDF, DOCX, HTML, TXT, MD 지원 |
| **문서 타입** | **5종** (GENERAL, LEGAL, BUSINESS, MEDICAL, TECHNICAL) — `ACADEMIC` → `MEDICAL` 변경 |
| **분석 출력** | **타입별 순위 + 강/약점 + 선정 근거** |

---

## 고정 파이프라인 설계

서비스 벤치마크의 모든 조합은 다음을 **항상 포함**:
```
[Dense Model] + [Sparse Model] + ColBERT Reranker + Contextual Retrieval
```

즉, 벤치마크에서 비교하는 변수는 **Dense × Sparse 조합(6가지)**뿐이며,
리랭커와 Contextual은 통제 변수(고정)다.

---

## 표준 벤치마크 데이터셋

> 상세 분석: `docs/research/service_bench/dataset_analysis.md`

| 카테고리 | 주 데이터셋 | 보조 데이터셋 | 코퍼스 (ko) | 쿼리 (ko) |
|---------|-----------|------------|-----------|---------|
| **GENERAL** | MIRACL (miracl/miracl) | Ko-StrategyQA, Belebele, MrTiDy | 1.5M (샘플링) | 1,081 |
| **LEGAL** | markers_bm (law 서브셋) | — | ~180 docs | ~30 |
| **BUSINESS** | markers_bm (finance+public+commerce) | — | ~540 docs | ~84 |
| **MEDICAL** | publichealth-qa (ko) | — | 77 QA | 77 |
| **TECHNICAL** | 사용자 업로드 문서 | — | 가변 | LLM 생성 |

### GENERAL 카테고리 데이터셋 특성 구분

| 데이터셋 | 특성 | 평가 관점 |
|---------|------|---------|
| MIRACL (ko) | 원어민 주석, 고품질 | 품질 기준점 (primary) |
| Ko-StrategyQA | Multi-hop, 한국어 | 복잡한 추론 검색 |
| Belebele (kor_Hang) | 짧은 지문, 4지선다 | 단순 사실 검색 |
| MrTiDy (ko) | 대규모 1.5M, MTEB | 확장성 검색 |

### 카테고리 변경 이력

| 변경 | 이유 |
|------|------|
| `ACADEMIC` → `MEDICAL` | publichealth-qa (CDC/WHO FAQ) 데이터셋이 의료 도메인을 대변 |
| Dense 모델 3종 → **4종** | snowflake-arctic-embed-l-v2.0-ko 추가 (한국어 Retrieval SOTA 0.7404) |

---

## 신규 Dense 모델: snowflake-arctic-embed-l-v2.0-ko

> 상세 분석: `docs/research/service_bench/report_structure_and_model_analysis.md`

| 항목 | 내용 |
|------|------|
| **모델 키** | `snowflake-ko` |
| **실제 모델명** | `dragonkue/snowflake-arctic-embed-l-v2.0-ko` |
| **파라미터** | 0.6B (BGE-M3와 동일 백본: bge-m3-retromae) |
| **임베딩 차원** | 1024 (256 압축 지원) |
| **최대 토큰** | 8192 |
| **한국어 평균** | **0.7404** (BGE-M3 0.7242, BGE-M3-ko 0.7300 초과) |
| **특이사항** | 1300 토큰 이상 긴 문서는 다른 모델 권장 |

**데이터셋별 강점** (우리 벤치마크 데이터셋 기준):

| 데이터셋 → 카테고리 | snowflake-ko | BAAI/bge-m3 | 우위 |
|--------------------|-------------|------------|------|
| AutoRAG(법률/금융) → LEGAL/BUSINESS | **0.9093** | 0.8301 | snowflake-ko +9.5% |
| PublicHealth → MEDICAL | **0.8337** | 0.8041 | snowflake-ko +3.7% |
| MIRACL → GENERAL | 0.6685 | **0.7015** | bge-m3 +4.9% |
| MrTiDy → GENERAL | 0.5712 | **0.6471** | bge-m3 +13.3% |

**결론**: snowflake-ko는 실무 문서(LEGAL/BUSINESS/MEDICAL) 카테고리에서 SOTA. bge-m3는 대규모 Wikipedia(GENERAL)에서 강점. 두 모델을 모두 포함하면 카테고리별 최적 모델 비교 가능.

**업데이트된 조합 수**: 4 HF Dense × 2 Sparse = **8개** (≤10 제약 충족 ✅)

---

## Phase 계획

---

### Phase 1: 문서 종류 시스템 + 파이프라인 고정 설정

**목표**: 문서 타입 정의, 멀티포맷 파서, "service" 프리셋 추가

#### Task 1.1 — `rag_bench/document_types/` 모듈 신규 생성
**파일**:
- `rag_bench/document_types/__init__.py`
- `rag_bench/document_types/types.py`
- `rag_bench/document_types/classifier.py`
- `rag_bench/document_types/sampler.py`

**구현 내용**:

`types.py`:
```python
class DocType(str, Enum):
    TECHNICAL  = "technical"   # API 문서, 개발 가이드, 기술 매뉴얼
    LEGAL      = "legal"       # 법률, 계약서, 판례 (markers_bm law)
    BUSINESS   = "business"    # 금융/공공/상업 보고서 (markers_bm finance+public+commerce)
    MEDICAL    = "medical"     # 의료/공중보건 FAQ (publichealth-qa CDC/WHO)
    GENERAL    = "general"     # 백과사전/위키 (MIRACL + Ko-StrategyQA + Belebele + MrTiDy)

DOC_TYPE_METADATA = {
    DocType.TECHNICAL:  {"sampling_ratio": 0.15, "chunk_emphasis": "code+structure",
                         "hf_dataset": None,  # 사용자 문서 기반
                         "expected_top": "e5+korean_bm25"},
    DocType.LEGAL:      {"sampling_ratio": 0.20, "chunk_emphasis": "precision",
                         "hf_dataset": "yjoonjang/markers_bm",  "hf_subset": "law",
                         "expected_top": "kosimcse+korean_bm25"},
    DocType.BUSINESS:   {"sampling_ratio": 0.20, "chunk_emphasis": "summary+figures",
                         "hf_dataset": "yjoonjang/markers_bm",  "hf_subset": "finance+public+commerce",
                         "expected_top": "e5+korean_bm25"},
    DocType.MEDICAL:    {"sampling_ratio": 1.00, "chunk_emphasis": "faq",
                         "hf_dataset": "xhluca/publichealth-qa", "hf_subset": "korean",
                         "expected_top": "bge-m3+splade"},
    DocType.GENERAL:    {"sampling_ratio": 0.10, "chunk_emphasis": "balanced",
                         "hf_dataset": "miracl/miracl",          "hf_subset": "ko",  # primary
                         "secondary": ["taeminlee/Ko-StrategyQA", "facebook/belebele", "mteb/mrtidy"],
                         "expected_top": "bge-m3+splade"},
}
```

`classifier.py`:
- 키워드 기반 1차 분류 (법률 용어, 기술 용어 등)
- 파일 확장자 힌트 활용
- LLM 기반 2차 분류 (optional, 정확도 향상)

`sampler.py`:
- 타입별 샘플링 전략 (페이지 비율, 최대 청크 수)
- `sample_document(path, doc_type) -> str` 인터페이스

#### Task 1.2 — `rag_bench/indexing/multi_parser.py` 신규 생성
**파일**: `rag_bench/indexing/multi_parser.py`

**지원 포맷**:
| 포맷 | 파서 | 의존성 |
|------|------|------|
| `.pdf` | pymupdf4llm (기존) | 이미 설치됨 |
| `.docx` | python-docx | 신규 추가 |
| `.html`/`.htm` | BeautifulSoup4 | 신규 추가 |
| `.txt`/`.md` | 직접 읽기 | 없음 |
| `.csv`/`.xlsx` | pandas | 신규 추가 (선택) |

```python
def parse_document(path: str | Path, sample: bool = False, doc_type: DocType | None = None) -> str:
    """통합 문서 파서. 확장자 자동 감지 후 파싱."""
    ...
```

#### Task 1.3 — `combo/spec.py` "service" 프리셋 추가
**파일**: `rag_bench/combo/spec.py`

```python
# combo/spec.py에 snowflake-ko 추가 후
_HF_DENSE_MODELS = ["kosimcse", "e5", "bge-m3", "snowflake-ko"]  # 3종 → 4종

PRESETS = {
    ...
    "service": {               # ← 신규 (서비스 모델 선정용)
        "dense_models": _HF_DENSE_MODELS,           # kosimcse, e5, bge-m3, snowflake-ko
        "sparse_models": list(SPARSE_TYPES),         # korean_bm25, splade
        "rerankers": ["colbert"],                    # ColBERT 고정
        "llm_support": ["contextual"],               # Contextual 고정
    },
}
```
→ 유효 조합: 4 × 2 × 1 × 1 = **8개** (≤10 제약 충족 ✅)

**`dense_sparse.py`에도 snowflake-ko 추가**:
```python
DENSE_MODELS = {
    ...
    "snowflake-ko": {
        "model_name": "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        "dim": 1024,
        "max_tokens": 8192,
        "note": "한국어 실무문서 SOTA (법률/금융/의료). 1300토큰 이하 권장",
    },
}
```

**의존성 변경**: `pyproject.toml`에 `python-docx`, `beautifulsoup4` 추가

---

### Phase 2: 문서 타입별 벤치마크 오케스트레이터

**목표**: HuggingFace 데이터셋 로드 + 사용자 문서 입력 → 타입별 벤치마크 실행

#### Task 2.0 — `rag_bench/datasets/hf_loader.py` 신규 생성

HuggingFace 표준 데이터셋을 BeIR 포맷으로 로드하는 통합 로더.

```python
class HFDatasetLoader:
    """HuggingFace 데이터셋 → (corpus, queries, qrels) 변환."""

    DATASETS = {
        DocType.GENERAL: {
            "primary":   ("miracl/miracl",         {"language": "ko"}),
            "secondary": [
                ("taeminlee/Ko-StrategyQA",  {}),
                ("facebook/belebele",         {"name": "kor_Hang"}),
                ("mteb/mrtidy",               {"name": "korean-corpus"}),
            ],
        },
        DocType.LEGAL:    ("yjoonjang/markers_bm",    {"subset": "law"}),
        DocType.BUSINESS: ("yjoonjang/markers_bm",    {"subset": "finance+public+commerce"}),
        DocType.MEDICAL:  ("xhluca/publichealth-qa",  {"name": "korean"}),
        # DocType.TECHNICAL: 사용자 업로드 문서 (HF 데이터셋 없음)
    }

    def load(self, doc_type: DocType, max_corpus: int = 50_000) -> BeirDataset:
        """지정 카테고리 데이터셋 로드 + 코퍼스 샘플링."""
        ...
```

**샘플링 전략**:
| 데이터셋 | 코퍼스 크기 | max_corpus | 쿼리 수 |
|---------|-----------|-----------|--------|
| MIRACL(ko) | 1.5M | 50,000 | 1,081 |
| Ko-StrategyQA | 27.8k | 전체 | 500 샘플 |
| Belebele(ko) | ~1,400 | 전체 | 전체 |
| MrTiDy(ko) | 1.5M | 50,000 | 200 샘플 |
| markers_bm | 720 | 전체 | 전체 |
| publichealth-qa(ko) | 77 | 전체 | 전체 |

#### Task 2.1 — `rag_bench/scripts/run_service_bench.py` 신규 생성

**CLI 인터페이스**:
```bash
# HuggingFace 표준 데이터셋 모드 (권장)
uv run python -m rag_bench.scripts.run_service_bench \
    --mode hf \                        # HuggingFace 데이터셋 사용
    --categories general,legal,business,medical \
    --preset service \                 # ColBERT + Contextual 고정

# 사용자 문서 업로드 모드
uv run python -m rag_bench.scripts.run_service_bench \
    --mode docs \
    --docs_dir /path/to/user/docs \   # 사용자 문서 디렉토리
    --preset service \                 # ColBERT + Contextual 고정
    --num_qa 20 \                      # 타입별 QA 수
    --output_dir _benchdata/service_run/
```

**실행 플로우**:
```
1. docs_dir 내 파일 스캔
2. 각 파일 → DocType 분류
3. DocType 그룹핑 (예: {TECHNICAL: [a.pdf, b.docx], LEGAL: [c.pdf]})
4. 타입별 루프:
   a. 샘플링 → 파싱 → 마크다운 변환
   b. Parent-Child 청킹
   c. QA 생성 (generate_qa.py 재활용)
   d. 6개 조합 × 각 QA 실행 (Pass 1: 레이턴시, Pass 2: RAGAS)
   e. 결과 JSON 저장 (타입별)
5. 타입별 결과 집계
```

#### Task 2.2 — 체크포인트 시스템
- 타입별 진행 상태 JSON 저장
- 재실행 시 완료된 타입 스킵
- `_benchdata/service_run/{doc_type}/checkpoint.json`

---

### Phase 3: 분석 모듈 — 순위 + 강/약점 + 선정 로직

**목표**: 결과 데이터 → 인사이트 + 최종 추천

#### Task 3.1 — `rag_bench/analysis/` 모듈 신규 생성
**파일**:
- `rag_bench/analysis/__init__.py`
- `rag_bench/analysis/ranker.py`
- `rag_bench/analysis/insight.py`
- `rag_bench/analysis/deduplication.py`
- `rag_bench/analysis/selector.py`

**`ranker.py`**:
```python
def rank_by_doc_type(results: Dict[DocType, pd.DataFrame]) -> Dict[DocType, pd.DataFrame]:
    """각 문서 타입별로 종합 점수 계산 후 조합 순위 반환.

    W&B Horangi v3 패턴 적용:
    - 1차: NDCG@10 기반 retrieval 순위 (표준 정보검색 지표)
    - 2차: RAGAS 다차원 평가 (Recall × 0.35 + Precision × 0.30 + Faithfulness × 0.20 + Relevancy × 0.15)
    - Zero-shot Pass(레이턴시) + RAGAS Pass(품질) 2단계 결과 통합
    """
    ...
```

**`insight.py`**:
```python
def analyze_strengths_weaknesses(
    ranked: Dict[DocType, pd.DataFrame]
) -> Dict[str, Dict[str, str]]:
    """
    각 조합의 강점/약점을 문서 타입 간 비교로 도출.

    예시 출력:
    {
        "bge-m3+splade": {
            "strengths": ["technical 1위", "academic 2위"],
            "weaknesses": ["legal 최하위", "business 4위"],
            "pattern": "전문 용어 밀집 문서에 강점, 한국어 법률 용어 약점"
        }
    }
    """
```

**`deduplication.py`**:
```python
def compress_similar_results(
    ranked: Dict[DocType, pd.DataFrame],
    similarity_threshold: float = 0.05  # 점수 차이 5% 이내를 동등 처리
) -> Dict[DocType, pd.DataFrame]:
    """
    통계적으로 유의미한 차이가 없는 조합을 그룹핑.
    '동점 그룹' 내에서는 비용/속도 기준으로 우선순위 결정.
    """
```

**`selector.py`**:
```python
def generate_selection_report(
    ranked: Dict[DocType, pd.DataFrame],
    insights: Dict[str, Dict],
    compressed: Dict[DocType, pd.DataFrame],
) -> SelectionReport:
    """
    최종 선정 보고서 생성.

    출력:
    - 타입별 1위 조합 + 선정 이유
    - 공통 우승 조합 (여러 타입에서 1위)
    - "모르면 이걸 써라" 기본 추천 1개
    - 선정 불가 사유 (데이터 부족 타입)
    """
```

#### Task 3.2 — 보고서 생성기
**파일**: `rag_bench/analysis/reporter.py`

출력 형식:
- **JSON**: `selection_report.json` (프로그래매틱 처리용)
- **Markdown**: `selection_report.md` (사람이 읽는 형식)

Markdown 보고서 구조 (W&B Horangi v3 패턴 적용):
```markdown
# RAG 모델 선정 보고서

---
## Section 1: 평가 개요
- 고정 파이프라인: [Dense] + [Sparse] + ColBERT Reranker + Contextual Retrieval
- 비교 변수: 8개 조합 (4 Dense × 2 Sparse)
- 평가 데이터셋: 5개 카테고리 (GENERAL/LEGAL/BUSINESS/MEDICAL/TECHNICAL)
- 평가 지표: NDCG@10 (주) + RAGAS 4종 (부)
- 실행 환경: [플랫폼, GPU, 모델 캐시 경로]

---
## Section 2: 종합 성능 리더보드

| 조합 | GENERAL | LEGAL | BUSINESS | MEDICAL | TECHNICAL | **평균** |
|------|---------|-------|----------|---------|-----------|---------|
| snowflake-ko + korean_bm25 | 0.XX | **0.XX** | **0.XX** | **0.XX** | 0.XX | **0.XX** |
| bge-m3 + splade | **0.XX** | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| ... | | | | | | |

---
## Section 3: 카테고리별 상세 비교
### 3-1. GENERAL (MIRACL + Ko-StrategyQA + Belebele + MrTiDy)
[카테고리별 Bar Chart + 해석]

### 3-2. LEGAL (markers_bm - law)
[카테고리별 Bar Chart + 해석]

...

---
## Section 4: 조합별 강점/약점 프로파일
[레이더 차트: 조합당 1개]
[히트맵: 조합 × 카테고리 성능 격자]

예: snowflake-ko + korean_bm25
- 강점: LEGAL(1위), BUSINESS(1위), MEDICAL(1위)
- 약점: GENERAL 대규모(MrTiDy에서 bge-m3 대비 -13%)
- 패턴: "실무 특화 문서(법률/금융/의료)에서 압도적, 대용량 Wikipedia 검색에서 약점"

---
## Section 5: 동점 그룹 압축
점수 차 5% 이내 조합 → 동점 그룹으로 통합
"A조합과 B조합의 차이는 통계적으로 유의미하지 않음 → 속도/비용 기준으로 A 권장"

---
## Section 6: 최종 선정 가이드
| 문서 타입 | 1순위 조합 | 선정 이유 |
|-----------|-----------|---------|
| GENERAL | bge-m3 + splade | MIRACL 벤치마크 최강 |
| LEGAL | snowflake-ko + korean_bm25 | AutoRAG 법률 SOTA +9.5% |
| BUSINESS | snowflake-ko + korean_bm25 | 금융법률 학습 데이터 |
| MEDICAL | snowflake-ko + splade | PublicHealth 최강 |
| TECHNICAL | e5 + korean_bm25 | 기술 용어 정확 매칭 |

공통 추천 (문서 타입 혼용 시):
→ snowflake-ko + korean_bm25 (LEGAL/BUSINESS/MEDICAL 3개 카테고리 1위)
→ bge-m3 + splade (GENERAL 특화, 대용량 Wikipedia)
```

---

### Phase 4: 시각화 + 노트북 통합

**목표**: 분석 결과를 시각적으로 표현하는 차트 + 통합 노트북

#### Task 4.1 — `rag_bench_local/visualizer.py` 확장
**추가 함수**:
- `plot_doctype_heatmap(ranked_by_type)` — 모델×타입 성능 히트맵
- `plot_model_radar(insights)` — 타입별 강점 레이더 차트 (조합당 1개)
- `plot_selection_summary(report)` — 타입별 추천 조합 요약 테이블
- `plot_score_distribution(results)` — 점수 분포로 "동점 그룹" 시각화

#### Task 4.2 — `rag_bench_local/rag_benchmark.ipynb` 신규 섹션 추가
새 섹션:
```
Section 0: 사용자 문서 입력 (경로 + 타입 분류 미리보기)
Section 1~8: 기존 (Quick Start, 청킹, QA 생성, 벤치마크 실행 등)
Section 9: 문서 타입별 결과 (타입별 히트맵 + 순위표)
Section 10: 강점/약점 분석 (레이더 차트)
Section 11: 최종 모델 선정 보고서 (selection_report.md 인라인 표시)
```

---

## 의존성 변경 사항

**`pyproject.toml` 추가**:
```toml
"python-docx>=1.1",
"beautifulsoup4>=4.12",
"lxml>=5.0",            # BeautifulSoup HTML 파서
```

**선택 추가** (Excel 지원 시):
```toml
"openpyxl>=3.1",
```

---

## 파일 변경 목록

| 파일 | 변경 | Phase |
|------|------|-------|
| `rag_bench/document_types/__init__.py` | **NEW** | 1 |
| `rag_bench/document_types/types.py` | **NEW** | 1 — GENERAL/LEGAL/BUSINESS/MEDICAL/TECHNICAL |
| `rag_bench/document_types/classifier.py` | **NEW** | 1 |
| `rag_bench/document_types/sampler.py` | **NEW** | 1 |
| `rag_bench/indexing/multi_parser.py` | **NEW** | 1 |
| `rag_bench/datasets/__init__.py` | **NEW** | 1 |
| `rag_bench/datasets/hf_loader.py` | **NEW** | 2 — 6개 HF 데이터셋 통합 로더 |
| `rag_bench/combo/spec.py` | MODIFIED | 1 |
| `pyproject.toml` | MODIFIED | 1 |
| `rag_bench/scripts/run_service_bench.py` | **NEW** | 2 — hf/docs 듀얼 모드 |
| `rag_bench/analysis/__init__.py` | **NEW** | 3 |
| `rag_bench/analysis/ranker.py` | **NEW** | 3 |
| `rag_bench/analysis/insight.py` | **NEW** | 3 |
| `rag_bench/analysis/deduplication.py` | **NEW** | 3 |
| `rag_bench/analysis/selector.py` | **NEW** | 3 |
| `rag_bench/analysis/reporter.py` | **NEW** | 3 |
| `rag_bench_local/visualizer.py` | MODIFIED | 4 |
| `rag_bench_local/rag_benchmark.ipynb` | MODIFIED | 4 |
| `MEMORY.md` | MODIFIED | 각 Phase 완료 시 |

---

## 실행 순서 (의존성)

```
Phase 1 (타입 시스템 + 파서 + 프리셋)
    ↓
Phase 2 (오케스트레이터) ← Phase 1 완료 필요
    ↓
Phase 3 (분석 모듈) ← Phase 2 완료 필요 (실제 결과 데이터로 검증)
    ↓
Phase 4 (시각화) ← Phase 3 완료 필요
```

Phase 1의 Task 1.1, 1.2, 1.3은 상호 독립적으로 병렬 진행 가능.

---

## 리스크 & 대응

| 리스크 | 가능성 | 대응 |
|--------|--------|------|
| DOCX/HTML 파서 텍스트 품질 저하 | 중 | 파싱 후 청크 최소 크기 필터로 불량 텍스트 제거 |
| 문서 타입 분류 오류 | 중 | classifier.py에 수동 override 옵션 (`--doc_type=legal`) |
| 특정 타입 문서가 없을 때 분석 불가 | 높 | 2개 미만 타입이면 타입별 분석 스킵 + 경고 출력 |
| ColBERT OOM (Apple Silicon) | 낮 | 기존 CPU 고정 + 싱글톤 캐시 적용됨 |
| API 모델(openai-large, upstage) 비용 | 중 | 기본값은 HF 3종만, `--include_api` 플래그로 선택 활성화 |
| QA 생성 비용 (타입×num_qa) | 중 | 기본 `num_qa=10` (타입당), `--num_qa` 조정 가능 |

---

## 리서치 참조

> 상세 근거: `docs/research/service_bench/rag_benchmark_references.md`

| 항목 | 연구 근거 |
|------|---------|
| ColBERT 고정 | 오프-토픽 응답 25% 감소, 대규모 확장성 우수 (IBM Developer) |
| Contextual 고정 | 검색 실패율 최대 67% 감소 (Anthropic, 2024.09) |
| BGE-M3 포함 | MIRACL nDCG@10 = 70.0, 한국어 포함 100+ 언어 오픈소스 최강 |
| BM25 vs SPLADE | 기술/법률 → BM25 강점 / 일반/학술 → SPLADE 강점 (BEIR SPLADE-v3) |
| 5종 문서 타입 | Harvard JOLT (법률), NVIDIA (기술), BEIR (학술), 청킹 전략 연구 기반 |
| RAGAS 가중치 | Context Recall 35% 최우선 — 서비스에서 누락이 오탐보다 치명적 |

---

## 성공 기준 (DoD)

- [ ] `parse_document(path)` — PDF, DOCX, HTML, TXT 4가지 포맷 정상 파싱
- [ ] `classify_document(text)` — 5가지 타입 분류 정확도 (수동 테스트 3개/타입)
- [ ] `run_service_bench.py` — `--preset service`로 6조합 × N 타입 E2E 실행 성공
- [ ] `generate_selection_report()` — Markdown 보고서 생성 (타입별 1위 + 이유 포함)
- [ ] `plot_doctype_heatmap()` — 히트맵 렌더링 (Jupyter)
- [ ] 중복 결과 압축: 점수 차 5% 이내 조합 그룹핑 로직 동작 확인
