# RAG Bench - 한국어 RAG 파이프라인 비교 평가

한국어 문서(PDF)를 대상으로 다양한 RAG 파이프라인 성능을 정량 평가하는 프로젝트입니다.

Strategy Pattern 기반 모듈화 벤치마크 시스템으로, RAGAS 평가를 통해 72개 전략 조합을 통일된 인터페이스로 비교합니다.

## 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RAG Bench 전체 파이프라인                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ① 문서 준비         ② QA 생성            ③ 벤치마크 실행                   │
│  ┌──────────┐      ┌──────────────┐      ┌───────────────────────────┐     │
│  │ PDF 문서 │──→   │ GPT-4o-mini  │──→   │ 72개 전략 조합 × 20 쿼리 │     │
│  │ (한국어) │      │ QA 자동 생성 │      │                           │     │
│  └────┬─────┘      └──────┬───────┘      │  Pass 1: 레이턴시 측정   │     │
│       │                   │              │  Pass 2: RAGAS 평가       │     │
│       ▼                   ▼              └─────────────┬─────────────┘     │
│  ┌──────────┐      ┌──────────────┐                    │                   │
│  │ Markdown │      │ qa_dataset   │                    ▼                   │
│  │  변환    │      │   .json      │      ┌───────────────────────────┐     │
│  └────┬─────┘      │ (20쌍)       │      │ 결과 산출물               │     │
│       │            └──────────────┘      │ ├── latency.csv (72개)   │     │
│       ▼                                  │ ├── ragas.csv (Top 10)   │     │
│  ┌──────────┐                            │ └── e2e_report.md        │     │
│  │ Parent   │                            └───────────────────────────┘     │
│  │ -Child   │                                                              │
│  │ 청킹     │                                                              │
│  └──────────┘                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3-Layer 교차 조합 다이어그램

```
          Layer 1              Layer 2              Layer 3
        Dense Model          Sparse Model       Retrieval Mode
      ┌─────────────┐     ┌───────────────┐   ┌──────────────────┐
      │  kosimcse    │     │ korean_bm25   │   │ hybrid           │
      │  (한국어)    │──┐  │ (OKt 형태소)  │──┐│ (기본)           │
      ├─────────────┤  │  ├───────────────┤  ││                  │
      │  e5         │  │  │ splade        │  │├──────────────────┤
      │  (다국어)    │──┼──│ (학습 희소)   │──┼│ +colbert_rerank  │
      ├─────────────┤  │  ├───────────────┤  ││ (2-stage 리랭킹) │
      │  bge-m3     │  │  │ fastembed_bm25│  │├──────────────────┤
      │  (올인원)    │──┼──│ (네이티브)    │──┼│ +flashrank       │
      ├─────────────┤  │  └───────────────┘  ││ (ONNX 경량)      │
      │  minilm     │  │       3종           │├──────────────────┤
      │  (경량)      │──┘                    ┘│ +contextual      │
      └─────────────┘                         │ (LLM 문맥 부착)  │
           4종                                ├──────────────────┤
                                              │ +colbert+ctx     │
        ─── × ──────── × ───────────────────→ ├──────────────────┤
        4개    3개        6개 = 72개 조합      │ +flashrank+ctx   │
                                              └──────────────────┘
                                                    6종
```

## 2-Pass 실행 전략

```
┌─ Pass 1 ─────────────────────────────────────────────────────────────┐
│                                                                      │
│  72개 전략 × 20 쿼리 = 1,440회 검색                                   │
│  ─────────────────────────────────────────                           │
│  측정 항목: 레이턴시 (ms)                                              │
│  API 비용: $0 (로컬 검색만)                                            │
│                                                                      │
│  결과: all_combos_latency.csv                                        │
│        ┌──────────────────────────────────────────┐                  │
│        │ #1 minilm+fastembed_bm25       0.045s    │ ─┐              │
│        │ #2 minilm+fastembed_bm25+flash 0.052s    │  │              │
│        │ #3 kosimcse+fastembed_bm25     0.089s    │  │ 상위 10개    │
│        │ ...                                      │  │ 선별         │
│        │ #10 bge-m3+splade+flashrank    0.234s    │ ─┘              │
│        │ ─────── 여기서 컷 ────────              │                  │
│        │ #11 ... (RAGAS 평가 안 함)               │                  │
│        │ #72 ...                                  │                  │
│        └──────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Pass 2 ─────────────────────────────────────────────────────────────┐
│                                                                      │
│  상위 10개 전략 × 20 쿼리 = 200회 평가                                │
│  ─────────────────────────────────────────                           │
│  측정 항목: RAGAS 4개 메트릭                                          │
│  API 비용: ~$2-5 (GPT-4o-mini)                                      │
│                                                                      │
│  결과: all_combos_ragas.csv + e2e_report.md                          │
└──────────────────────────────────────────────────────────────────────┘
```

## 설계 의도

### 왜 만들었는가

한국어 RAG 파이프라인을 구축할 때, 어떤 임베딩 모델 + 희소 검색 + 리랭커 조합이 최적인지 객관적으로 판단하기 어렵습니다. 이 프로젝트는 **동일한 문서, 동일한 QA, 동일한 평가 기준**으로 모든 조합을 자동 비교하여 데이터 기반 의사결정을 지원합니다.

### 핵심 설계 원칙

1. **Strategy Pattern**: 모든 RAG 전략이 `BaseRAGStrategy` ABC를 구현. 새 전략 추가 시 인터페이스만 맞추면 자동으로 벤치마크에 편입.
2. **3-Layer 교차 조합**: Dense Model(4종) × Sparse Model(3종) × Retrieval Mode(6종) = 72개 조합을 체계적으로 탐색.
3. **2-Pass 실행**: Pass 1(레이턴시 스크리닝) → Pass 2(상위 N개만 RAGAS 평가)로 API 비용을 90% 절감.
4. **인덱스 캐싱**: 동일 (Dense, Sparse) 쌍은 Qdrant 인덱스를 재사용하여 72개 중 실제 인덱싱은 12회만 수행.

## 현재 구현 상태

### 구현 완료

| 구분 | 항목 | 상태 |
|------|------|:----:|
| **전략** | DenseSparse 4종 (KoSimCSE, E5, BGE-M3, MiniLM) | 완료 |
| **전략** | ColBERT Late Interaction (PyLate) | 완료 |
| **전략** | ColBERT 2-stage Reranking | 완료 |
| **전략** | FlashRank 경량 Reranking (ONNX) | 완료 |
| **전략** | Contextual Retrieval (LLM 문맥 부착) | 완료 |
| **전략** | GraphRAG (LightRAG) | 완료 |
| **파이프라인** | PDF → Markdown 변환 | 완료 |
| **파이프라인** | Parent-Child 청킹 | 완료 |
| **파이프라인** | QA 데이터셋 자동 생성 (GPT-4o-mini) | 완료 |
| **벤치마크** | 72개 3-Layer 교차 조합 파이프라인 | 완료 |
| **벤치마크** | 2-Pass 실행 (레이턴시 → RAGAS) | 완료 |
| **벤치마크** | 레이어별 기여도 분석 | 완료 |
| **평가** | RAGAS v0.4+ 통합 (4개 메트릭) | 완료 |
| **에이전트** | LangGraph Agentic RAG 대화 | 완료 |

### 3-Layer 조합 구조

```
Layer 1: Dense Model ──── kosimcse │ e5 │ bge-m3 │ minilm        (4종)
Layer 2: Sparse Model ─── korean_bm25 │ splade │ fastembed_bm25  (3종)
Layer 3: Retrieval Mode ─ hybrid × reranker × llm_support        (6종)
                           ├── hybrid (기본)
                           ├── hybrid + contextual
                           ├── hybrid + colbert_rerank
                           ├── hybrid + colbert_rerank + contextual
                           ├── hybrid + flashrank_rerank
                           └── hybrid + flashrank_rerank + contextual

총 유효 조합: 4 × 3 × 6 = 72개
```

## 비교 대상 임베딩 모델

| # | 임베딩 모델 | 특징 | 차원 | 한국어 |
|---|-----------|------|------|:------:|
| 1 | BM-K/KoSimCSE-roberta-multitask | 한국어 특화 SimCSE | 768 | ★★★ |
| 2 | intfloat/multilingual-e5-large | 다국어 균형 | 1024 | ★★ |
| 3 | BAAI/bge-m3 | 올인원 통합 (Dense+Sparse) | 1024 | ★★ |
| 4 | sentence-transformers/all-MiniLM-L6-v2 | 경량/빠름 | 384 | ★ |

추가로 BM25(한국어 토크나이저), SPLADE, Hybrid Retrieval, ColBERT Rerank, FlashRank Rerank, Contextual Retrieval 조합을 교차 비교합니다.

---

## 실행 가이드

### 사전 요구사항

| 항목 | 필수 여부 | 설명 |
|------|----------|------|
| **Python 3.12+** | 필수 | `.python-version` 파일에 명시 |
| **uv** | 필수 | Python 패키지 매니저 ([설치](https://docs.astral.sh/uv/getting-started/installation/): `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| **OpenAI API Key** | RAGAS 평가 시 | GPT-4o-mini 기반 평가 + QA 생성 |
| **Java JDK** | KoNLPy 사용 시 | OKt 형태소 분석기 (Combo 1: KoSimCSE+BM25) |

### Step 1: 환경 설정

```bash
# 1-1. 의존성 설치
uv sync

# 1-2. 환경변수 설정 (.env 파일 생성)
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

### Step 2: QA 데이터셋 생성

벤치마크 대상 문서(`rag_bench/docs/*.md`)에서 QA 쌍을 자동 생성합니다.

```bash
uv run python -m rag_bench.scripts.generate_qa --num_qa 20
```

생성 결과: `rag_bench/_benchdata/qa_dataset.json`

### Step 3: 벤치마크 실행

#### A. 72개 조합 전체 벤치마크 (권장)

```bash
# dry-run: 72개 조합 목록 미리보기
uv run python -m rag_bench.scripts.run_all_combos --preset full --dry-run

# 실제 실행: Pass 1(레이턴시) → Pass 2(상위 10개 RAGAS)
uv run python -m rag_bench.scripts.run_all_combos \
    --preset full \
    --top_n 10 \
    --k 3 \
    --layers
```

#### B. 빠른 검증 (4개 조합)

```bash
uv run python -m rag_bench.scripts.run_all_combos --preset quick --pass1-only
```

#### C. 레이턴시만 측정 (RAGAS 없이)

```bash
uv run python -m rag_bench.scripts.run_all_combos --preset full --pass1-only
```

### Step 4: 결과 확인

```bash
# CSV 결과 확인
ls rag_bench/_benchdata/all_combos_*.csv

# 리포트 확인
cat rag_bench/_benchdata/e2e_report.md
```

---

## 벤치마크 설정

### run_all_combos.py — 새 모드 (3-Layer 조합)

```
--preset PRESET      프리셋 선택: quick(4) | standard(24) | full(72)
--top_n N            Pass 1 후 상위 N 조합만 RAGAS 평가
--pass1-only         레이턴시만 측정 (RAGAS 없음)
--dry-run            조합 목록만 출력 (실행 안 함)
--layers             레이어별 기여도 분석 출력
--k K                검색 결과 수 (기본: 3)
--no_ragas           RAGAS 평가 건너뛰기
--reindex            기존 인덱스 삭제 후 재인덱싱
```

### run_all_combos.py — 레거시 모드

```
--combos 1,3,4       DenseSparse 조합 ID 지정
--skip_colbert       ColBERT 단독 전략 건너뛰기
--skip_rerank        ColBERTRerank 전략 건너뛰기
--skip_graphrag      GraphRAG 전략 건너뛰기
--skip_contextual    Contextual Retrieval 건너뛰기
--skip_flashrank     FlashRank Rerank 건너뛰기
--contextual_base N  Contextual Retrieval 기반 조합 ID (기본: 3)
```

## 산출물

벤치마크 실행 후 `rag_bench/_benchdata/`에 생성되는 파일:

| 파일 | 설명 |
|------|------|
| `qa_dataset.json` | QA 데이터셋 (질문-정답 쌍) |
| `all_combos_latency.csv` | 72개 전략 레이턴시 측정 결과 |
| `all_combos_ragas.csv` | 상위 N개 RAGAS 평가 점수 |
| `e2e_report.md` | 종합 리포트 (레이턴시 Top 10 + RAGAS) |

## 주요 결과 요약 (레거시 10종, 20 QA)

**RAGAS 품질 (상위 5):**

| 전략 | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|------|:-:|:-:|:-:|:-:|
| Rerank-DS3 (BGE-M3) | **0.7592** | 0.7639 | 0.9500 | **1.0000** |
| DS3 BGE-M3 | 0.7317 | **0.8647** | 0.9250 | 0.9250 |
| Rerank-DS1 | 0.7258 | 0.8161 | 0.9500 | 0.9250 |
| Rerank-DS4 | 0.6917 | 0.7632 | 0.8000 | 0.8750 |
| Rerank-DS2 | 0.6275 | 0.7240 | **0.9917** | 0.9750 |

## 프로젝트 구조

```
.
├── README.md                          # 프로젝트 개요 (이 파일)
├── pyproject.toml                     # 의존성 정의
├── uv.lock                            # 의존성 잠금
├── .env                               # 환경변수 (OPENAI_API_KEY 등)
├── docs/                              # 평가 대상 원본 PDF 문서
│   ├── 20250910_AI 현황 보고서.pdf
│   └── SPRi AI Brief_1월호.pdf
├── rag_bench/                         # 핵심 패키지 (상세: rag_bench/README.md)
│   ├── base.py                        # BaseRAGStrategy ABC
│   ├── config.py                      # 전역 설정
│   ├── runner.py                      # BenchmarkRunner
│   ├── cli.py                         # RAGChat 대화 인터페이스
│   ├── strategies/                    # RAG 전략 6종
│   │   ├── dense_sparse.py            # Dense+Sparse Hybrid (4종 임베딩)
│   │   ├── colbert.py                 # ColBERT Late Interaction
│   │   ├── colbert_rerank.py          # ColBERT 2-stage Reranking
│   │   ├── flashrank_rerank.py        # FlashRank 경량 Reranking
│   │   ├── contextual_retrieval.py    # Contextual Retrieval
│   │   └── graph_rag.py              # GraphRAG (LightRAG)
│   ├── indexing/                      # 문서 처리 파이프라인
│   │   ├── pdf_converter.py           # PDF → Markdown
│   │   └── chunker.py                # Parent-Child 청킹
│   ├── evaluation/                    # RAGAS 평가
│   ├── graph/                         # LangGraph 에이전트
│   ├── scripts/                       # 벤치마크 실행 스크립트
│   │   ├── generate_qa.py             # QA 자동 생성
│   │   ├── run_bench.py               # 3종 벤치마크
│   │   └── run_all_combos.py          # 72개 조합 벤치마크
│   ├── docs/                          # 벤치마크 대상 Markdown 문서
│   └── _benchdata/                    # 산출물 (.gitignore)
└── scripts/                           # 환경 검증 스크립트
    ├── verify_env.py
    ├── verify_rag_bench.py
    └── verify_ragas_eval.py
```

## 평가 메트릭

| 메트릭 | 분류 | 설명 |
|--------|------|------|
| **Context Precision** | Retrieval | 검색된 문서 중 관련 문서 비율 |
| **Context Recall** | Retrieval | 필요한 정보가 검색 결과에 포함된 정도 |
| **Faithfulness** | Generation | 답변이 검색 문서 내용에 충실한 정도 |
| **Answer Relevancy** | Generation | 답변이 질문에 적합한 정도 |

## 트러블슈팅

### uv가 설치되어 있지 않음
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### konlpy 관련 오류 (ko_okt 토크나이저)
Java JDK 설치 필요: macOS: `brew install openjdk`, Ubuntu: `sudo apt install default-jdk`

### OpenAI API 오류
`OPENAI_API_KEY` 환경변수가 올바르게 설정되어 있는지 확인 (`.env` 파일 또는 `export`)

### 사설 CA 인증서 (기업 네트워크)
```bash
export SSL_CERT_FILE="/path/to/your/ca-bundle.pem"
```

### Qdrant 파일 잠금 오류
이전 실행이 비정상 종료된 경우, 잠금 파일이 남아있을 수 있습니다:
```bash
rm -rf rag_bench/_benchdata/qdrant_db_*
```
