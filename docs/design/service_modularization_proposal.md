# RAG Bench → 실 서비스 모듈화 방안

**작성일**: 2026-02-20
**대상 코드베이스**: `/autorag/rag_bench/`
**목적**: 벤치마크용 RAG 로직을 실 서비스 개발에서 재사용 가능한 블록으로 분리·문서화

---

## 1. 현황 분석

### 코드 구성 비율

```
재사용 가능 (70~80%)                    벤치마크 전용 (20~30%)
─────────────────────────────────────  ────────────────────────
base.py          — Strategy ABC        run_all_combos.py — 72조합 오케스트레이션
strategies/*.py  — 5개 전략 구현        runner.py         — 비교 실행기
indexing/*.py    — 청킹 · PDF 변환      colab_runner.py   — Colab 체크포인트
evaluation/*.py  — RAGAS 평가기        generate_qa.py    — 벤치마크 데이터 생성
run_tracker.py   — 관측성 추적          generate_html_report.py
```

### 전체 디렉토리 구조

```
autorag/
├── rag_bench/                          # CORE: 모듈형 RAG 벤치마크 프레임워크 (6,837 LOC)
│   ├── base.py                         # BaseRAGStrategy ABC (90 lines)
│   ├── config.py                       # 전역 설정, 모델 캐시, SSL 우회 (134 lines)
│   ├── runner.py                       # BenchmarkRunner (295 lines)
│   ├── run_tracker.py                  # RunTracker: 플랫폼 정보, 타이밍, 토큰 (447 lines)
│   ├── cli.py                          # RAGChat CLI 인터페이스 (94 lines)
│   │
│   ├── strategies/                     # RAG 전략 구현 5종
│   │   ├── dense_sparse.py             # Dense+Sparse 하이브리드 (476 lines)
│   │   ├── colbert.py                  # ColBERT Late Interaction (229 lines)
│   │   ├── colbert_rerank.py           # ColBERT 2단계 리랭킹 (201 lines, Decorator)
│   │   ├── flashrank_rerank.py         # FlashRank ONNX 리랭킹 (181 lines)
│   │   ├── contextual_retrieval.py     # Contextual Retrieval (329 lines)
│   │   ├── openai_embed.py             # OpenAI 임베딩 Dense 검색
│   │   └── upstage_embed.py            # Upstage Solar 임베딩 Dense 검색
│   │
│   ├── evaluation/                     # RAGAS v0.4 평가 서브패키지
│   │   ├── evaluator.py                # ExtendedRAGEvaluator (다관점 역질문 생성)
│   │   ├── metrics.py                  # MetricRegistry, MetricPreset
│   │   └── legacy.py                   # 하위호환 shim
│   │
│   ├── indexing/                       # 문서 처리 파이프라인
│   │   ├── pdf_converter.py            # PDF → Markdown (pymupdf4llm)
│   │   └── chunker.py                  # Parent-Child 청킹
│   │
│   ├── graph/                          # LangGraph 에이전틱 RAG
│   │   ├── builder.py
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── prompts.py
│   │
│   └── scripts/                        # 벤치마크 실행 스크립트 (전용)
│       ├── generate_qa.py
│       ├── run_all_combos.py
│       ├── run_bench.py
│       ├── generate_html_report.py
│       └── prefetch_models.py
│
└── rag_bench_colab/                    # (삭제됨) Google Colab 벤치마크 환경
```

---

## 2. 아키텍처 패턴 분석

### 핵심 디자인 패턴

| 패턴 | 모듈 | 목적 |
|------|------|------|
| **Strategy Pattern** | `base.py`, `strategies/*.py` | 통일된 `BaseRAGStrategy` 인터페이스 |
| **Decorator Pattern** | `colbert_rerank.py`, `flashrank_rerank.py`, `contextual_retrieval.py` | base_strategy 래핑으로 조합 |
| **Singleton Cache** | `run_all_combos.py:IndexCacheManager` | ColBERT 모델, FlashRank ONNX, Qdrant 클라이언트 재사용 |
| **Context Manager** | `run_tracker.py:phase()`, `track_openai_tokens()` | 시간/토큰 추적, 메모리 정리 |
| **Lazy Initialization** | 각 전략 `_ensure_initialized()` | LLM/모델 최초 사용 시 로드 |
| **Pydantic Structured Output** | `evaluator.py:_MultiPerspectiveOutput` | RAGAS 역질문 생성 단일 API 호출 |

### 데이터 흐름

```
PDF / Markdown 문서
  ↓
pdf_converter.py : PDF → Markdown
  ↓
chunker.py : Parent-Child Chunks (LangChain Documents)
  ↓
strategy.index(chunks)
  ├─→ DenseSparseStrategy → Qdrant (Dense + Sparse 벡터)
  ├─→ ColBERTStrategy     → PyLate (토큰 레벨 인코딩)
  ├─→ ContextualRetrieval → GPT-4o-mini (문맥 prefix 부착)
  └─→ (Reranker Decorator) → 기존 전략 위에 2단계 적용
        ↓
    strategy.retrieve(query, k=3)
        ↓
    RAGAS Evaluator
        ├─→ Core 4 (Faithfulness, AnswerRelevancy, CtxPrecision, CtxRecall)
        └─→ Extended 5 + Lightweight 2
```

---

## 3. 재사용 가능 블록 vs 벤치마크 전용 코드

### 재사용 가능 블록 (서비스 추출 가능)

| 블록 | 위치 | 서비스 준비도 | 비고 |
|------|------|:---:|------|
| **BaseRAGStrategy ABC** | `base.py` | ✅ 즉시 | 변경 불필요 |
| **Parent-Child Chunker** | `indexing/chunker.py` | ✅ 즉시 | LangChain TextSplitters |
| **PDF 변환기** | `indexing/pdf_converter.py` | ✅ 즉시 | pymupdf4llm 래퍼 |
| **DenseSparseStrategy** | `strategies/dense_sparse.py` | ✅ 즉시 | 설정 주입 필요 |
| **ColBERTStrategy** | `strategies/colbert.py` | ✅ 즉시 | PyLate 백엔드 |
| **ColBERTRerankStrategy** | `strategies/colbert_rerank.py` | ✅ 즉시 | Decorator 패턴 |
| **FlashRankRerankStrategy** | `strategies/flashrank_rerank.py` | ✅ 즉시 | ONNX, CPU-only |
| **ContextualRetrievalStrategy** | `strategies/contextual_retrieval.py` | ✅ 즉시 | OpenAI API 필요 |
| **OpenAIEmbedStrategy** | `strategies/openai_embed.py` | ✅ 즉시 | OPENAI_API_KEY |
| **UpstageEmbedStrategy** | `strategies/upstage_embed.py` | ✅ 즉시 | UPSTAGE_API_KEY |
| **ExtendedRAGEvaluator** | `evaluation/evaluator.py` | ⚠️ 부분 | LLM 모델 하드코딩 |
| **RunTracker** | `run_tracker.py` | ✅ 즉시 | 범용 LLM 파이프라인 관측 |

### 벤치마크 전용 코드 (서비스에서 불필요)

| 코드 | 위치 | 문제점 |
|------|------|------|
| **BenchmarkRunner** | `runner.py` | 비교 실행 로직, RAGAS 통합에 결합 |
| **run_all_combos.py** | `scripts/` | 72조합 오케스트레이션, IndexCacheManager 결합 |
| **colab_runner.py** | `rag_bench_colab/` (삭제됨) | Colab 체크포인트, Google Drive, monkey-patching |
| **generate_qa.py** | `scripts/` | 벤치마크 데이터 생성 전용 |
| **ComboSpec / IndexCacheManager** | `run_all_combos.py` | 3-Layer 벤치마크 전용 구조 |

---

## 4. 현재 코드 품질 평가

### 강점

1. **우수한 모듈성**: BaseRAGStrategy ABC가 간결하고 명확하게 정의됨
2. **포괄적 에러 처리**: SSL 우회, MPS OOM, Qdrant 파일 잠금 모두 대응
3. **실용적 설계**: 기업 네트워크 지원, GPU/CPU 자동 감지, Apple Silicon 처리
4. **관측성**: RunTracker, 단계별 타이밍, 토큰 추적 내장
5. **성능 최적화**: Singleton 캐싱, Pass 1→2 결과 주입, 배치 처리
6. **문서화**: MEMORY.md, README.md, 인라인 docstring 충실

### 기술 부채

1. **하드코딩된 모델 목록**: `config.py`의 `REQUIRED_HF_MODELS`, `evaluator.py`의 LLM 모델
2. **벤치마크 결합**: `run_all_combos.py`의 `IndexCacheManager`가 3-Layer 로직에 묶여 있음
3. **타입 힌트 불완전**: 일부 모듈 타입 커버리지 미흡
4. **테스트 부재**: `tests/` 디렉토리 없음
5. **Colab monkey-patching**: 라이브러리 업데이트 시 취약

---

## 5. 방안 비교

### 방안 A — `rag_sdk/` 별도 패키지 신규 생성

```
autorag/
├── rag_bench/      ← 기존 유지 (벤치마크 전용)
└── rag_sdk/        ← 신규 (서비스용 SDK)
    ├── __init__.py
    ├── strategies/ ← rag_bench/strategies에서 re-export + 설정 주입
    ├── pipeline.py ← 단일 진입점 파이프라인
    ├── config.py   ← 외부화된 설정 (YAML/env)
    └── cookbook/   ← 코드 스니펫 문서
```

| 항목 | 평가 |
|------|------|
| 기존 코드 안전성 | ✅ 100% 보전 |
| 유지보수 부담 | ❌ 두 곳 동시 관리 |
| 코드 이중화 | ❌ re-export 레이어 필요 |
| 독립 버전 관리 | ✅ 가능 |

---

### 방안 B — `rag_bench/` 직접 수정 (Refactor-in-place)

```
rag_bench/
├── base.py
├── strategies/
│   └── *.py       ← 설정 외부화 (하드코딩 제거)
├── config/        ← 신규: YAML 기반 설정 관리
│   ├── schema.py
│   └── loader.py
└── cookbook/      ← 신규: 스니펫 + 가이드 문서
```

| 항목 | 평가 |
|------|------|
| 기존 코드 안전성 | ⚠️ 리팩터 중 기존 스크립트 깨질 위험 |
| 단일 소스 | ✅ 중복 없음 |
| 즉시 실행 가능 | ❌ 리팩터 선행 필요 |
| 역할 경계 명확성 | ⚠️ 벤치마크 vs 서비스 여전히 혼재 |

---

### 방안 C — 하이브리드: 구조 정비 + Cookbook ⭐ 권장

```
autorag/
├── rag_bench/             ← 최소 수정 (벤치마크 역할 유지)
│   ├── base.py            ← 변경 없음 (이미 서비스 레디)
│   ├── strategies/        ← 변경 없음 (이미 캡슐화됨)
│   ├── indexing/          ← 변경 없음
│   ├── evaluation/        ← 변경 없음
│   └── ...
│
└── cookbook/              ← 신규 (실 서비스용 블록 문서화)
    ├── README.md          ← 사용 가이드 및 블록 인덱스
    ├── 01_indexing.py     ← 문서 수집 → 청킹 → 인덱싱 스니펫
    ├── 02_retrieval.py    ← 검색 전략 선택 · 실행 스니펫
    ├── 03_reranking.py    ← 리랭커 적용 스니펫
    ├── 04_contextual.py   ← Contextual Retrieval 스니펫
    ├── 05_evaluation.py   ← RAGAS 평가 스니펫
    ├── 06_pipeline.py     ← 단일 통합 파이프라인 스니펫
    ├── 07_openai_embed.py ← OpenAI / Upstage 임베딩 스니펫
    └── configs/
        ├── quick_start.yaml   ← 최소 설정 예시
        └── production.yaml    ← 프로덕션 설정 예시
```

| 항목 | 평가 |
|------|------|
| 기존 코드 안전성 | ✅ 100% 보전 |
| 유지보수 부담 | ✅ 낮음 (단방향 의존) |
| 코드 이중화 | ✅ 없음 |
| 즉시 실행 가능 | ✅ 예 |
| 서비스 개발 속도 | ✅ 빠름 (복사·붙여넣기 즉시 사용) |

---

## 6. 최종 권장: 방안 C 2단계 실행 계획

### 1단계: `cookbook/` 디렉토리 생성 (즉시 실행 가능)

`rag_bench/` 코드를 **그대로 두고**, 실 서비스에서 복사해서 쓸 수 있는 코드 스니펫을 별도 문서화합니다.

각 파일 원칙:
- `rag_bench`를 라이브러리처럼 import (수정 없이)
- 완전히 독립 실행 가능 (`python cookbook/01_indexing.py`)
- 환경변수 / YAML 설정으로 모델·경로 주입
- 상단에 블록 설명, 하단에 실행 예시 포함

### 2단계: `rag_bench/config.py` 최소 수정 (선택적)

하드코딩된 모델 목록과 경로를 YAML 설정으로 분리.
기존 벤치마크 스크립트는 하위호환성 유지.

```python
# 현재 (하드코딩)
REQUIRED_HF_MODELS = ["BM-K/KoSimCSE-roberta-multitask", ...]

# 목표 (설정 주입)
REQUIRED_HF_MODELS = load_config("models.yaml").get("hf_models", [])
```

---

## 7. 서비스 준비도 결론

| 항목 | 현재 상태 | 평가 |
|------|----------|------|
| 모듈성 | 우수 (ABC 패턴, Decorator 패턴) | 서비스 추출 즉시 가능 |
| 코드 품질 | 높음 (포괄적 문서, 에러 처리, 최적화) | 프로덕션급 |
| 재사용성 | 70~80% 범용 코드 | 나머지 20~30%는 벤치마크 전용 오케스트레이션 |
| 하드코딩 문제 | 중간 (모델 목록, LLM 모델명) | 설정 외부화로 해결 가능 |
| 테스트 | 없음 | pytest 추가 필요 |
| API 레이어 | 없음 | REST API 추가 필요 (서비스화 시) |
| 클라우드 준비 | 로컬 전용 (Qdrant 파일, .env) | 클라우드 백엔드 마이그레이션 필요 |
| 문서화 | 우수 (MEMORY.md, README.md) | 포괄적, 한국어/영어 병행 |
| 성능 | 최적화됨 (캐싱, 배치, 지연 로딩) | 프로덕션 레디 |

**결론**: 이 코드베이스는 ~70%가 즉시 재사용 가능한 프로덕션급 벤치마크 프레임워크입니다.
나머지 20~30%는 오케스트레이션 레이어로, 설정 외부화 및 API/퍼시스턴스 레이어 추가가 주요 서비스화 작업입니다.
근본적인 아키텍처 변경 없이 `cookbook/` 접근으로 빠르게 서비스 개발을 시작할 수 있습니다.

---

*생성: Claude Sonnet 4.6 | 2026-02-20*
