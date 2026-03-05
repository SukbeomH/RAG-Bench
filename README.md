# AutoRAG - 한국어 RAG 파이프라인 비교 평가

한국어 문서(PDF)를 대상으로 **PDF 파싱 → RAG 파이프라인** 전 과정의 성능을 정량 평가하는 모노레포 프로젝트입니다.

**uv workspace 기반 모노레포** 구조로, 6개 Python 패키지 + Next.js 프론트엔드로 구성됩니다. PDF 파서 벤치마크(OmniDocBench NED/TEDS)와 RAG 전략 벤치마크(RAGAS)를 LangGraph StateGraph 파이프라인으로 통합 실행합니다.

---

## 패키지 구조

| 패키지 | import 이름 | 역할 |
|---|---|---|
| `pdf-parsers` | `autorag_parsers` | PDF→Markdown, Protocol+Registry, bbox, chunking, provenance |
| `pdf-eval` | `autorag_pdf_eval` | NED/TEDS 벤치마크 평가, 마크다운 정규화, 보고서 생성 |
| `rag-retrieval` | `autorag_retrieval` | Dense+Sparse 하이브리드, ColBERT/FlashRank reranking |
| `rag-eval` | `autorag_rag_eval` | RAGAS 평가 메트릭 + RAG 벤치마크 보고서 생성 |
| `rag-api` | `autorag_api` | FastAPI (/api/parse, /api/ask) |
| `rag-pipeline` | `autorag_pipeline` | LangGraph StateGraph 기반 파이프라인 (RAG, 벤치마크) |
| `rag-frontend` | — | Next.js 15 + Tailwind |

### 패키지 의존성

```
parsers(독립) → pdf-eval, retrieval(독립) → rag-eval
api → parsers + retrieval
pipeline → parsers + retrieval
```

---

## LangGraph 파이프라인

3개의 StateGraph 기반 파이프라인을 제공합니다.

### 1. RAG Pipeline

```
START → parse → chunk → [index_ready?] → retrieve → generate → END
                              ↓
                        build_and_index
```

### 2. RAG Bench Pipeline

```
START → [prep_done?] → load_hf → enrich → run_benchmark → evaluate_ragas → END
              ↓
        run_benchmark (skip prep) → evaluate_ragas → END
```

### 3. PDF Bench Pipeline

```
START → [skip_parse?] ─── False → parse_all_pdfs → normalize_pdfs → evaluate_pdfs → collect_summary → END
                    └── True  → normalize_pdfs → evaluate_pdfs → collect_summary → END
```

`skip_parse=True`로 기존 파싱 결과에 **정규화만 재적용** 후 재평가가 가능합니다.

---

## PDF 파서 벤치마크

OmniDocBench (CVPR 2025) 프레임워크 기반으로 PDF 파서 솔루션을 비교합니다.

### 평가 지표

| 지표 | 의미 | 범위 |
|---|---|---|
| **NED** | 텍스트 정확도 (Normalized Edit Distance) | 0~1 |
| **BLEU** | n-gram 정밀도 | 0~100 |
| **METEOR** | 정밀도+재현율+어순 | 0~100 |
| **TEDS-HTML** | HTML 기반 표 구조 정확도 (IBM) | 0~1 |

### 지원 백엔드

| 백엔드 | 유형 | 특징 |
|---|---|---|
| `pymupdf` | 로컬 (규칙 기반) | 극도로 빠름, 이미지 PDF 처리 불가 |
| `docling` | 로컬 (OCR 파이프라인) | IBM Research, 레이아웃+OCR+표 인식 통합 |
| `openai` | API (VLM) | GPT-4o 페이지별 이미지 변환 |
| `upstage` | API (Document Parse) | 문서 파싱 전용 API |
| `paddleocr-vl` | 로컬 (VLM) | PaddleOCR-VL-1.5 0.9B, OmniDocBench SOTA |
| `deepseek-ocr2` | 로컬 (VLM) | DeepSeek-OCR-2, GPU 필수 |

### 마크다운 정규화

파서 출력과 GT 간 서식 차이를 통일하여 공정한 비교를 보장합니다. 7개 규칙을 순서대로 적용:

1. `code_block_wrapper` — VLM 코드블록 래퍼 제거
2. `vlm_location_tokens` — `<|LOC_XX|>` 위치 토큰 제거
3. `bullet_markers` — 불릿 기호 통일 (*, +, • → -)
4. `bold_in_headers` — 헤더 내 볼드 마커 제거
5. `blockquote_markers` — 인용구 마커 제거
6. `table_whitespace` — 테이블 셀 공백 정규화
7. `whitespace` — 연속 공백/줄바꿈 정규화

### 실행

```bash
# 전체 벤치마크 (파싱 + 정규화 + 평가 + 보고서)
uv run python -m autorag_pdf_eval.runner --preset phase1

# 기존 결과에 정규화만 재적용 + 재평가
uv run python -m autorag_pdf_eval.runner --reeval-only --results-dir ./bench_results/20260305-1528

# 보고서만 생성
uv run python -m autorag_pdf_eval.runner --report-only --results-dir ./bench_results/20260305-1528
```

---

## RAG 전략 벤치마크

### 3-Layer 교차 조합

```
Layer 1: Dense Model ──── kosimcse │ e5 │ bge-m3 │ openai-large │ upstage  (5종)
Layer 2: Sparse Model ─── korean_bm25 │ splade                              (2종)
Layer 3: Retrieval Mode ─ hybrid × reranker × llm_support                   (6종)
                           ├── hybrid (기본)
                           ├── hybrid + contextual
                           ├── hybrid + colbert_rerank
                           ├── hybrid + colbert_rerank + contextual
                           ├── hybrid + flashrank_rerank
                           └── hybrid + flashrank_rerank + contextual

총 유효 조합: 5 × 2 × 6 = 60개
```

### 2-Pass 실행 전략

- **Pass 1**: 60개 전략 × N 쿼리 → 레이턴시 측정 (API 비용 $0)
- **Pass 2**: 상위 N개 전략만 RAGAS 4개 메트릭 평가 (GPT-4o-mini, ~$2-5)

```bash
# 60개 조합 전체 벤치마크
uv run python -m rag_bench.scripts.run_all_combos --preset full --top_n 10 --k 3 --layers

# 빠른 검증 (2개 조합)
uv run python -m rag_bench.scripts.run_all_combos --preset quick --pass1-only
```

---

## 프로젝트 구조

```
.
├── packages/                          # uv workspace 패키지
│   ├── pdf-parsers/                   # PDF→Markdown 변환 (Registry+Protocol)
│   ├── pdf-eval/                      # PDF 파서 벤치마크 평가
│   │   └── src/autorag_pdf_eval/
│   │       ├── runner.py              # 벤치마크 러너 (run_spec, reeval_spec, reeval_dir)
│   │       ├── evaluator.py           # NED/TEDS 평가
│   │       ├── normalize.py           # 마크다운 정규화 (7개 규칙)
│   │       ├── report.py              # 보고서 자동 생성
│   │       ├── spec.py                # BenchSpec + 프리셋
│   │       └── omnidoc_metrics.py     # OmniDocBench 메트릭 (BLEU, METEOR, TEDS)
│   ├── rag-retrieval/                 # Dense+Sparse 하이브리드 검색
│   ├── rag-eval/                      # RAGAS 평가 + 보고서
│   ├── rag-api/                       # FastAPI 서버
│   ├── rag-pipeline/                  # LangGraph StateGraph 파이프라인
│   │   └── src/autorag_pipeline/
│   │       ├── states/                # TypedDict 상태 (RAGState, RAGBenchState, PDFBenchState)
│   │       ├── nodes/                 # 노드 함수 (parse, index, retrieve, generate, pdf_bench)
│   │       ├── graphs/                # 컴파일된 그래프 (rag_pipeline, rag_bench, pdf_bench)
│   │       └── integration/           # FastAPI v2 어댑터 + CLI
│   └── rag-frontend/                  # Next.js 15 + Tailwind
├── servers/                           # OCR 서버 (독립 uv 프로젝트)
│   ├── paddleocr_vl/                  # PPStructureV3 (CPU)
│   ├── mlx_vlm/                       # PaddleOCR-VL-1.5 mlx-vlm (macOS)
│   ├── deepseek_ocr2/                 # DeepSeek-OCR-2 (CUDA)
│   └── got_ocr2/                      # GOT-OCR2.0
├── orchestrators/                     # K8s 오케스트레이터 (rag_bench, pdf_parser)
├── deploy/k8s/                        # Dockerfiles, manifests, scripts
├── isolated_backends/                 # 격리 venv 백엔드 (docling, paddleocr, deepseek)
├── rag_bench/                         # 레거시 RAG 벤치마크 (strategies, combo, scripts)
├── docs/                              # 원본 PDF 문서 + 리서치
└── bench_results/                     # PDF 파서 벤치마크 결과 (.gitignore)
```

---

## 핵심 아키텍처 패턴

| 패턴 | 설명 |
|---|---|
| **Registry + Lazy Loading** | `@register()` 데코레이터, `get_parser()` 시 지연 import |
| **Bridge/Worker Subprocess** | venv python → stdout JSON 통신 (docling, paddleocr, deepseek) |
| **LangGraph StateGraph** | 노드=기존 패키지 래핑, 조건부 라우팅 (`skip_parse`, `index_ready`, `prep_done`) |
| **Strategy Pattern** | 모든 RAG 전략이 `BaseRAGStrategy` ABC 구현 |
| **2-Phase K8s 벤치마크** | Phase1(prep) → Phase2(bench) → 결과 수집 |
| **OCR 서버 공통** | FastAPI + OpenAI-compat API, 비동기 모델 로드, `/health` 헬스체크 |

---

## 사전 요구사항

| 항목 | 필수 여부 | 설명 |
|------|----------|------|
| **Python 3.12+** | 필수 | `.python-version` 파일에 명시 |
| **uv** | 필수 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **OpenAI API Key** | RAGAS 평가 시 | GPT-4o-mini 기반 평가 + QA 생성 |
| **Upstage API Key** | Upstage 전략 사용 시 | `UPSTAGE_API_KEY` 환경변수 |

```bash
# 의존성 설치
uv sync

# 환경변수 설정
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

---

## 테스트

```bash
# 전체 테스트 (185개: 단위 77 + E2E 67 + 보고서 19 + 파이프라인 22)
uv run pytest packages/*/tests/ -v

# 특정 패키지만
uv run pytest packages/pdf-eval/tests/ -v
uv run pytest packages/rag-pipeline/tests/ -v
```

---

## 트러블슈팅

### 사설 CA 인증서 (기업 네트워크)
```bash
export SSL_CERT_FILE="/path/to/your/ca-bundle.pem"
```

### Apple Silicon MPS OOM
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### HuggingFace 모델 다운로드 오류
```bash
export HF_HUB_DISABLE_XET=1
```

### 벤치마크 환경변수 미전달
```bash
# subprocess에도 전달되도록 export 필수
set -a && source .env && set +a
```
