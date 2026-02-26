# RAG Bench - 한국어 RAG 파이프라인 비교 평가

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SukbeomH/autorag/blob/main/rag_bench_colab/rag_benchmark.ipynb)

한국어 문서(PDF)를 대상으로 다양한 RAG 파이프라인 성능을 정량 평가하는 프로젝트입니다.

Strategy Pattern 기반 모듈화 벤치마크 시스템으로, RAGAS 평가를 통해 다양한 전략 조합을 통일된 인터페이스로 비교합니다. 로컬(60개 조합), K8s 병렬(2-Phase), Google Colab 세 가지 실행 환경을 지원합니다.

> **최근 변경**: K8s 2-Phase 병렬 벤치마크 시스템 구축 — 문서 카테고리별(GENERAL/LEGAL/BUSINESS/MEDICAL) 6개 Dense×Sparse 조합을 EKS 클러스터에서 병렬 실행. ColBERT Reranker + Contextual Retrieval 고정 파이프라인.

## 전체 흐름도

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RAG Bench 전체 파이프라인                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ① 문서 준비 + QA 데이터셋 생성                                              │
│                                                                              │
│  docs/*.pdf                                                                  │
│      │ [선택: --sample_pages]                                                │
│      │  pdfs_to_markdowns(ratio=10%, max=5pages)                             │
│      ▼                                                                       │
│  rag_bench/docs/*.md ──→ create_parent_child_chunks()                        │
│                                    │                                         │
│                                    ▼                                         │
│                       effective_num_qa                                       │
│                       = min(num_qa, chunks × max_qa_per_page)                │
│                                    │                                         │
│                                    ▼                                         │
│                       RAGAS KnowledgeGraph 구축                              │
│                       TestsetGenerator.generate()                            │
│                                    │                                         │
│                                    ▼                                         │
│                       _benchdata/qa_dataset.json                             │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ② 벤치마크 실행 (2-Pass)                                                    │
│                                                                              │
│  qa_dataset.json                                                             │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─ Pass 1 ─────────────────────────────────────────────────────────────┐   │
│  │  60개 전략 × N 쿼리 → 레이턴시 측정 (API 비용 없음)                  │   │
│  │  결과: all_combos_latency.csv  →  상위 top_n 전략 선별               │   │
│  └──────────────────────────────────┬─────────────────────────────────── ┘   │
│                                     ▼                                       │
│  ┌─ Pass 2 ─────────────────────────────────────────────────────────────┐   │
│  │  top_n 전략 × N 쿼리 → RAGAS 4개 메트릭 평가 (GPT-4o-mini)         │   │
│  │  결과: all_combos_ragas.csv  +  e2e_report.md                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ③ 결과 산출물                                                               │
│                                                                              │
│  _benchdata/                                                                 │
│  ├── all_combos_latency.csv    (60개 전략 레이턴시)                          │
│  ├── all_combos_ragas.csv      (top_n 전략 RAGAS 점수)                      │
│  ├── e2e_report.md             (텍스트 요약 보고서)                          │
│  └── benchmark_report.html    (인터랙티브 HTML 보고서)                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
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
      ├─────────────┤  │  └───────────────┘  ││ (2-stage 리랭킹) │
      │  bge-m3     │  │       2종           │├──────────────────┤
      │  (올인원)    │──┤                    ┤│ +flashrank       │
      ├─────────────┤  │                    ││ (ONNX 경량)      │
      │  openai-large│  │                    │├──────────────────┤
      │  (유료 API)  │──┤                    │ +contextual      │
      ├─────────────┤  │                    │ (LLM 문맥 부착)  │
      │  upstage    │  │                    │├──────────────────┤
      │  (유료 API)  │──┘                    │ +colbert+ctx     │
      └─────────────┘                        ├──────────────────┤
           5종                               │ +flashrank+ctx   │
                                             └──────────────────┘
        ─── × ──── × ───────────────────→         6종
        5개    2개   6개 = 60개 조합
```

## 2-Pass 실행 전략

```
┌─ Pass 1 ─────────────────────────────────────────────────────────────┐
│                                                                      │
│  60개 전략 × 20 쿼리 = 1,200회 검색                                   │
│  ─────────────────────────────────────────                           │
│  측정 항목: 레이턴시 (ms)                                              │
│  API 비용: $0 (로컬 검색만, HF 모델 기준)                              │
│                                                                      │
│  결과: all_combos_latency.csv                                        │
│        ┌──────────────────────────────────────────┐                  │
│        │ #1 kosimcse+korean_bm25        0.089s    │ ─┐              │
│        │ #2 kosimcse+korean_bm25+flash  0.102s    │  │              │
│        │ #3 bge-m3+korean_bm25          0.234s    │  │ 상위 10개    │
│        │ ...                                      │  │ 선별         │
│        │ #10 bge-m3+splade+flashrank    0.456s    │ ─┘              │
│        │ ─────── 여기서 컷 ────────              │                  │
│        │ #11 ... (RAGAS 평가 안 함)               │                  │
│        │ #60 ...                                  │                  │
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
2. **3-Layer 교차 조합**: Dense Model(5종) × Sparse Model(2종) × Retrieval Mode(6종) = 60개 조합을 체계적으로 탐색.
3. **2-Pass 실행**: Pass 1(레이턴시 스크리닝) → Pass 2(상위 N개만 RAGAS 평가)로 API 비용을 90% 절감.
4. **인덱스 캐싱**: 동일 (Dense, Sparse) 쌍은 Qdrant 인덱스를 재사용하여 60개 중 실제 인덱싱은 10회만 수행.

## 현재 구현 상태

### 구현 완료

| 구분 | 항목 | 상태 |
|------|------|:----:|
| **전략** | DenseSparse 5종 (KoSimCSE, E5, BGE-M3, OpenAI-large, Upstage) | 완료 |
| **전략** | ColBERT Late Interaction (PyLate) | 완료 |
| **전략** | ColBERT 2-stage Reranking | 완료 |
| **전략** | FlashRank 경량 Reranking (ONNX) | 완료 |
| **전략** | Contextual Retrieval (LLM 문맥 부착) | 완료 |
| **전략** | OpenAI Embedding (text-embedding-3-large) | 완료 |
| **전략** | Upstage Embedding (solar-embedding-1-query) | 완료 |
| **파이프라인** | PDF → Markdown 변환 | 완료 |
| **파이프라인** | Parent-Child 청킹 | 완료 |
| **파이프라인** | QA 데이터셋 자동 생성 (GPT-4o-mini / RAGAS KG) | 완료 |
| **벤치마크** | 60개 3-Layer 교차 조합 파이프라인 | 완료 |
| **벤치마크** | 2-Pass 실행 (레이턴시 → RAGAS) | 완료 |
| **벤치마크** | 레이어별 기여도 분석 | 완료 |
| **보고서** | HTML 벤치마크 보고서 자동 생성 (차트 + Bootstrap) | 완료 |
| **평가** | RAGAS v0.4+ 통합 (Core 4종 + Extended 5종 + Lightweight 2종) | 완료 |
| **에이전트** | LangGraph Agentic RAG 대화 | 완료 |
| **인프라** | K8s 2-Phase 병렬 벤치마크 (EKS) | 완료 |
| **인프라** | Google Colab T4 GPU 벤치마크 환경 | 완료 |
| **인프라** | HuggingFace 모델 로컬 캐시 (심링크) | 완료 |
| **최적화** | FlashRank 싱글톤 + LLM 병렬화 + SPLADE 배치 | 완료 |
| **추적** | 수행 이력 추적 (RunTracker — 플랫폼, 타이밍, 토큰, 비중%) | 완료 |

### 3-Layer 조합 구조

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

## 비교 대상 임베딩 모델

| # | 임베딩 모델 | 특징 | 차원 | 한국어 | 비고 |
|---|-----------|------|------|:------:|------|
| 1 | BM-K/KoSimCSE-roberta-multitask | 한국어 특화 SimCSE | 768 | ★★★ | HF |
| 2 | intfloat/multilingual-e5-large | 다국어 균형 | 1024 | ★★ | HF |
| 3 | BAAI/bge-m3 | 올인원 통합 (Dense+Sparse) | 1024 | ★★ | HF |
| 4 | text-embedding-3-large | OpenAI 고품질 | 3072 | ★★ | 유료 API |
| 5 | solar-embedding-1-query | Upstage 한국어 특화 | 4096 | ★★★ | 유료 API |

추가로 BM25(한국어 토크나이저), SPLADE, Hybrid Retrieval, ColBERT Rerank, FlashRank Rerank, Contextual Retrieval 조합을 교차 비교합니다.

---

## 실행 가이드

### 사전 요구사항

| 항목 | 필수 여부 | 설명 |
|------|----------|------|
| **Python 3.12+** | 필수 | `.python-version` 파일에 명시 |
| **uv** | 필수 | Python 패키지 매니저 ([설치](https://docs.astral.sh/uv/getting-started/installation/): `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| **OpenAI API Key** | RAGAS 평가 / QA 생성 시 | GPT-4o-mini 기반 평가 + QA 생성 + OpenAI 임베딩 전략 |
| **Upstage API Key** | Upstage 전략 사용 시 | `UPSTAGE_API_KEY` 환경변수 필요 ([발급](https://console.upstage.ai)) |
| **Java JDK** | KoNLPy 사용 시 | OKt 형태소 분석기 (Combo 1: KoSimCSE+BM25) |

### Step 1: 환경 설정

```bash
# 1-1. 의존성 설치
uv sync

# 1-2. 환경변수 설정 (.env 파일 생성)
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env

# Upstage 전략 사용 시 추가 설정
echo "UPSTAGE_API_KEY=up_your-api-key-here" >> .env
```

### Step 2: QA 데이터셋 생성

벤치마크 대상 문서(`rag_bench/docs/*.md`)에서 QA 쌍을 자동 생성합니다.

```bash
# 기본 (기존 rag_bench/docs/*.md 파일 사용)
uv run python -m rag_bench.scripts.generate_qa --num_qa 20

# PDF 페이지 샘플링 적용 (docs/*.pdf → rag_bench/docs/*.md 재변환, QA 수 자동 상한)
uv run python -m rag_bench.scripts.generate_qa --sample_pages --num_qa 20

# KG만 사전 구축 (QA 생성 없이)
uv run python -m rag_bench.scripts.generate_qa --build-kg-only

# 기존 KG 재사용하여 QA 생성
uv run python -m rag_bench.scripts.generate_qa --num_qa 50 --reuse-kg
```

생성 결과: `rag_bench/_benchdata/qa_dataset.json`

### Step 3: 벤치마크 실행

#### A. 60개 조합 전체 벤치마크 (권장)

```bash
# dry-run: 60개 조합 목록 미리보기
uv run python -m rag_bench.scripts.run_all_combos --preset full --dry-run

# 실제 실행: Pass 1(레이턴시) → Pass 2(상위 10개 RAGAS)
uv run python -m rag_bench.scripts.run_all_combos \
    --preset full \
    --top_n 10 \
    --k 3 \
    --layers
```

#### B. 빠른 검증 (2개 조합)

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

### Step 5 (선택): 보고서 및 시각화

```python
# HTML 벤치마크 보고서 생성 (Python API 방식)
import pandas as pd
from rag_bench.scripts.generate_html_report import generate_html_report

latency_df = pd.read_csv("rag_bench/_benchdata/all_combos_latency.csv")
ragas_df = pd.read_csv("rag_bench/_benchdata/all_combos_ragas.csv")
generate_html_report(latency_df, ragas_df, output_path="rag_bench/_benchdata/benchmark_report.html")
```

```bash
# 시각화 노트북 (7종 차트: 레이턴시 바, RAGAS 레이더, 품질-속도 Scatter 등)
uv run jupyter notebook rag_bench/scripts/bench_visualize.ipynb
```

### HuggingFace 모델 프리페치 (선택)

```bash
# 6종 모델 캐시 상태 확인
uv run python -m rag_bench.scripts.prefetch_models --status

# 모델 미리 다운로드 (오프라인 환경 대비)
uv run python -m rag_bench.scripts.prefetch_models
```

---

## Google Colab 실행

로컬 환경 없이 Google Colab T4 GPU에서 벤치마크를 실행할 수 있습니다.

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SukbeomH/autorag/blob/main/rag_bench_colab/rag_benchmark.ipynb) 클릭
2. **런타임 → 런타임 유형 변경 → GPU (T4)** 선택
3. Colab Secrets에 `OPENAI_API_KEY` 등록
4. 노트북 셀 순서대로 실행

### Colab 프리셋

| 프리셋 | 조합 수 | Dense | Sparse | 예상 시간 | API 비용 |
|--------|---------|-------|--------|----------|---------|
| `quick` | 2 | bge-m3 (1종) | korean_bm25 (1종) | ~10분 | ~$0.3 |
| `standard` | 20 | HF 3종 + 유료 2종 | 2종 | ~45분 | ~$2 |
| `full` | 60 | 5종 | 2종 | ~3시간 | ~$5 |

### Colab 특징

- **Monkey-patch 접근**: `rag_bench` 코어 코드 수정 없이 런타임 패치로 Colab 환경 대응 (CUDA 디바이스, Qdrant 인메모리 등)
- **체크포인트 시스템**: 전략별 JSON을 Google Drive에 저장. 12시간 세션 제한으로 커널이 재시작되어도 완료된 전략을 건너뛰고 이어서 실행
- **Qdrant 3모드**: `ephemeral` (로컬), `drive` (Google Drive 영속), `memory` (인메모리)

---

## K8s 병렬 벤치마크

EKS 클러스터에서 문서 카테고리별 벤치마크를 병렬 실행합니다. 서비스 모델 선정을 위한 대규모 벤치마크에 적합합니다.

### 실행 환경

| 항목 | 값 |
|------|------|
| 클러스터 | EKS (ap-northeast-2), 5노드 (~13 vCPU) |
| 프리셋 | `service` — 3 Dense(kosimcse, e5, bge-m3) × 2 Sparse = 6 조합 |
| 고정 파이프라인 | ColBERT Reranker + Contextual Retrieval |
| 카테고리 | GENERAL, LEGAL, BUSINESS, MEDICAL |
| 리소스 | Prep CPU 1/1, Bench CPU 1/2, Mem 4Gi/8Gi |

### 2-Phase 아키텍처

```
Phase 1 (Prep)                        Phase 2 (Bench)
──────────────                        ───────────────
카테고리당 1 Job (병렬)                카테고리 × 조합 Job (병렬)
  HF 데이터 로드                         Phase 1 데이터 역직렬화
  Parent-Child 청킹                      전략 빌드 (Dense+Sparse+ColBERT+Ctx)
  Contextual enrichment (LLM)            Pass 1: 레이턴시 측정
  결과 PVC에 직렬화                      Pass 2: RAGAS 평가
                                         결과 PVC에 직렬화
         ↓ PVC 가시성 검증 ↓
                                    → 오케스트레이터가 결과 수집 + 병합
```

### 실행 방법

```bash
# 이미지 빌드 + 푸시 (K8s 원격 빌더)
docker buildx build --builder k8s-amd64 --platform linux/amd64 --push \
    -t $HARBOR_REGISTRY/rag-bench-test/worker:latest -f k8s/Dockerfile .

# 전체 벤치마크 실행
python3 k8s/orchestrator.py --image $IMAGE

# 특정 카테고리만
python3 k8s/orchestrator.py --image $IMAGE --categories general,legal

# 데이터 크기 제한 (테스트용)
python3 k8s/orchestrator.py --image $IMAGE --categories general \
    --max-corpus 1000 --max-queries 50
```

상세 배포 가이드: [`k8s/DEPLOY_GUIDE.md`](k8s/DEPLOY_GUIDE.md)

---

## 벤치마크 설정

### run_all_combos.py — 3-Layer 조합 모드

```
--preset PRESET      프리셋 선택: quick(2) | standard(20) | full(60)
--top_n N            Pass 1 후 상위 N 조합만 RAGAS 평가
--pass1-only         레이턴시만 측정 (RAGAS 없음)
--dry-run            조합 목록만 출력 (실행 안 함)
--layers             레이어별 기여도 분석 출력
--k K                검색 결과 수 (기본: 3)
--no_ragas           RAGAS 평가 건너뛰기
--reindex            기존 인덱스 삭제 후 재인덱싱
```

## 산출물

벤치마크 실행 후 `rag_bench/_benchdata/`에 생성되는 파일:

| 파일 | 설명 |
|------|------|
| `qa_dataset.json` | QA 데이터셋 (질문-정답 쌍) |
| `all_combos_latency.csv` | 60개 전략 레이턴시 측정 결과 |
| `all_combos_ragas.csv` | 상위 N개 RAGAS 평가 점수 |
| `e2e_report.md` | 종합 리포트 (레이턴시 Top 10 + RAGAS + 실행 환경 + 비중%) |
| `benchmark_report.html` | HTML 벤치마크 보고서 (차트 인라인, 브라우저에서 바로 열기) |
| `run_history/run_*.json` | 수행 이력 (플랫폼, 전략별 타이밍, 토큰 사용량) |
| `run_history/latest.json` | 최신 실행 이력 심링크 |

## 주요 결과 요약 (10종, 20 QA)

**레이턴시 (상위 5):**

| 전략 | 평균 레이턴시 |
|------|------------|
| kosimcse+korean_bm25 | 197.5ms |
| bge-m3+korean_bm25 | 443.2ms |
| e5+splade | 488.6ms |
| bge-m3+splade+flashrank | 512.3ms |
| ColBERT | 669.8ms |

**RAGAS 품질 (상위 5):**

| 전략 | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|------|:-:|:-:|:-:|:-:|
| bge-m3+korean_bm25+colbert | **0.7592** | 0.7639 | 0.9500 | **1.0000** |
| bge-m3+korean_bm25 | 0.7317 | **0.8647** | 0.9250 | 0.9250 |
| kosimcse+korean_bm25+colbert | 0.7258 | 0.8161 | 0.9500 | 0.9250 |
| e5+splade+flashrank | 0.6917 | 0.7632 | 0.8000 | 0.8750 |
| e5+splade+colbert | 0.6275 | 0.7240 | **0.9917** | 0.9750 |

**인사이트:** BGE-M3 + korean_bm25 + ColBERT Rerank 조합이 최고 품질 (context_recall 완벽). 유료 API 모델(openai-large, upstage)은 HF 모델 대비 품질 우위 여부를 full 프리셋으로 검증 가능.

## 프로젝트 구조

```
.
├── README.md                          # 프로젝트 개요 (이 파일)
├── pyproject.toml                     # 의존성 정의
├── uv.lock                            # 의존성 잠금
├── .env                               # 환경변수 (OPENAI_API_KEY 등)
├── docker-compose.yml                 # Qdrant 컨테이너 (선택)
├── docs/                              # 평가 대상 원본 PDF 문서 + 리서치
│   └── research/                      # RAG 전략/도구 리서치 문서
├── k8s/                               # K8s 병렬 벤치마크 시스템
│   ├── Dockerfile                     # 멀티스테이지 워커 이미지 (CPU-only torch)
│   ├── orchestrator.py                # Job 생성/모니터링/수집/병합 오케스트레이터
│   ├── worker_entrypoint.py           # 2-Phase 워커 (prep/bench)
│   ├── ARCHITECTURE.md                # K8s 설계 문서
│   ├── DEPLOY_GUIDE.md                # 단계별 배포 가이드
│   └── manifests/                     # K8s 리소스 템플릿 (namespace, PVC, Job)
├── rag_bench/                         # 핵심 패키지
│   ├── base.py                        # BaseRAGStrategy ABC
│   ├── config.py                      # 전역 설정 + 모델 캐시 + SSL 우회
│   ├── runner.py                      # BenchmarkRunner
│   ├── run_tracker.py                 # 수행 이력 추적 (플랫폼, 타이밍, 토큰)
│   ├── cli.py                         # RAGChat 대화 인터페이스
│   ├── strategies/                    # RAG 전략 7종
│   │   ├── dense_sparse.py            # Dense+Sparse Hybrid (5종 임베딩)
│   │   ├── colbert.py                 # ColBERT Late Interaction
│   │   ├── colbert_rerank.py          # ColBERT 2-stage Reranking
│   │   ├── flashrank_rerank.py        # FlashRank 경량 Reranking (ONNX)
│   │   ├── contextual_retrieval.py    # Contextual Retrieval (LLM 문맥 부착)
│   │   ├── openai_embed.py            # OpenAI text-embedding-3-small/large
│   │   └── upstage_embed.py           # Upstage solar-embedding-1-large
│   ├── indexing/                      # 문서 처리 파이프라인
│   │   ├── pdf_converter.py           # PDF → Markdown (pymupdf4llm)
│   │   └── chunker.py                # Parent-Child 청킹
│   ├── evaluation/                    # RAGAS 평가 서브패키지
│   │   ├── evaluator.py               # ExtendedRAGEvaluator
│   │   ├── metrics.py                 # MetricRegistry + MetricPreset
│   │   └── legacy.py                  # 레거시 호환 shim
│   ├── graph/                         # LangGraph Agentic RAG
│   │   ├── builder.py                 # build_agent_graph()
│   │   ├── nodes.py                   # Agent 노드 (analyze, rewrite, aggregate)
│   │   ├── state.py                   # State TypedDicts
│   │   └── prompts.py                 # 프롬프트 템플릿
│   ├── combo/                         # ComboSpec, IndexCacheManager, builder
│   ├── datasets/                      # HF 데이터셋 로더
│   ├── scripts/                       # 벤치마크 실행 스크립트
│   │   ├── generate_qa.py             # QA 자동 생성 (페이지 샘플링 + QA 수 상한 지원)
│   │   ├── generate_html_report.py    # HTML 벤치마크 보고서 생성
│   │   ├── run_bench.py               # 3종 벤치마크
│   │   ├── run_all_combos.py          # 60개 조합 벤치마크
│   │   ├── prefetch_models.py         # HuggingFace 모델 프리페치
│   │   └── bench_visualize.ipynb      # 시각화 차트 노트북 (10섹션)
│   ├── docs/                          # 벤치마크 대상 Markdown 문서
│   ├── _benchdata/                    # 산출물 (.gitignore)
│   │   └── run_history/               # 수행 이력 JSON + latest.json 심링크
│   └── _models/                       # HF 모델 로컬 캐시 (.gitignore)
├── rag_bench_colab/                   # Google Colab 벤치마크 환경
│   ├── rag_benchmark.ipynb            # 메인 Colab 노트북 (9 섹션)
│   ├── colab_config.py                # Colab 환경 설정 + monkey-patch
│   ├── colab_runner.py                # 체크포인트 지원 벤치마크 러너
│   ├── colab_visualizer.py            # 12개 시각화 함수 (수행 이력 4종 포함)
│   ├── requirements_colab.txt         # Colab 전용 의존성
│   └── data/                          # QA 데이터셋 + 문서 복사본
└── scripts/                           # 환경 검증 스크립트
    ├── verify_env.py
    ├── verify_rag_bench.py
    └── verify_ragas_eval.py
```

## 평가 메트릭

### Core 메트릭 (4종)

| 메트릭 | 분류 | 설명 |
|--------|------|------|
| **Context Precision** | Retrieval | 검색된 문서 중 관련 문서 비율 |
| **Context Recall** | Retrieval | 필요한 정보가 검색 결과에 포함된 정도 |
| **Faithfulness** | Generation | 답변이 검색 문서 내용에 충실한 정도 |
| **Answer Relevancy** | Generation | 답변이 질문에 적합한 정도 |

### Extended 메트릭 (5종, COMPREHENSIVE 프리셋)

| 메트릭 | 분류 | 설명 |
|--------|------|------|
| **Context Entity Recall** | Retrieval | 검색 문서의 엔터티 재현율 |
| **Response Relevancy** | Generation | 응답의 질문 관련성 (LLM 기반) |
| **String Presence** | Lightweight | 정답 문자열 포함 여부 |
| **Exact Match** | Lightweight | 정답과 정확 일치 여부 |
| **Non-LLM String Similarity** | Lightweight | 문자열 유사도 (LLM 불필요) |

### Scoring Profiles

| 프로파일 | 메트릭 | 용도 |
|----------|--------|------|
| `default` | Core 4종 | 기본 벤치마크 |
| `comprehensive` | Core + Extended 핵심 5종 | 상세 평가 |
| `lightweight` | Lightweight 2종 | 빠른 검증 (LLM 불필요) |

## 성능 최적화

벤치마크 실행 시 적용되는 최적화:

| 최적화 | 효과 |
|--------|------|
| **인덱스 캐싱** | 동일 (Dense, Sparse) 쌍은 Qdrant 인덱스를 재사용. 60개 중 실제 인덱싱 10회 |
| **ColBERT 싱글톤** | 60개 전략이 단일 ColBERT 모델 인스턴스를 공유 |
| **FlashRank 싱글톤** | 20회 → 1회 ONNX 모델 로드 |
| **Pass 1→2 결과 재사용** | Pass 2에서 재검색 없이 Pass 1 결과 직접 주입 |
| **Answer 생성 병렬화** | `ThreadPoolExecutor(max_workers=8)` + lazy LLM 초기화 |
| **SPLADE 배치 처리** | `batch_size=32` 일괄 인코딩 |
| **MPS OOM 방지** | Apple Silicon에서 ColBERT CPU 강제 + MPS 캐시 해제 |
| **HF 모델 로컬 캐시** | `~/.cache/huggingface/hub` → `rag_bench/_models/` 심링크 |

## 수행 이력 추적 (RunTracker)

각 벤치마크 실행의 상세 이력을 자동으로 JSON에 기록합니다.

**기록 항목:**
- 실행 환경: OS, CPU, RAM, GPU, Apple Silicon 칩, Python 버전, Git 커밋
- 단계별 소요 시간: QA 로드, 청킹, 인덱싱, Pass 1 레이턴시, Pass 2 RAGAS (비중% 포함)
- 전략별 빌드 타이밍: 빌드 시간, 쿼리 레이턴시 통계 (avg/p50/p95), RAGAS 점수
- 토큰 사용량: prompt/completion/total 토큰, API 비용, LLM 호출 수

**저장 위치:** `rag_bench/_benchdata/run_history/run_{YYYYMMDD_HHMMSS}.json`
**최신 실행:** `run_history/latest.json` 심링크로 바로 접근

**시각화:** `bench_visualize.ipynb` 섹션 10에서 자동 로드하여 4종 차트로 표시 (실행 정보 카드, 단계별 타임라인, 전략별 빌드 시간, 토큰 사용량)

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

### Apple Silicon MPS OOM
ColBERT 모델이 MPS GPU 메모리를 초과하는 경우 자동으로 CPU로 폴백합니다. 수동 설정:
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Qdrant 파일 잠금 오류
이전 실행이 비정상 종료된 경우, 잠금 파일이 남아있을 수 있습니다:
```bash
rm -rf rag_bench/_benchdata/qdrant_db_*
```

### HuggingFace 모델 다운로드 오류
XET CDN 오류 발생 시:
```bash
export HF_HUB_DISABLE_XET=1
```
