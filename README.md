# AutoRAG Benchmark - 한국어 RAG 파이프라인 비교 평가

한국어 문서(PDF)를 대상으로 다양한 RAG 파이프라인 성능을 정량 평가하는 프로젝트입니다.

**두 가지 벤치마크 프레임워크를 동일 데이터로 비교합니다:**

| 프레임워크 | 설명 | 스크립트 |
|-----------|------|----------|
| **rag_bench** (자체) | Strategy Pattern 기반 모듈화 RAG 벤치마크 (10종 전략) | `rag_bench/scripts/run_all_combos.py` |
| **AutoRAG** (외부) | AutoRAG 프레임워크 파이프라인 탐색 | `rag_bench/scripts/run_autorag.py` |

## 비교 대상

| # | 임베딩 모델 | 특징 | 차원 |
|---|-----------|------|------|
| 1 | BM-K/KoSimCSE-roberta-multitask | 한국어 특화 | 768 |
| 2 | intfloat/multilingual-e5-large | 다국어 균형 | 1024 |
| 3 | BAAI/bge-m3 | 올인원 | 1024 |
| 4 | sentence-transformers/all-MiniLM-L6-v2 | 경량/빠름 | 384 |
| 5 | OpenAI text-embedding-3-large | 고성능 API | 3072 |

추가로 BM25(한국어 토크나이저), Hybrid Retrieval, Reranker, ColBERT, GraphRAG 조합도 비교합니다.

---

## 실행 가이드 (zip 압축 해제 후)

### 사전 요구사항

| 항목 | 필수 여부 | 설명 |
|------|----------|------|
| **Python 3.12+** | 필수 | `.python-version` 파일에 명시 |
| **uv** | 필수 | Python 패키지 매니저 ([설치](https://docs.astral.sh/uv/getting-started/installation/): `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| **Docker Desktop** | 벤치마크 재실행 시 | Qdrant 벡터DB 실행용 |
| **OpenAI API Key** | 벤치마크 재실행 시 | GPT-4o-mini 평가 + QA 생성용 |

### A. 결과만 확인하기 (가장 빠른 방법)

벤치마크 결과가 이미 포함되어 있으므로, 노트북만 실행하면 됩니다.

```bash
# 1. 압축 해제
unzip autorag-benchmark.zip -d autorag
cd autorag

# 2. 의존성 설치
uv sync

# 3. 분석 노트북 실행
uv run jupyter notebook autorag_benchmark_analysis.ipynb
```

> Docker, OpenAI API Key 없이도 결과 확인 가능합니다.

### B. 벤치마크 전체 재실행

처음부터 벤치마크를 직접 재현하고 싶다면 아래 순서를 따릅니다.

```bash
# 1. 압축 해제 및 의존성 설치
unzip autorag-benchmark.zip -d autorag
cd autorag
uv sync

# 2. 환경변수 설정
export OPENAI_API_KEY="sk-your-api-key-here"

# (선택) 사설 CA 인증서 사용 환경
# export SSL_CERT_FILE="/path/to/your/ca-bundle.pem"

# 3. Qdrant 벡터DB 시작
docker compose up -d

# 정상 확인
curl http://localhost:6333/healthz
# → "healthz check passed"

# 4. (선택) 데이터 재생성 — 이미 포함되어 있으므로 건너뛰어도 됨
# uv run python scripts/01_parse_and_chunk.py    # PDF 파싱 → 청킹
# uv run python scripts/02_create_qa_dataset.py  # QA 데이터셋 생성 (API 비용 발생)

# 5. Dense-only 벤치마크 실행 (Trial 0)
uv run python scripts/03_run_benchmark.py

# 6. Hybrid 벤치마크 실행 (Trial 1)
#    scripts/03_run_benchmark.py 110행의 config 경로를 변경:
#    benchmark_config.yaml → hybrid_benchmark_config.yaml
uv run python scripts/03_run_benchmark.py

# 7. 결과 확인
uv run jupyter notebook autorag_benchmark_analysis.ipynb

# 또는 AutoRAG 대시보드 (웹 UI)
uv run autorag dashboard --trial_dir autorag_benchmark/results/0
```

---

## 프로젝트 구조

```
.
├── docker-compose.yml                 # Qdrant 벡터DB (Docker Compose)
├── pyproject.toml                     # Python 의존성 (uv)
├── uv.lock                            # 의존성 lock 파일
├── docs/                              # 평가 대상 PDF 문서
│   ├── 20250910_AI 현황 보고서.pdf
│   └── SPRi AI Brief_1월호_산업동향_0102_F.pdf
├── autorag_benchmark/
│   ├── config/                        # 벤치마크 설정
│   │   ├── parse_config.yaml          #   PDF 파싱 설정
│   │   ├── chunk_config.yaml          #   청킹 설정
│   │   ├── benchmark_config.yaml      #   Dense-only 벤치마크
│   │   └── hybrid_benchmark_config.yaml  # Hybrid 벤치마크
│   ├── data/                          # AutoRAG 독립 데이터셋 (100 QA)
│   │   ├── qa.parquet                 #   QA 100쌍
│   │   └── corpus.parquet             #   문서 청크 299개
│   ├── data_ragbench/                 # rag_bench 동일 데이터 변환본 (20 QA)
│   └── results/                       # 벤치마크 결과 (포함됨)
│       ├── 0/                         #   Trial 0: Dense-only
│       └── 1/                         #   Trial 1: Hybrid
├── scripts/                           # AutoRAG 독립 스크립트 (레거시)
│   ├── 01_parse_and_chunk.py          # PDF → 청크
│   ├── 02_create_qa_dataset.py        # 청크 → QA 쌍 생성
│   ├── 03_run_benchmark.py            # 벤치마크 실행
│   └── 04_analyze_results.py          # 결과 분석
├── rag_bench/                         # 모듈화 RAG 벤치마크 패키지
│   ├── scripts/
│   │   ├── generate_qa.py             # QA 데이터셋 자동 생성
│   │   ├── run_all_combos.py          # 전체 10종 조합 비교
│   │   └── run_autorag.py             # AutoRAG 크로스 프레임워크 벤치마크
│   └── ...                            # 전략, 인덱싱, 평가 모듈
├── autorag_benchmark_analysis.ipynb   # 결과 시각화/분석 노트북
├── embedding_combinations_lab.ipynb   # 임베딩 조합 실험 노트북
└── autorag_research.md                # AutoRAG 리서치 노트
```

## 크로스 프레임워크 벤치마크 (rag_bench ↔ AutoRAG)

rag_bench의 QA 20개 + child_chunks를 AutoRAG parquet 포맷으로 변환하여 동일 데이터 기반 비교:

```bash
# AutoRAG 의존성 설치 (optional)
uv pip install -e '.[autorag]'

# Dense 벤치마크
docker compose up -d
python -m rag_bench.scripts.run_autorag --config dense

# Hybrid + Reranker 벤치마크
python -m rag_bench.scripts.run_autorag --config hybrid

# rag_bench 결과와 비교
python -m rag_bench.scripts.run_autorag --config dense --compare
```

> AutoRAG는 LangChain 버전 충돌 위험이 있으므로 optional dependency로 분리되어 있습니다.

## 벤치마크 설정

### Dense-only (`benchmark_config.yaml`)
- 5개 임베딩 모델 비교 (VectorDB)
- Prompt → GPT-4o-mini Generator

### Hybrid (`hybrid_benchmark_config.yaml`)
- Dense Retrieval (5개 임베딩)
- Sparse Retrieval (BM25 + 한국어 토크나이저: ko_kiwi, ko_okt)
- Hybrid Fusion (RRF, CC with mm/tmm normalization)
- Reranker (PassReranker, KoReranker, FlashRank)
- Prompt Maker (2가지 한국어 프롬프트)
- Generator (GPT-4o-mini, temperature 0/0.3)

## 주요 결과 요약

### Dense Retrieval (Trial 0)

| 모델 | Recall | F1 | 속도(초) |
|------|--------|----|---------|
| **E5 multilingual** | 0.940 | 0.118 | 1.22 |
| BGE-M3 | 0.945 | 0.118 | 1.32 |
| OpenAI Large | 0.940 | 0.118 | 0.38 |
| KoSimCSE | 0.755 | 0.098 | 0.53 |
| MiniLM | 0.695 | 0.091 | 0.39 |

### Hybrid Pipeline (Trial 1) — 최적 조합

```
E5 → BM25(ko_kiwi) → HybridCC(tmm, w=0.48) → KoReranker → GPT-4o-mini(temp=0)
```

| 메트릭 | 값 |
|-------|------|
| Retrieval Recall | 0.94 |
| ROUGE | 0.731 |
| BLEU | 61.85 |
| Semantic Score | 0.970 |

## 평가 메트릭

- **Retrieval**: Recall, F1, Precision
- **Generation**: ROUGE, BLEU, METEOR, Semantic Score

## Qdrant 관리

```bash
# 시작
docker compose up -d

# 중지
docker compose down

# 데이터 포함 완전 삭제
docker compose down -v
```

## 트러블슈팅

### uv가 설치되어 있지 않음
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 설치 후 터미널 재시작
```

### Qdrant 연결 실패
```
[WARN] Qdrant에 연결할 수 없습니다.
```
→ Docker Desktop이 실행 중인지 확인 후 `docker compose up -d`

### konlpy 관련 오류 (ko_okt 토크나이저)
```
ModuleNotFoundError: No module named 'konlpy'
```
→ Java JDK 설치 필요:
  - macOS: `brew install openjdk`
  - Ubuntu: `sudo apt install default-jdk`

### OpenAI API 오류
→ `OPENAI_API_KEY` 환경변수가 올바르게 설정되어 있는지 확인

### 사설 CA 인증서 (기업 네트워크)
→ SSL 오류 발생 시:
```bash
export SSL_CERT_FILE="/path/to/your/ca-bundle.pem"
```

## 라이선스

이 프로젝트는 학습 및 연구 목적으로 작성되었습니다.
