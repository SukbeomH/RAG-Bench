# AutoRAG Benchmark - 한국어 RAG 파이프라인 비교 평가

한국어 문서(PDF)를 대상으로 6가지 임베딩 모델 조합의 RAG 파이프라인 성능을 [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) 프레임워크로 정량 평가하는 프로젝트입니다.

## 비교 대상

| # | 임베딩 모델 | 특징 | 차원 |
|---|-----------|------|------|
| 1 | BM-K/KoSimCSE-roberta-multitask | 한국어 특화 | 768 |
| 2 | intfloat/multilingual-e5-large | 다국어 균형 | 1024 |
| 3 | BAAI/bge-m3 | 올인원 | 1024 |
| 4 | sentence-transformers/all-MiniLM-L6-v2 | 경량/빠름 | 384 |
| 5 | OpenAI text-embedding-3-large | 고성능 API | 3072 |

추가로 BM25(한국어 토크나이저), Hybrid Retrieval, Reranker 조합도 비교합니다.

## 프로젝트 구조

```
.
├── docker-compose.yml              # Qdrant 벡터DB
├── pyproject.toml                  # Python 의존성 (uv)
├── docs/                           # 평가 대상 PDF 문서
│   ├── 20250910_AI 현황 보고서.pdf
│   └── SPRi AI Brief_1월호_산업동향_0102_F.pdf
├── autorag_benchmark/
│   ├── config/
│   │   ├── parse_config.yaml       # PDF 파싱 설정
│   │   ├── chunk_config.yaml       # 청킹 설정
│   │   ├── benchmark_config.yaml   # Dense-only 벤치마크
│   │   └── hybrid_benchmark_config.yaml  # Hybrid 벤치마크
│   └── data/
│       ├── qa.parquet              # QA 데이터셋 (100쌍)
│       └── corpus.parquet          # 청킹된 문서 (299청크)
├── scripts/
│   ├── 01_parse_and_chunk.py       # PDF → 청크
│   ├── 02_create_qa_dataset.py     # 청크 → QA 쌍 생성
│   ├── 03_run_benchmark.py         # 벤치마크 실행
│   └── 04_analyze_results.py       # 결과 분석
├── autorag_benchmark_analysis.ipynb  # 결과 시각화 노트북
└── embedding_combinations_lab.ipynb  # 임베딩 조합 실험 노트북
```

## 사전 요구사항

- **Python** 3.11+
- **Docker** (Qdrant 실행용)
- **uv** (Python 패키지 매니저) — [설치 가이드](https://docs.astral.sh/uv/getting-started/installation/)
- **OpenAI API Key** (QA 생성 + GPT-4o-mini 평가용)

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd autorag

# Python 가상환경 + 의존성 설치
uv sync

# 환경변수 설정
export OPENAI_API_KEY="sk-your-api-key-here"

# (선택) 사설 CA 인증서 사용 시
# export SSL_CERT_FILE="/path/to/your/ca-bundle.pem"
```

### 2. Qdrant 실행

```bash
docker compose up -d
```

정상 실행 확인:
```bash
curl http://localhost:6333/healthz
# 응답: {"title":"qdrant - vectorass engine","version":"..."}
```

### 3. 데이터 준비 (이미 포함됨)

`autorag_benchmark/data/`에 QA 데이터셋과 코퍼스가 이미 포함되어 있습니다.
처음부터 재생성하려면:

```bash
# PDF 파싱 → 청킹
uv run python scripts/01_parse_and_chunk.py

# QA 데이터셋 생성 (OpenAI API 호출, 비용 발생)
uv run python scripts/02_create_qa_dataset.py
```

### 4. 벤치마크 실행

```bash
# Trial 1: Dense-only (5개 임베딩 비교)
uv run python scripts/03_run_benchmark.py

# Trial 2: Hybrid (Dense + BM25 + Reranker)
# config/benchmark_config.yaml → config/hybrid_benchmark_config.yaml 로 변경 후 실행
```

또는 Python API 직접 사용:

```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(
    qa_data_path="autorag_benchmark/data/qa.parquet",
    corpus_data_path="autorag_benchmark/data/corpus.parquet",
    project_dir="autorag_benchmark/results",
)
evaluator.start_trial("autorag_benchmark/config/hybrid_benchmark_config.yaml")
```

### 5. 결과 확인

```bash
# AutoRAG 대시보드 (웹 UI)
uv run autorag dashboard --trial_dir autorag_benchmark/results/0

# 또는 분석 노트북 실행
uv run jupyter notebook autorag_benchmark_analysis.ipynb
```

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
| E5 multilingual | 0.940 | 0.118 | 1.22 |
| BGE-M3 | 0.945 | 0.118 | 1.32 |
| OpenAI Large | 0.940 | 0.118 | 0.38 |
| KoSimCSE | 0.755 | 0.098 | 0.53 |
| MiniLM | 0.695 | 0.091 | 0.39 |

### Hybrid Pipeline (Trial 1) - 최적 조합
```
E5 → BM25(ko_kiwi) → HybridCC(tmm, w=0.48) → KoReranker → GPT-4o-mini(temp=0)
```
- Retrieval Recall: 0.94
- ROUGE: 0.731, BLEU: 61.85, Semantic Score: 0.970

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

### Qdrant 연결 실패
```
[WARN] Qdrant에 연결할 수 없습니다.
```
→ `docker compose up -d`로 Qdrant 시작 확인

### konlpy 관련 오류 (ko_okt 토크나이저)
```
ModuleNotFoundError: No module named 'konlpy'
```
→ Java JDK 설치 필요: `brew install openjdk` (macOS)

### OpenAI API 오류
→ `OPENAI_API_KEY` 환경변수 확인

## 라이선스

이 프로젝트는 학습 및 연구 목적으로 작성되었습니다.
