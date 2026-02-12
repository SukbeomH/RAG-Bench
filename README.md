# RAG Bench - 한국어 RAG 파이프라인 비교 평가

한국어 문서(PDF)를 대상으로 다양한 RAG 파이프라인 성능을 정량 평가하는 프로젝트입니다.

Strategy Pattern 기반 모듈화 벤치마크 시스템으로, RAGAS 평가를 통해 15종 전략을 통일된 인터페이스로 비교합니다.

## 비교 대상

| # | 임베딩 모델 | 특징 | 차원 |
|---|-----------|------|------|
| 1 | BM-K/KoSimCSE-roberta-multitask | 한국어 특화 | 768 |
| 2 | intfloat/multilingual-e5-large | 다국어 균형 | 1024 |
| 3 | BAAI/bge-m3 | 올인원 | 1024 |
| 4 | sentence-transformers/all-MiniLM-L6-v2 | 경량/빠름 | 384 |

추가로 BM25(한국어 토크나이저), Hybrid Retrieval, ColBERT Rerank, FlashRank Rerank, Contextual Retrieval, GraphRAG 조합도 비교합니다.

---

## 실행 가이드

### 사전 요구사항

| 항목 | 필수 여부 | 설명 |
|------|----------|------|
| **Python 3.12+** | 필수 | `.python-version` 파일에 명시 |
| **uv** | 필수 | Python 패키지 매니저 ([설치](https://docs.astral.sh/uv/getting-started/installation/): `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| **Docker Desktop** | 벤치마크 실행 시 | Qdrant 벡터DB 실행용 |
| **OpenAI API Key** | 벤치마크 실행 시 | GPT-4o-mini 평가 + QA 생성용 |

### A. 결과만 확인하기 (가장 빠른 방법)

벤치마크 결과가 이미 포함되어 있으므로, 노트북만 실행하면 됩니다.

```bash
# 1. 의존성 설치
uv sync

# 2. 분석 노트북 실행
uv run jupyter notebook rag_bench/scripts/bench_visualize.ipynb
```

> Docker, OpenAI API Key 없이도 결과 확인 가능합니다.

### B. 벤치마크 전체 재실행

```bash
# 1. 의존성 설치
uv sync

# 2. 환경변수 설정
export OPENAI_API_KEY="sk-your-api-key-here"

# 3. Qdrant 벡터DB 시작
docker compose up -d

# 4. QA 데이터셋 생성
python -m rag_bench.scripts.generate_qa --num_qa 20

# 5. 전체 조합 벤치마크
python -m rag_bench.scripts.run_all_combos

# 6. 결과 확인
uv run jupyter notebook rag_bench/scripts/bench_visualize.ipynb
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
├── rag_bench/                         # 모듈화 RAG 벤치마크 패키지
│   ├── strategies/                    # RAG 전략 모듈 (6종)
│   │   ├── dense_sparse.py            # 4가지 Dense+Sparse 조합
│   │   ├── colbert.py                 # ColBERT Late Interaction (PyLate)
│   │   ├── colbert_rerank.py          # ColBERT 2-stage 리랭킹
│   │   ├── flashrank_rerank.py        # FlashRank 경량 리랭킹
│   │   ├── contextual_retrieval.py    # Contextual Retrieval (LLM 문맥 부착)
│   │   └── graph_rag.py              # GraphRAG (LightRAG 기반)
│   ├── indexing/                      # 문서 처리
│   ├── scripts/                       # 벤치마크 실행 스크립트
│   │   ├── generate_qa.py             # QA 데이터셋 자동 생성
│   │   ├── run_bench.py               # 3종 통합 벤치마크
│   │   └── run_all_combos.py          # 전체 15종 조합 비교
│   └── ...                            # 평가, 설정, 그래프 모듈
├── scripts/                           # 유틸리티 스크립트
│   ├── verify_rag_bench.py            # 환경 검증
│   ├── verify_ragas_eval.py           # RAGAS 평가 검증
│   └── verify_graphrag.py            # GraphRAG 검증
└── embedding_combinations_lab.ipynb   # 임베딩 조합 실험 노트북
```

## 벤치마크 설정

### run_all_combos.py 옵션

```
--k K                검색 결과 수 (기본: 3)
--combos 1,3,4       DenseSparse 조합 ID 지정 (미지정 시 전체)
--skip_colbert       ColBERT 단독 전략 건너뛰기
--skip_rerank        ColBERTRerank 전략 건너뛰기
--skip_graphrag      GraphRAG 전략 건너뛰기
--skip_contextual    Contextual Retrieval 건너뛰기
--skip_flashrank     FlashRank Rerank 건너뛰기
--no_ragas           RAGAS 평가 건너뛰기 (레이턴시만 측정)
--reindex            기존 인덱스 삭제 후 재인덱싱
--contextual_base N  Contextual Retrieval 기반 조합 ID (기본: 3)
```

## 주요 결과 요약 (10종, 20 QA)

**RAGAS 품질 (상위 5):**

| 전략 | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|------|:-:|:-:|:-:|:-:|
| Rerank-DS3 (BGE-M3) | **0.7592** | 0.7639 | 0.9500 | **1.0000** |
| DS3 BGE-M3 | 0.7317 | **0.8647** | 0.9250 | 0.9250 |
| Rerank-DS1 | 0.7258 | 0.8161 | 0.9500 | 0.9250 |
| Rerank-DS4 | 0.6917 | 0.7632 | 0.8000 | 0.8750 |
| Rerank-DS2 | 0.6275 | 0.7240 | **0.9917** | 0.9750 |

## 평가 메트릭

- **Retrieval**: RAGAS Context Precision, Context Recall
- **Generation**: RAGAS Faithfulness, Answer Relevancy

## Qdrant 관리

```bash
docker compose up -d      # 시작
docker compose down        # 중지
docker compose down -v     # 데이터 포함 완전 삭제
```

## 트러블슈팅

### uv가 설치되어 있지 않음
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Qdrant 연결 실패
→ Docker Desktop이 실행 중인지 확인 후 `docker compose up -d`

### konlpy 관련 오류 (ko_okt 토크나이저)
→ Java JDK 설치 필요: macOS: `brew install openjdk`, Ubuntu: `sudo apt install default-jdk`

### OpenAI API 오류
→ `OPENAI_API_KEY` 환경변수가 올바르게 설정되어 있는지 확인

### 사설 CA 인증서 (기업 네트워크)
```bash
export SSL_CERT_FILE="/path/to/your/ca-bundle.pem"
```

## 라이선스

이 프로젝트는 학습 및 연구 목적으로 작성되었습니다.
