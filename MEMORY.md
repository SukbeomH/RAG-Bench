# Project Memory

## 2026-02-11: RAG Bench 검증 및 환경 구성 (Python 3.12)

### 주요 활동
- `rag_bench` 패키지의 기능 검증을 위한 Python 3.12 환경 구성 및 스크립트 실행 완료.
- `scripts/verify_rag_bench.py`를 통해 다음 기능 검증 성공:
  - 패키지 Import
  - Parent-Child 청킹 (합성 Markdown)
  - 인덱싱 및 검색 (`DenseSparseStrategy`: MiniLM + BM25)
  - 벤치마크 Runner 실행
  - LangGraph Agent 빌드

### 해결된 이슈
1. **LangChain 버전 호환성**
   - `langchain>=1.0`, `langchain-core>=0.3` 등을 명시하여 해결.

2. **SSL 인증서 및 모델 다운로드**
   - 보안 네트워크 환경에서 HuggingFace Hub의 SSL 검증 실패.
   - `rag_bench/config.py` 수정: `REQUESTS_CA_BUNDLE` 초기화 코드 제거, `HF_HUB_DISABLE_SSL_VERIFY=1` 등 환경변수 추가.

3. **Qdrant 파일 락**
   - `DenseSparseStrategy` 인덱싱 중 Qdrant 클라이언트 중복 초기화로 인한 `BlockingIOError` 발생.
   - `_init_qdrant` 메소드 수정: 클라이언트 객체 재사용 로직 추가.

### 참고 사항
- 검증 상세 내용은 `walkthrough.md` 참조.
- `config.py`는 로컬 인증서 경로(`.env`의 `REQUESTS_CA_BUNDLE`)를 활용하도록 수정됨.

## 2026-02-11: RAGAS 평가 통합 및 환경변수 설정

### 주요 활동
- **RAGAS(Evaluator) 통합**: `rag_bench/evaluation.py` 생성 및 `BenchmarkRunner` 연동 완료.
  - 주요 메트릭: Faithfulness, Answer Relevancy, Context Precision, Context Recall.
  - 검색 전용 전략에 대해 `gpt-3.5-turbo`로 답변 생성을 자동화하여 평가 가능하도록 구현.
  - **SSL/Proxy 우회**: 기업 네트워크 환경 대응을 위해 `httpx.Client(verify=False)`를 강제 적용 (Sync/Async 모두).

- **환경변수 자동 로드**:
  - `rag_bench/config.py`에서 `python-dotenv`를 사용하여 프로젝트 루트의 `.env` 파일을 자동으로 로드 설정.
  - `rag_bench` 패키지 import 시 즉시 적용됨.

### 해결된 이슈
1. **OpenAI API 연결 실패 (SSL/Proxy)**
   - 원인: 기업 보안 네트워크에서 SSL 인증서 검증 실패.
   - 해결: `rag_bench/evaluation.py` 및 `runner.py` 내의 `ChatOpenAI`, `OpenAIEmbeddings` 초기화 시 `verify=False` 옵션을 가진 `httpx` 클라이언트를 주입하여 우회.

2. **Ragas 결과 객체 처리**
   - 원인: `ragas` v0.4+의 `evaluate()` 반환 객체(`EvaluationResult`)가 dict처럼 동작하지 않음(`.items()` 부재).
   - 해결: `evaluation.py`에서 결과 객체의 내부 `.scores` 리스트를 순회하여 평균 점수를 계산한 후 dict로 반환하도록 수정.

### 검증 결과
- `scripts/verify_ragas_eval.py`: Mock 전략을 사용한 검색 및 평가 파이프라인 정상 동작 확인.
- `scripts/verify_env.py`: `.env` 파일의 `OPENAI_API_KEY`가 정상 로드됨을 확인.

## 2026-02-11: ColBERT Late Interaction 전략 구현 (PyLate 기반)

### 주요 활동
- **ColBERTStrategy 전체 구현**: 스텁 상태였던 `rag_bench/strategies/colbert.py`를 PyLate 백엔드로 완전 구현.
  - `ColBERTRetriever`: LangChain `BaseRetriever` 래퍼 (~10 LOC).
  - `ColBERTStrategy`: 메인 클래스 (~170 LOC), `BaseRAGStrategy` ABC 준수.
  - **Brute-force 모드 (기본)**: `pylate.rank.rerank()`으로 MaxSim 스코어링. 소규모 코퍼스에 적합.
  - **Voyager 인덱스 모드** (`use_index=True`): `pylate.indexes.Voyager` ANN 인덱스 사용. 대규모 검색용.
  - Lazy 모델 로드 (`_ensure_initialized`), CUDA/MPS/CPU 자동 감지.
  - 메타데이터(`parent_id`, `source`) 완전 보존, k clamp 처리.
  - `cleanup()`: 메모리/인덱스 파일 정리.

- **의존성 추가**: `pyproject.toml`에 `pylate>=1.0`, `einops>=0.8.2` 추가.

- **커밋 정리**: 전체 변경사항을 6개 논리적 커밋으로 분리.

### 해결된 이슈
1. **`trust_remote_code` 필요**
   - jina-colbert-v2 모델이 커스텀 XLM-RoBERTa 구현을 사용하여 `trust_remote_code=True` 필수.
   - `models.ColBERT()` 생성자에 해당 옵션 추가.

2. **`einops` 누락 의존성**
   - jina-colbert-v2 모델의 커스텀 코드가 `einops` 패키지를 요구.
   - `pyproject.toml`에 `einops>=0.8.2` 추가.

3. **HuggingFace XET CDN 오류**
   - jina-colbert-v2 가중치 다운로드 시 `CAS service error: Request failed after 5 retries` 발생.
   - 원인: HuggingFace의 XET 스토리지 백엔드 CDN 문제.
   - 해결: `HF_HUB_DISABLE_XET=1` 환경변수 설정으로 기존 다운로드 방식 사용.

### 검증 결과
- jina-colbert-v2 모델: 한국어 쿼리 검색, 메타데이터 보존, LangChain Retriever 호환 모두 통과.
- sentence-transformers/all-MiniLM-L6-v2: 경량 모델로 전체 로직 검증 완료.

### 현재 프로젝트 구현 상태
| 전략 | 상태 | 비고 |
|------|------|------|
| `DenseSparseStrategy` | **완료** | 6가지 임베딩 조합 (Qdrant 하이브리드) |
| `ColBERTStrategy` | **완료** | PyLate 기반, brute-force + Voyager 지원 |
| `GraphRAGStrategy` | 스텁 | LightRAG 기반 구현 예정 |

### 커밋 히스토리
```
d68eb50 feat: 임베딩 조합 실험 노트북 업데이트
e781df0 feat: 환경/기능 검증 스크립트 추가
22bf118 docs: README 갱신 및 설정 가이드, 전략 리서치 문서 추가
97be793 feat: rag_bench 모듈형 벤치마크 프레임워크 패키지 추가
df92494 chore: Python 3.12로 업그레이드 및 lockfile 갱신
73dd2b9 feat: ColBERT Late Interaction 검색 전략 구현 (PyLate 기반)
```

## 2026-02-11: GraphRAGStrategy 구현 (LightRAG 기반) + ColBERTRerankStrategy

### 주요 활동
- **ColBERTRerankStrategy 구현**: 임의의 1차 검색 전략 위에 ColBERT MaxSim 리랭킹을 얹는 2단계 전략.
  - `ColBERTRerankRetriever`: LangChain `BaseRetriever` 래퍼.
  - `ColBERTRerankStrategy` (~200 LOC): base_strategy → rerank_n개 후보 → ColBERT 인코딩 → MaxSim 재정렬 → top-k 반환.
  - 커밋: `4ed947e feat: ColBERTRerankStrategy 구현 — 2단계 리랭킹 전략 추가`

- **GraphRAGStrategy 전체 구현**: 스텁 상태였던 `graph_rag.py`를 LightRAG 백엔드로 완전 구현.
  - `GraphRAGRetriever`: LangChain `BaseRetriever` 래퍼.
  - `GraphRAGStrategy` (~180 LOC): LightRAG 기반 엔티티-관계 지식 그래프 RAG.
    - async→sync 래핑 (`_run_async`): Jupyter 환경 지원 (`nest_asyncio`).
    - Lazy 초기화 (`_ensure_initialized`): `openai_complete_if_cache`로 커스텀 LLM 함수 생성.
    - 스토리지: JsonKV + NanoVectorDB + NetworkX (파일 기반, 서버 불필요).
    - 검색 모드: local, global, hybrid, naive, mix.
  - 기본 LLM: `gpt-4.1-nano` (GPT-4o-mini 대비 입력 7.5배, 출력 4배 저렴).
  - 커밋: `da5bced feat: GraphRAGStrategy 구현 — LightRAG 기반 지식 그래프 RAG 전략`

- **의존성 추가**: `lightrag-hku>=1.0`, `nest-asyncio>=1.6`.
- **`.gitignore` 갱신**: `lightrag_index/` 추가.

### LLM 비용 비교 (그래프 구축용)
| 모델 | Input / 1M tokens | Output / 1M tokens |
|------|-------------------|---------------------|
| gpt-4.1-nano (기본) | $0.02 | $0.15 |
| gpt-4o-mini | $0.15 | $0.60 |
| Gemini 2.5 Flash | $0.10 | $0.40 |

### 현재 전략 구현 상태 (4/4 완료)
| 전략 | 상태 | 비고 |
|------|------|------|
| `DenseSparseStrategy` | **완료** | 6가지 임베딩 조합 (Qdrant 하이브리드) |
| `ColBERTStrategy` | **완료** | PyLate 기반, brute-force + Voyager |
| `ColBERTRerankStrategy` | **완료** | 2단계 리랭킹 (임의 1차 전략 + ColBERT MaxSim) |
| `GraphRAGStrategy` | **완료** | LightRAG 기반, gpt-4.1-nano, hybrid 모드 |

### 다음 작업
- ~~**통합 벤치마크**: DenseSparse vs ColBERT vs GraphRAG 비교 + RAGAS 평가~~ → **완료** (아래 세션 참조)
- **GraphRAG E2E 검증**: 소규모 문서로 실제 index/retrieve 테스트
- **Contextual Retrieval**: Anthropic 방식 인덱싱 보강 구현

## 2026-02-11: 통합 벤치마크 스크립트 + 전체 9종 비교 + 시각화 노트북

### 주요 활동
- **rag_bench 패키지 독립 공유 구조화**: 스크립트, 문서, 의존성 파일을 패키지 내부에 배치하여 `rag_bench/` 단독 공유 가능.
- **벤치마크 스크립트 3종 신규 작성**:
  - `scripts/generate_qa.py`: `docs/*.md` → Parent-Child 청킹 → GPT-4o-mini QA 자동 생성 (해시 캐싱, `--force` 재생성).
  - `scripts/run_bench.py`: DenseSparse(combo4) + ColBERT + ColBERTRerank 3종 벤치마크 + RAGAS 평가.
  - `scripts/run_all_combos.py`: DenseSparse 6종 + ColBERT + ColBERTRerank×N 전체 조합 비교 (실패 내성, `--skip_paid`, `--combos`, `--no_ragas`).
- **시각화 노트북**: `scripts/bench_visualize.ipynb` — 7종 차트 (레이턴시 바, RAGAS Grouped Bar, 레이더, 품질-속도 Scatter, 히트맵, 쿼리별 분포, 종합 순위표). 각 섹션에 **상세 해석 가이드** 포함 (읽는 법, 해석 포인트, 전략 선택 기준 등 한국어 설명).
- **패키지 내부 파일 배치**:
  - `rag_bench/docs/`: 벤치마크 대상 markdown 문서 (2개).
  - `rag_bench/pyproject.toml`, `uv.lock`, `.python-version`, `docker-compose.yml`: 의존성 및 인프라 파일 복사.
  - `rag_bench/_benchdata/`: 벤치마크 중간 산출물 (.gitignore 제외).
- **config.py 확장**: `PACKAGE_ROOT`, `BENCH_DOCS_DIR`, `BENCH_DATA_DIR` 경로 상수 추가.
- **README.md 전면 개편**: 아키텍처 다이어그램, 전략 4종 상세, 스크립트 사용법, 독립 공유 안내 반영.

### 전체 9종 벤치마크 실행 결과 (2개 QA, --skip_paid)

**레이턴시 순위:**
| 전략 | 평균 레이턴시 |
|------|-------------|
| DS4-MiniLM+BM25 | 107ms |
| DS1-KoSimCSE+BM25/OKt | 1,309ms |
| DS3-BGE-M3 | 1,882ms |
| DS2-E5+SPLADE | 3,251ms |
| ColBERT (jina-colbert-v2) | 8,143ms |
| ColBERTRerank (DS4) | 8,640ms |
| ColBERTRerank (DS2) | 12,363ms |
| ColBERTRerank (DS3) | 12,417ms |
| ColBERTRerank (DS1) | 13,550ms |

**RAGAS 품질 순위 (종합):**
| 전략 | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|------|:-:|:-:|:-:|:-:|
| ColBERT | **1.00** | 0.86 | **1.00** | **1.00** |
| Rerank-E5+SPLADE | 0.75 | **0.93** | **1.00** | **1.00** |
| Rerank-KoSimCSE | 0.67 | 0.92 | **1.00** | **1.00** |
| DS3-BGE-M3 | 0.75 | 0.84 | 0.92 | **1.00** |
| DS2-E5+SPLADE | **1.00** | 0.46 | 0.92 | **1.00** |
| DS4-MiniLM | 0.42 | 0.86 | 0.25 | 0.50 |

**인사이트**: ColBERT가 품질 최고(4개 메트릭 만점), ColBERTRerank가 기존 전략 위에 품질 향상 효과 확인. MiniLM+BM25는 속도 최고지만 한국어 품질 약함.

### 디렉토리 구조 (최종)
```
rag_bench/
├── scripts/
│   ├── generate_qa.py       # QA 데이터셋 자동 생성
│   ├── run_bench.py         # 3종 통합 벤치마크
│   ├── run_all_combos.py    # 전체 조합 비교
│   └── bench_visualize.ipynb # 시각화 노트북
├── docs/                    # 벤치마크 대상 문서
├── _benchdata/              # 중간 산출물 (.gitignore)
├── pyproject.toml           # 의존성 (복사본)
├── uv.lock                  # 버전 잠금 (복사본)
├── docker-compose.yml       # Qdrant (복사본)
└── .python-version          # Python 3.12
```

### 커밋 히스토리
```
1f5ae06 feat: 통합 벤치마크 스크립트 추가 — QA 생성 + RAGAS 평가 + 전체 조합 비교
```

### 다음 작업
1. **QA 데이터셋 확충**: `--num_qa 20`+ 로 더 많은 QA 생성하여 벤치마크 신뢰도 향상
2. **GraphRAG 벤치마크 통합**: `run_all_combos.py`에 GraphRAG 전략 추가
3. **Contextual Retrieval**: Anthropic 방식 인덱싱 보강 구현

## 2026-02-12: QA 확충 + GraphRAG 벤치마크 통합 + 인덱스 재사용

### 주요 활동
- **QA 데이터셋 확충**: 2개 → 20개로 확대. `generate_qa.py`의 `_sample_parents()` 에서 `max_size=5000` 제한 제거 (대부분 parent가 5000자 이상이어서 필터링됨).
- **GraphRAG 벤치마크 통합**: `run_all_combos.py`에 GraphRAG 전략 추가. parent 단위 삽입(33개)으로 LLM 비용 절감 (child 763개 대비).
- **10종 전략 벤치마크 완료**: DenseSparse 4종 + ColBERT + ColBERTRerank 4종 + GraphRAG. 20 QA × 10 전략 = 200회 검색 + RAGAS 평가.
- **인덱스 재사용 기능 구현**: `--reindex` 옵션 추가. 기본값은 기존 인덱스 재사용 (DenseSparse: Qdrant 연결 + BM25 fit, GraphRAG: LightRAG 초기화만). 인덱스 없으면 자동 폴백.
- **진행률 표시 추가**: `[1/10] ▶ 생성 중: ...` + `[기존 로드]`/`[재인덱싱]` 단계별 출력.
- **bench_visualize.ipynb GraphRAG 호환**: shorten/classify/TYPE_COLORS에 GraphRAG 추가 (#54A24B 초록).

### 10종 벤치마크 결과 (20 QA, --skip_paid)

**레이턴시:**
| 전략 | 평균 레이턴시 |
|------|------------|
| DS4 MiniLM+BM25 | 60.8ms |
| DS1 KoSimCSE+BM25 | 197.5ms |
| DS3 BGE-M3 | 443.2ms |
| DS2 E5+SPLADE | 488.6ms |
| ColBERT | 669.8ms |
| GraphRAG | 1,734.2ms |
| Rerank-DS4 | 1,925.2ms |
| Rerank-DS1 | 2,991.8ms |
| Rerank-DS3 | 5,378.8ms |
| Rerank-DS2 | 6,615.2ms |

**RAGAS 품질 (상위 5):**
| 전략 | Faithfulness | Answer Rel. | Context Prec. | Context Recall |
|------|:-:|:-:|:-:|:-:|
| Rerank-DS3 (BGE-M3) | **0.7592** | 0.7639 | 0.9500 | **1.0000** |
| DS3 BGE-M3 | 0.7317 | **0.8647** | 0.9250 | 0.9250 |
| Rerank-DS1 | 0.7258 | 0.8161 | 0.9500 | 0.9250 |
| Rerank-DS4 | 0.6917 | 0.7632 | 0.8000 | 0.8750 |
| Rerank-DS2 | 0.6275 | 0.7240 | **0.9917** | 0.9750 |

**인사이트**: BGE-M3 + ColBERT Rerank 조합이 최고 품질 (context_recall 완벽). GraphRAG는 context_recall 0.975로 높지만 precision 0.5, answer_relevancy 0.65로 noise 많음.

### 해결된 이슈
1. **QA 생성 2개만 생성**: `_sample_parents()`의 `max_size=5000` 필터 → 대부분 parent 17,860자 → 제거하여 해결.
2. **GraphRAG 763 child_chunks 인덱싱 6시간+**: parent 33개로 변경하여 수분 내 완료.

### 커밋 히스토리
```
TBD — 이 세션에서 커밋 예정
```

### 다음 작업
1. ~~Contextual Retrieval 구현~~ → **완료** (아래 세션 참조)
2. 추가 평가 메트릭 확장

## 2026-02-12: FlashRank/Contextual Retrieval 추가 + 유료 모델 제거 + AutoRAG 전면 제거

### 주요 활동
- **FlashRank 리랭커 전략 추가**: `strategies/flashrank_rerank.py` 신규 (~110 LOC).
  - ms-marco-MultiBERT-L-12 (ONNX, CPU 전용, ~150MB).
  - ColBERTRerank과 동일한 Decorator 패턴 (base_strategy.retrieve → FlashRank 리랭킹).
- **Contextual Retrieval 전략 추가**: `strategies/contextual_retrieval.py` 신규 (~200 LOC).
  - Anthropic 방식 LLM 문맥 부착. parent 문서 기반으로 child chunk에 컨텍스트 프리픽스 생성.
  - JSON 해시 캐싱으로 중복 LLM 호출 방지. 검색 시 원본 콘텐츠 복원.
- **유료 API 전략 제거**: DenseSparse combo 5(OpenAI), 6(Upstage) 제거.
  - `dense_sparse.py`: COMBO_DEFINITIONS에서 제거, `_init_dense()` HuggingFace 전용으로 단순화.
  - `run_all_combos.py`: `PAID_COMBO_IDS`, `--skip_paid` 제거, 새 전략 플래그 추가.
- **AutoRAG 전면 제거** (`refactor/remove-autorag` 브랜치 → main 머지):
  - `autorag_benchmark/` 디렉토리 전체 삭제 (config, data, results).
  - `scripts/01-04`, `run_autorag_isolated.py`, `rag_bench/scripts/run_autorag.py` 삭제.
  - `autorag_benchmark_analysis.ipynb`, `autorag_research.md`, `main.py` 삭제.
  - README.md, rag_bench/README.md, pyproject.toml, setup_guide.md 정리.
  - AutoRAG 포기 사유: langchain-core 버전 충돌, httpx 클라이언트 호환성 문제.
- **조합형 벤치마크 리팩토링 실현 가능성 분석** (`rag_bench/review_report.md`):
  - 5-Layer 파이프라인 설계: Dense × Sparse × Mode × Reranker × LLM Support.
  - 유효 조합 134개 (이론적 288개 중 무효 제거).
  - 핵심: DenseSparseStrategy 분해 (combo_id → 독립 파라미터).
  - Reranker/LLM 레이어는 이미 Decorator 패턴으로 분리 → 변경 불필요.
  - 실행 시간: 레이턴시만 ~30분, 선별 RAGAS ~1시간.
  - API 비용: ~$4.40.

### 커밋 히스토리
```
c73c41a feat: FlashRank/Contextual Retrieval 전략 추가 + 유료 모델 제거 + AutoRAG 분리
f547dfa refactor: AutoRAG 의존성 및 관련 파일 전면 제거
```

### 현재 전략 구현 상태 (6종)
| 전략 | 상태 | 비고 |
|------|------|------|
| `DenseSparseStrategy` | **완료** | 4가지 임베딩 조합 (combo 1-4, HuggingFace 전용) |
| `ColBERTStrategy` | **완료** | PyLate 기반, brute-force + Voyager |
| `ColBERTRerankStrategy` | **완료** | 2단계 리랭킹 (base + ColBERT MaxSim) |
| `FlashRankRerankStrategy` | **완료** | 2단계 리랭킹 (base + FlashRank ONNX) |
| `ContextualRetrievalStrategy` | **완료** | Anthropic 방식 LLM 문맥 부착 + base 래핑 |
| `GraphRAGStrategy` | **완료** | LightRAG 기반, gpt-4.1-nano |

### 다음 작업 (별도 세션에서 진행)
1. ~~**DenseSparseStrategy 분해**~~ → **완료** (아래 세션 참조)
2. ~~**RAGAS E2E 파이프라인**~~ → **완료** (아래 세션 참조)
3. **evaluation 서브패키지**: RAGAS v0.4 API 마이그레이션 + Extended 메트릭 + per-sample 점수
4. ~~**QA 데이터셋 개선**~~ → **폐기** (아래 세션에서 worktree/브랜치 제거)

## 2026-02-12: 3-Layer 조합 벤치마크 + 프로젝트 레거시 정리

### 주요 활동
- **3-Layer 조합 벤치마크 파이프라인 구현** (`feat/ragas-e2e-pipeline` 브랜치):
  - DenseSparseStrategy 분해: combo_id → (dense_model, sparse_type, retrieval_mode) 독립 파라미터화.
  - 72개 교차 조합 자동 열거 + 2-Pass 실행 (레이턴시 → 선별 RAGAS 평가).
  - `rag_bench/evaluation/` 서브패키지 신설: evaluator.py, metrics.py, legacy.py 분리.

- **프로젝트 루트 레거시 전면 정리**:
  - **파일 삭제**: `embedding_combinations_lab.ipynb` (git rm), `markdown/`, `autorag_benchmark_analysis_executed.ipynb`, `parent_store/`, `qdrant_db_combo1/`, `.mypy_cache/` (~725MB 디스크 절감).
  - **`.gitignore` 정리**: AutoRAG 패턴 제거 (autorag_benchmark/*), `markdown/` 추가.
  - **research 문서 현행화**: 4개 문서에서 "현재 AutoRAG 구현" → "rag_bench" 참조 변경.
    - `ragatouille_research.md`: ColBERTStrategy 구현 완료 반영, 연동 방안 → 결과 요약으로 교체.
    - `noderag_research.md`: AutoRAG → rag_bench 용어 통일.
    - `rag_dataset_creation_methodology.md`: 현행 generate_qa.py 방식 반영, 현황 참고 추가.
    - `raghub_ecosystem_research.md`: 프로젝트 관계를 "AutoRAG 활용" → "자체 프레임워크 전환"으로 현행화.
  - **README.md**: 프로젝트 구조에서 삭제된 노트북 참조 제거.

- **브랜치/워크트리 정리**:
  - `feat/ragas-e2e-pipeline` → main Fast-forward 머지 후 삭제.
  - `refactor/remove-autorag` → 삭제 (이미 main에 머지됨).
  - `feature/qa-dataset-improvement` + worktree (`autorag-qa-improvement`) → 강제 제거 (main 이전 시점의 레거시 코드, 독자 작업 없음).

### 커밋 히스토리
```
3850eba feat: 3-Layer 조합 벤치마크 파이프라인 구현 (72개 교차 조합 + 2-Pass 실행)
f587073 chore: 레거시 정리 — AutoRAG 잔존 참조 현행화 + 불필요 파일 제거
```

### 현재 프로젝트 상태
- **브랜치**: main만 존재 (모든 작업 브랜치 정리 완료)
- **전략 구현**: 6종 완료 (DenseSparse, ColBERT, ColBERTRerank, FlashRankRerank, ContextualRetrieval, GraphRAG)
- **벤치마크**: 72개 교차 조합 지원 (3-Layer: Dense × Sparse × Reranker)
- **evaluation**: 서브패키지 구조 전환 (evaluator.py + metrics.py + legacy.py)

### 다음 작업
1. ~~**72개 조합 벤치마크 실행**~~ → 아래 세션에서 시도, MPS OOM으로 실패 후 수정
2. **evaluation 메트릭 확장**: Extended 메트릭 + per-sample 점수
3. **벤치마크 시각화 갱신**: bench_visualize.ipynb를 72개 조합 결과에 맞게 업데이트

## 2026-02-19: 72개 벤치마크 실행 + MPS OOM 수정 + 모델 캐시 + RAGAS 리서치

### 주요 활동

#### 1. 72개 조합 벤치마크 실행 시도 → MPS OOM 실패
- `uv run python -m rag_bench.scripts.run_all_combos --preset full --top_n 10 --layers` 실행.
- **MPS backend out of memory** (exit code 144): ColBERT 모델이 Apple Silicon GPU (18.13 GiB)에 반복 로드되면서 OOM 발생.
- 원인: `colbert.py`, `colbert_rerank.py`의 `_detect_device()`가 MPS를 자동 선택 + 72개 전략 각각이 ColBERT 모델을 새로 로드.

#### 2. MPS OOM 수정 (메인 세션 + 별도 세션 협업)
- **`config.py`**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` + `torch.set_default_device("cpu")` + `torch.mps.empty_cache()` 추가.
- **`colbert.py`** / **`colbert_rerank.py`**: `_detect_device()`에서 MPS 제거, CUDA → CPU만 사용.
- **`colbert_rerank.py`**: `shared_model` 파라미터 추가 — 외부에서 ColBERT 모델 인스턴스를 주입받아 공유.
- **`run_all_combos.py`**:
  - `IndexCacheManager._colbert_model`: ColBERT 싱글톤 캐시 (`get_colbert_model()` 메서드로 1회 로드, 이후 공유).
  - `_release_memory()`: 전략 빌드/실행 후 `gc.collect()` + `torch.mps.empty_cache()` + `torch.cuda.empty_cache()` 호출.
  - `_cleanup_strategies()`: 종료 시 메모리 캐시 해제 추가.

#### 3. HuggingFace 모델 로컬 캐시 구현
- **`config.py`**: `MODELS_DIR`, `REQUIRED_HF_MODELS` (6종), `_hf_cache_dir_name()`, `ensure_model_cache()` 추가.
  - `~/.cache/huggingface/hub`에 모델 있으면 → `rag_bench/_models/hub/`에 심링크 생성.
  - 없으면 → `HF_HOME` 설정으로 프로젝트 내부에 다운로드.
  - `setup_ssl_bypass()` 호출 시 자동 실행.
- **`scripts/prefetch_models.py`** (신규): `huggingface_hub.snapshot_download()` 기반 프리페치 스크립트.
  - `--status`: 캐시 상태 출력, `--force`: 강제 로컬 다운로드.
- **`.gitignore`**: `rag_bench/_models/` 패턴 추가.
- 검증 결과: 6개 모델 심링크 정상 생성 확인.

#### 4. RAGAS Testset Generation v2 리서치
- **`docs/research/ragas_testset_generation_v2_research.md`** 신규 작성 (304줄).
- RAGAS v0.4+ Knowledge Graph 기반 진화적 QA 생성 파이프라인 분석.
- 현행 `generate_qa.py` (20개, 단일 유형) → RAGAS v2 방식 (100개, 4종 유형) 전환 전략 수립.
- Query Types: SingleHop-Specific (35%), SingleHop-Keyphrases (15%), MultiHop-Specific (25%), MultiHop-Abstract (25%).
- 한국어 지원: `adapt_prompts("korean")` 메서드 활용.
- 예상 비용: ~$0.08 (gpt-4o-mini 기준).
- CLI: `--method ragas` / `--method legacy` / `--build-kg-only` / `--reuse-kg`.

### 커밋 (4개, 논리적 분리 완료)
```
cb5165c fix: MPS OOM 해결 — ColBERT CPU 강제 + 싱글톤 캐시 + 메모리 해제
f684e28 feat: HF 모델 로컬 캐시 — 6종 모델 심링크 + prefetch 스크립트
8460952 docs: RAGAS v2 리서치 문서 + MEMORY 세션 기록 갱신
bf321b6 perf: 벤치마크 실행 최적화 — FlashRank 싱글톤 + Pass 결과 재사용 + LLM 병렬화 + SPLADE 배치
```

#### 5. 벤치마크 실행 최적화 (HIGH 4개)
- **FlashRank 싱글톤**: `IndexCacheManager.get_flashrank_ranker()` — 24회 → 1회 ONNX 로드.
- **Pass 1→2 결과 재사용**: `BenchmarkRunner.inject_results()` — 재검색 완전 제거.
- **Answer 생성 병렬화**: `ThreadPoolExecutor(max_workers=8)` + lazy LLM 초기화.
- **SPLADE 배치 처리**: `SpladeEncoder.embed_documents()` batch_size=32.
- **검증**: `--preset quick --pass1-only` 4전략 × 20쿼리 = 80회 성공, MPS OOM 없음.

### 다음 작업
1. **72개 조합 풀 벤치마크 재실행**: `--preset full --top_n 10 --layers`
2. **QA 데이터셋 고도화 구현**: RAGAS v2 방식 `--method ragas` 구현 (리서치 완료)
3. **evaluation 메트릭 확장**: Extended 메트릭 + per-sample 점수
4. **MEDIUM 최적화** (선택): cleanup 인덱스 보존, 전략 실행 병렬화 등
5. **벤치마크 시각화 갱신**: bench_visualize.ipynb 업데이트

## 2026-02-19: Google Colab 벤치마크 환경 구축

### 주요 활동
- **`rag_bench_colab/` 디렉토리 전체 구현** (10개 파일, ~1,560줄):
  - `colab_config.py` (368줄): Colab 환경 초기화 + rag_bench monkey-patch (경로, CUDA, Qdrant 인메모리)
  - `colab_runner.py` (652줄): `ColabBenchmarkRunner` — 체크포인트 지원 2-Pass 벤치마크 래퍼
  - `colab_visualizer.py` (443줄): 8개 시각화 함수 (matplotlib/plotly/seaborn)
  - `rag_benchmark.ipynb`: 9 섹션 메인 Colab 노트북
  - `requirements_colab.txt`: Colab 전용 의존성
  - `data/`: QA 데이터셋 + 마크다운 문서 복사본
  - `README.md`: Colab 뱃지, 프리셋 테이블, 사용법

### 핵심 설계
- **Monkey-patch 접근**: rag_bench 코드 수정 없이 런타임 패치로 Colab 환경 대응
  - `patch_dense_device()`: 임베딩 모델 CPU → CUDA
  - `patch_colbert_device()`: ColBERT 모델 CPU → CUDA
  - `patch_qdrant_memory_mode()`: Qdrant `:memory:` 인메모리 지원
- **체크포인트 시스템**: 전략별 JSON → Google Drive, 커널 재시작 시 완료된 전략 스킵 (12시간 세션 제한 대응)
- **Qdrant 3모드**: ephemeral (세션 내 /content), drive (영속), memory (인메모리)

### 예상 실행시간 (T4 GPU)
| 프리셋 | 조합 수 | 총 예상 |
|--------|---------|---------|
| quick | 4 | ~15분 |
| standard | 24 | ~50분 |
| full | 72 | ~3시간 |

### 다음 작업
1. **Colab 실제 테스트**: T4 GPU에서 quick 프리셋 E2E 실행 검증
2. **README.md `<user>` 플레이스홀더**: 실제 GitHub 사용자명으로 교체
3. **72개 조합 풀 벤치마크 재실행** (로컬)
4. **QA 데이터셋 고도화**: RAGAS v2 방식 구현

## 2026-02-19: 벤치마크 수행 이력 추적 시스템 + 시각화 통합

### 주요 활동

#### 1. RunTracker 모듈 신규 구현 (`rag_bench/run_tracker.py`)
- **플랫폼 정보 수집**: OS, CPU, RAM, GPU(CUDA/MPS), Apple Silicon 칩, Git 커밋 해시 자동 감지.
- **TokenUsage 데이터 구조**: prompt/completion/total 토큰, 비용, LLM 호출 수 추적.
- **track_openai_tokens()**: LangChain `get_openai_callback()` 기반 토큰 사용량 컨텍스트 매니저.
- **StrategyTiming**: 전략별 빌드 시간, 쿼리 레이턴시 통계 (avg/min/max/p50/p95), RAGAS 점수, 인덱싱 토큰.
- **PhaseTime**: 단계별 소요 시간 + 토큰 사용량.
- **BenchmarkRunRecord**: 전체 실행 기록 (run_id, 설정, 플랫폼, 전략 타이밍, 단계 시간, 토큰 총계).
- **RunTracker 클래스**: `phase()` 컨텍스트 매니저, `start_build()`/`end_build()`, `record_query_stats()`, `record_ragas()`, `finalize()` (JSON 저장 + latest.json 심링크 + 콘솔 비중% 요약).

#### 2. 벤치마크 스크립트 통합
- **`run_all_combos.py`**: RunTracker 통합 — QA 로드, 청킹, 전략 빌드/인덱싱, Pass 1(레이턴시), Pass 2(RAGAS) 각 단계를 `tracker.phase()`로 래핑. 토큰 추적 포함.
- **`generate_qa.py`**: QA 생성 시 RunTracker + track_openai_tokens() 통합.
- **e2e_report.md**: 실행 환경, 단계별 시간(비중%), 토큰 사용량 테이블 자동 생성.

#### 3. 시각화 통합 (`colab_visualizer.py` + `bench_visualize.ipynb`)
- **plot_run_info()**: 플랫폼/설정/단계별 비중/토큰 요약 테이블 카드.
- **plot_phase_timeline()**: 단계별 가로 막대 + `{dur}s ({pct}%) [{tok} tok]` 레이블.
- **plot_build_times()**: 전략별 빌드 시간, LLM 사용 여부 색상 구분, 비중% 표시.
- **plot_token_usage()**: 단계별 토큰 파이차트 + prompt/completion 비율 막대.
- **bench_visualize.ipynb**: 섹션 10 "수행 이력 (Run History)" 추가 — `latest.json` 자동 로드 + 4개 차트.
- **display_dashboard()**: `run_record` 파라미터 추가, 있으면 수행 이력 차트 최상단 렌더링.

#### 4. 비중(%) 표시 전면 추가
- 모든 출력 지점 (plot_run_info, plot_build_times, e2e_report.md, RunTracker.finalize() 콘솔)에 전체 소요시간 대비 각 요소의 비중 표시.

### 파일 변경 목록
| 파일 | 변경 | 내용 |
|------|------|------|
| `rag_bench/run_tracker.py` | **NEW** | 수행 이력 추적 모듈 (448줄) |
| `rag_bench/scripts/run_all_combos.py` | MODIFIED | RunTracker 통합, 단계별 phase/토큰 추적 |
| `rag_bench/scripts/generate_qa.py` | MODIFIED | RunTracker + 토큰 추적 통합 |
| `rag_bench_colab/colab_visualizer.py` | MODIFIED | 시각화 함수 4종 + display_dashboard 확장 |
| `rag_bench/scripts/bench_visualize.ipynb` | MODIFIED | 섹션 10 수행 이력 추가 |
| `README.md` | MODIFIED | 수행 이력 추적 기능 설명 추가 |

### 다음 작업
1. **72개 조합 풀 벤치마크 실행** (백그라운드 진행 중)
2. **QA 데이터셋 고도화**: RAGAS v2 방식 구현
3. **evaluation 메트릭 확장**: Extended 메트릭 + per-sample 점수

## 2026-02-19: 버그 수정 + 최적화 + 기능 확장 + 문서 최신화

### 주요 활동

#### 1. 레이어별 기여도 분석 빈 출력 버그 수정
- **원인**: `runner.to_dataframe()`이 쿼리별 raw 행(`latency_ms`)을 반환하는데, `_print_layer_contribution()`/리포트/Pass2 선별에서 전략별 요약(`avg_latency` 컬럼)을 기대.
- **수정**: `_build_latency_summary()` 헬퍼 추가 — 쿼리별 raw DataFrame → 전략별 요약 DataFrame (avg/min/max/p50 레이턴시, ms→s 변환).
- 모든 참조를 `summary_df`로 교체 (`_print_layer_contribution()`, Pass 2 선별, `_generate_report()`).

#### 2. Contextual Retrieval 중복 초기화 최적화
- **원인**: `get_or_build_contextual()`이 새 DenseSparseStrategy를 생성하여 Dense/Sparse 모델 재로드 + 재인덱싱 발생.
- **수정**:
  - `IndexCacheManager.get_or_build_contextual()`: 캐시된 base 전략의 `_dense_embeddings`/`_sparse_embeddings` 객체를 ctx_base에 주입.
  - `DenseSparseStrategy._ensure_initialized()`: `elif self._client is None` 분기 추가 — 모델 주입 시 Qdrant만 초기화.
  - `ContextualRetrievalStrategy._enrich_chunks()`: 캐시 100% 히트 시 진행 로그 억제, 불필요 캐시 저장 스킵.

#### 3. RAGAS KG 기반 QA 생성 (`generate_qa.py`)
- `--method ragas` 옵션 추가: RAGAS KnowledgeGraph 기반 다양한 QA 유형 생성 (~302줄 추가).
- `--build-kg-only`, `--reuse-kg` 옵션으로 KG 사전 구축 및 재사용 지원.
- Query Types: SingleHop-Specific, SingleHop-Keyphrases, MultiHop-Specific, MultiHop-Abstract.

#### 4. EvaluationReport 통합 (`runner.py`)
- `EvaluationReport` 클래스: per-sample 리포트 + `aggregate_dict` 양방향 호환.
- `BenchmarkRunner._reports` dict + `reports` property 추가.

#### 5. Evaluation 메트릭 확장
- `metrics.py`: `COMPREHENSIVE` 프리셋, Extended 5종 추가 (context_entity_recall, response_relevancy, string_presence, exact_match, non_llm_string_similarity).
- `evaluator.py`: `comprehensive` scoring profile 추가.
- `__init__.py`: `SCORING_PROFILES`, `MetricPreset`, `create_metrics` export 추가.

#### 6. Colab 경로 수정
- `colab_config.py`, `rag_benchmark.ipynb`: 프로젝트 경로 `autorag` → `RAG-Bench` 통일.

### 커밋 히스토리
```
70e5689 fix: Colab 경로 autorag → RAG-Bench 수정
df07626 feat: evaluation 메트릭 확장 — COMPREHENSIVE 프리셋 + Extended 5종
d2759e5 feat: RAGAS KG 기반 QA 생성 + EvaluationReport 통합
216d7e4 perf: Contextual Retrieval 중복 초기화 최적화
ba25dcf fix: 레이어별 기여도 분석 빈 출력 버그 수정
5b9daf2 docs: README 및 문서 최신화 — RunTracker 수행 이력 추적 기능 반영
bccd25b feat: 벤치마크 수행 이력 추적 시스템 구현 — RunTracker + 토큰 추적 + 시각화 통합
```

### 파일 변경 목록
| 파일 | 변경 | 내용 |
|------|------|------|
| `rag_bench/scripts/run_all_combos.py` | MODIFIED | `_build_latency_summary()` 추가, Contextual 모델 공유, 레이어 분석 버그 수정 |
| `rag_bench/strategies/dense_sparse.py` | MODIFIED | `_ensure_initialized()` 조건부 초기화 분기 |
| `rag_bench/strategies/contextual_retrieval.py` | MODIFIED | 캐시 100% 히트 로그 억제 |
| `rag_bench/scripts/generate_qa.py` | MODIFIED | RAGAS KG QA 생성 (~302줄 추가) |
| `rag_bench/runner.py` | MODIFIED | EvaluationReport 통합 |
| `rag_bench/evaluation/metrics.py` | MODIFIED | COMPREHENSIVE 프리셋 + Extended 5종 |
| `rag_bench/evaluation/evaluator.py` | MODIFIED | comprehensive scoring profile |
| `rag_bench/evaluation/__init__.py` | MODIFIED | 추가 exports |
| `rag_bench_colab/colab_config.py` | MODIFIED | 경로 autorag → RAG-Bench |
| `rag_bench_colab/rag_benchmark.ipynb` | MODIFIED | 경로 autorag → RAG-Bench |

### 다음 작업
1. **72개 조합 풀 벤치마크 실행** (백그라운드 진행 중)
2. **Colab T4 GPU 실제 테스트**
3. **벤치마크 시각화 갱신**: 72개 조합 결과에 맞게 bench_visualize.ipynb 업데이트

## 2026-02-11: RAGHub 생태계 분석 및 프로젝트 컨텍스트 정립

### 주요 활동
- **RAGHub 저장소 분석**: [Andrew-Jang/RAGHub](https://github.com/Andrew-Jang/RAGHub) (1.6k+ stars) 전체 분석 완료.
  - RAG 생태계 90개+ 도구를 7개 카테고리로 분류 정리.
  - 카테고리: Frameworks(24), Evaluation(13), Engines(20+), Data Prep(3), Projects(25+), Resources(6), Leaderboards(3).

- **리서치 문서 작성**: `docs/research/raghub_ecosystem_research.md` 생성.
  - RAG 프레임워크 4개 그룹 분류 (범용 오케스트레이션, GraphRAG, 특화형, DB-Native).
  - 평가/최적화 도구 심층 분석 및 우리 프로젝트 적용 가능성 평가.
  - 2024-2025 RAG 트렌드 5가지 도출: GraphRAG 부상, 멀티모달 RAG, RAG-as-a-Service, 평가 도구 성숙, DB-Native RAG.
  - 벤치마크 추가 후보 도구 선정: Flash-Rank(리랭킹), Chonkie/zchunk(청킹), Trulens(평가).

### 프로젝트 제작 의도 및 구현 방향 정립
- **목적**: 엔터프라이즈 레벨에서 사용할 RAG 아키텍처/구성을 테스트하고 성능을 비교.
- **방법론**: `rag_bench/` 패키지에 모델별/구성별 RAG 전략을 Strategy Pattern으로 추가하고, RAGAS로 정량 비교.
- **현재 구현 상태**:
  - 구현 완료: `DenseSparseStrategy` (6가지 임베딩 조합), `RAGEvaluator`, `BenchmarkRunner`, LangGraph Agent, PDF→Markdown→Parent-Child 청킹.
  - 스텁(TODO): `ColBERTStrategy` (RAGatouille), `GraphRAGStrategy` (NodeRAG/LightRAG).
- **아키텍처**: `BaseRAGStrategy` ABC → 전략별 `index()`, `retrieve()`, `get_retriever()` 구현 → `BenchmarkRunner`로 통합 비교.

### 기존 리서치 문서 현황
| 파일 | 주제 |
|------|------|
| `docs/research/ragatouille_research.md` | ColBERT/RAGatouille Late Interaction 검색 |
| `docs/research/noderag_research.md` | NodeRAG 이질적 그래프 기반 RAG |
| `docs/research/raghub_ecosystem_research.md` | RAG 생태계 전체 조감도 (신규) |
