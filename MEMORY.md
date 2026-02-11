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
   - `autorag` 패키지가 구버전 `langchain-core`를 강제하여 충돌 발생.
   - `pyproject.toml`에서 `autorag` 의존성을 제거하고 `langchain>=1.0` 등을 명시하여 해결.

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
- **방법론**: `rag_bench/` 패키지에 모델별/구성별 RAG 전략을 Strategy Pattern으로 추가하고, AutoRAG + ragas로 정량 비교.
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
