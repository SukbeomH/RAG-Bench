# 환경 구성 및 설정 가이드 (Environment & Setup Guide)

이 문서는 `rag_bench` 프로젝트 실행을 위한 상세 환경 구성 및 설정 방법을 기술합니다.

## 시스템 요구사항 (System Requirements)

- **OS**: macOS, Linux, Windows (WSL2 권장)
- **Container**: Docker Desktop (Qdrant 벡터 DB 실행용)
- **Java**: JDK 11+ (KoNLPy 등 한국어 전처리기용, 선택 사항)

### 필수 프레임워크 및 버전

다음 버전 조건을 **반드시 준수**해야 합니다:

1.  **패키지 매니저**: **`uv`** (필수)
    - 본 프로젝트는 `uv`를 통한 빠른 의존성 관리를 전제로 합니다.
2.  **Python 버전**: **3.12 이상**
    - 최신 언어 기능 및 타입 힌트 지원을 위해 필수입니다.
3.  **LangChain 버전**: **1.0.0 이상** (`langchain-core` >= 0.3)
    - 최신 안정화 버전의 인터페이스를 따릅니다.

## 설치 (Installation)

이 프로젝트는 `uv` 패키지 매니저를 사용합니다.

```bash
# 1. uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 의존성 설치
uv sync
```

## 환경 변수 설정 (Environment Variables)

프로젝트 루트의 `.env` 파일을 통해 설정을 관리합니다. `python-dotenv`가 이를 자동으로 로드합니다.

### 필수 변수 (Required)

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `OPENAI_API_KEY` | LLM(GPT-4o) 및 임베딩 사용을 위한 OpenAI 키 | `sk-proj-...` |

### 선택 변수 (Optional)

| 변수명 | 설명 | 기본값 / 예시 |
|--------|------|---------------|
| `QDRANT_URL` | Qdrant 서버 주소 | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant 접속 키 (로컬 실행 시 불필요) | `your-api-key` |
| `HF_TOKEN` | HuggingFace 모델 다운로드 토큰 (일부 모델 접근 시) | `hf_...` |
| `SSL_CERT_FILE` | 사설 CA 인증서 경로 (기업 네트워크) | `/path/to/cert.pem` |
| `REQUESTS_CA_BUNDLE` | Python Requests 라이브러리용 CA 번들 | `/path/to/cert.pem` |

### 자동 적용되는 설정 (Auto-configured)

다음 설정은 `rag_bench/config.py` 및 코드 레벨에서 자동으로 적용되므로, **별도로 설정할 필요가 없습니다**.

- `HF_HUB_DISABLE_SSL_VERIFY=1`: HuggingFace Hub SSL 검증 비활성화 (보안망 대응)
- `TOKENIZERS_PARALLELISM=false`: 토크나이저 병렬 처리 경고 억제
- `SSL_CERT_FILE` ← `REQUESTS_CA_BUNDLE`: `requests` 설정이 있으면 자동으로 SSL 인증서 경로로 동기화

## 네트워크 및 SSL 처리 (Network & SSL)

기업 보안 네트워크(Proxy, SSL Inspection) 환경에서의 실행을 위해 다음과 같은 **우회 로직**이 내장되어 있습니다:

1. **HuggingFace Hub**:
   - `rag_bench/config.py` 로딩 시 SSL 검증을 강제로 비활성화합니다.
   - 인증서 오류(`SSLCertVerificationError`)를 방지합니다.

2. **OpenAI API (LangChain)**:
   - `rag_bench/evaluation.py`의 `RAGEvaluator` 초기화 시, `httpx.Client(verify=False)`를 주입하여 SSL 검증을 건너뜁니다.
   - `OPENAI_API_KEY`만 올바르다면 프록시 환경에서도 동작하도록 구성되었습니다.

3. **Custom CA**:
   - 만약 사설 인증서를 사용해야 한다면 `.env`에 `REQUESTS_CA_BUNDLE` 경로를 지정하십시오.

## 실행 방법 (Usage)

환경 설정 후 다음 스크립트로 동작을 검증할 수 있습니다.

```bash
# 환경 설정 및 Ragas 평가 파이프라인 검증 (Mock 데이터 사용)
uv run python scripts/verify_ragas_eval.py

# 벤치마크 실행 (Dense 검색)
uv run python scripts/run_benchmark.py
```
