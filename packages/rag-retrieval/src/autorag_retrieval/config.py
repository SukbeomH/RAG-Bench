"""
전역 설정 모듈

프로젝트 디렉토리, LLM, SSL 등 전역 설정을 관리한다.
"""

import gc
import os
import ssl
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# 프로젝트 루트 + 패키지 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# 패키지 내부 경로 — BENCH_DATA_DIR 기준으로 통합
#
# 모든 벤치마크 산출물(인덱스, 마크다운, parent_store 등)은
# BENCH_DATA_DIR (_benchdata/) 하위에 배치하여 경로 기준을 단일화한다.
# ---------------------------------------------------------------------------
BENCH_DATA_DIR = PACKAGE_ROOT / "_benchdata"  # 벤치마크 중간 산출물 (기준점)
BENCH_DOCS_DIR = PACKAGE_ROOT / "docs"  # 벤치마크 대상 markdown 문서
MODELS_DIR = PACKAGE_ROOT / "_models"  # 로컬 모델 캐시
MARKDOWN_DIR = BENCH_DATA_DIR / "markdown"  # 변환된 마크다운 저장 경로
PARENT_STORE_PATH = BENCH_DATA_DIR / "parent_store"  # Parent 청크 저장 경로

# ---------------------------------------------------------------------------
# 레거시 참조용 별칭 (하위 호환)
# ---------------------------------------------------------------------------
DOCS_DIR = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Qdrant 인덱스 경로 접두사 / 기본 컬렉션 이름
# ---------------------------------------------------------------------------
# _benchdata/ 하위 Qdrant 인덱스 디렉토리 이름 공통 접두사.
# "qdrant_db_" 문자열을 코드 전역에서 하드코딩하는 대신 이 상수를 사용한다.
QDRANT_DB_PREFIX = "qdrant_db_"
# Qdrant 기본 컬렉션 이름 — DenseSparseStrategy 생성자의 기본값으로 사용된다.
QDRANT_COLLECTION_NAME = "document_child_chunks"

# ---------------------------------------------------------------------------
# HuggingFace 모델 레지스트리 + 기본 모델 상수
# ---------------------------------------------------------------------------
# ColBERT / FlashRank 기본 모델 — CacheConfig 기본값과 이 상수가 한 곳에서 관리된다.
DEFAULT_COLBERT_MODEL = "jinaai/jina-colbert-v2"
DEFAULT_FLASHRANK_MODEL = "ms-marco-MultiBERT-L-12"

REQUIRED_HF_MODELS = [
    "BM-K/KoSimCSE-roberta-multitask",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "sentence-transformers/all-MiniLM-L6-v2",
    "naver/splade-cocondenser-ensembledistil",
    "jinaai/jina-colbert-v2",
]

# .env 파일 로드
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# LLM 제공자 설정
# LLM_PROVIDER: "ollama" (기본) 또는 "openai"
#
# 용도별 모델을 환경변수로 독립 지정 가능:
#   LLM_PROVIDER=openai              → 전체 OpenAI 전환
#   OLLAMA_BASE_URL=http://...       → Ollama 서버 주소 (기본: localhost:11434)
#   OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M  → 기본 Ollama 모델
#   OLLAMA_ANSWER_MODEL=qwen3:8b     → 답변 생성 전용 모델 (미설정 시 OLLAMA_MODEL 사용)
#   OLLAMA_CONTEXTUAL_MODEL=qwen3:4b-instruct-2507-q4_K_M → Contextual 전용 (속도 우선)
#   OLLAMA_AGENT_MODEL=qwen3:8b      → LangGraph Agent 전용 모델
#
# M2 Pro 16GB 벤치마크 동시 실행 기준 권장 설정:
#   OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M   (2.5 GB — 안전 마진 확보)
#   OLLAMA_CONTEXTUAL_MODEL=qwen3:4b-instruct-2507-q4_K_M  (청크 수백 개 처리, 속도 우선)
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "ollama")  # "ollama" | "openai"
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL: str = os.environ.get(
    "OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M"
)
# 용도별 Ollama 모델 (미설정 시 OLLAMA_DEFAULT_MODEL로 통일)
OLLAMA_ANSWER_MODEL: str = os.environ.get("OLLAMA_ANSWER_MODEL", OLLAMA_DEFAULT_MODEL)
OLLAMA_CONTEXTUAL_MODEL: str = os.environ.get("OLLAMA_CONTEXTUAL_MODEL", OLLAMA_DEFAULT_MODEL)
OLLAMA_AGENT_MODEL: str = os.environ.get("OLLAMA_AGENT_MODEL", OLLAMA_DEFAULT_MODEL)

# ---------------------------------------------------------------------------
# LLM 기본 설정
# ---------------------------------------------------------------------------
DEFAULT_LLM_MODEL = OLLAMA_DEFAULT_MODEL if LLM_PROVIDER == "ollama" else "gpt-4o-mini"
DEFAULT_LLM_TEMPERATURE = 0

# 용도별 LLM 모델 상수
DEFAULT_ANSWER_LLM = OLLAMA_ANSWER_MODEL if LLM_PROVIDER == "ollama" else "gpt-4o-mini"
DEFAULT_CONTEXTUAL_LLM = OLLAMA_CONTEXTUAL_MODEL if LLM_PROVIDER == "ollama" else "gpt-4o-mini"
DEFAULT_AGENT_LLM = OLLAMA_AGENT_MODEL if LLM_PROVIDER == "ollama" else "gpt-4o-mini"
DEFAULT_EVAL_LLM = "gpt-4o-mini"      # RAGAS 평가용 — OpenAI 고정 (평가 신뢰도)
DEFAULT_RAGAS_QA_LLM = "gpt-4o-mini"  # RAGAS QA 생성용 — OpenAI 고정 (품질)

# ---------------------------------------------------------------------------
# 실행 파라미터 상수
# ---------------------------------------------------------------------------
DEFAULT_RERANK_N = 10  # 리랭킹 후보 수 (ColBERT/FlashRank)
DEFAULT_LLM_WORKERS = 8  # LLM 답변 생성 병렬 워커 수
CONV_SUMMARY_MIN_MESSAGES = 4  # 대화 요약 최소 메시지 수
CONV_HISTORY_WINDOW = 6  # 대화 이력 window 크기


def make_llm(model: str | None = None, temperature: float = 0, **kwargs):
    """LLM_PROVIDER 설정에 따라 ChatOllama 또는 ChatOpenAI를 반환한다.

    Args:
        model: 모델명. None이면 DEFAULT_LLM_MODEL 사용.
        temperature: 샘플링 온도.
        **kwargs: 모델별 추가 파라미터.
            - max_tokens: ChatOllama의 경우 num_predict로 자동 변환.
    """
    model = model or DEFAULT_LLM_MODEL
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        # ChatOllama는 num_predict를 사용하므로 max_tokens 자동 변환
        if "max_tokens" in kwargs:
            kwargs["num_predict"] = kwargs.pop("max_tokens")
        # Ollama는 http_client 파라미터 불필요
        kwargs.pop("http_client", None)
        kwargs.pop("http_async_client", None)
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=OLLAMA_BASE_URL,
            **kwargs,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            http_client=make_http_client(),
            **kwargs,
        )


def make_http_client(verify: bool = False):
    """SSL 검증 설정이 통일된 httpx.Client를 반환한다.

    기업 네트워크/프록시 환경에서는 verify=False (기본값)로 사용한다.
    """
    import httpx
    return httpx.Client(verify=verify)


def make_async_http_client(verify: bool = False):
    """SSL 검증 설정이 통일된 httpx.AsyncClient를 반환한다."""
    import httpx
    return httpx.AsyncClient(verify=verify)


def setup_ssl_bypass() -> None:
    """
    SSL 인증서 검증을 전역으로 우회한다.

    경고: 이 함수는 프로세스 전체에 영향을 미치며 되돌릴 수 없다.
    변경 항목:
      - ssl._create_default_https_context: 인증서 미검증 컨텍스트로 교체
      - requests.Session.request: verify=False 강제 적용 (monkeypatch)
      - 환경변수: CURL_CA_BUNDLE, REQUESTS_CA_BUNDLE 등 7개 초기화

    사용처: run_service_bench.py main(), run_all_combos.py main()
    용도: 사내 네트워크 / 개발 환경에서 HuggingFace 모델 다운로드 시 SSL 오류 우회.
    주의: 라이브러리로 사용 시 이 함수를 호출하지 말 것.
    """
    warnings.filterwarnings("ignore")

    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]

    import requests  # type: ignore[import-untyped]

    _original_request = requests.Session.request

    def _patched_request(self, *args, **kwargs):
        kwargs["verify"] = False
        return _original_request(self, *args, **kwargs)

    requests.Session.request = _patched_request  # type: ignore[method-assign]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # MPS OOM 방지: 공식 환경 변수로 제어
    # - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0: MPS 메모리 상한 제거
    # - PYTORCH_ENABLE_MPS_FALLBACK=1: MPS 미지원 연산을 CPU로 자동 폴백
    # detect_device()가 이미 MPS를 제외하므로 자체 코드에서는 MPS가 선택되지 않음.
    # 외부 라이브러리가 MPS를 시도할 경우에도 OOM 대신 CPU로 안전하게 폴백한다.
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            gc.collect()
    except Exception:
        pass
    # os.environ["CURL_CA_BUNDLE"] = ""
    # os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # SSL_CERT_FILE 동기화 (requests와 동일하게)
    if "REQUESTS_CA_BUNDLE" in os.environ:
        os.environ["SSL_CERT_FILE"] = os.environ["REQUESTS_CA_BUNDLE"]

    ensure_model_cache()


def _hf_cache_dir_name(model_id: str) -> str:
    """HuggingFace 캐시 디렉토리명 변환 (예: 'BAAI/bge-m3' → 'models--BAAI--bge-m3')."""
    return f"models--{model_id.replace('/', '--')}"


def ensure_model_cache() -> None:
    """프로젝트 로컬 모델 캐시 설정.

    ~/.cache/huggingface/hub 에 모델이 있으면 심링크로 재사용하고,
    없으면 HF_HOME을 로컬로 설정하여 이후 다운로드가 프로젝트 내부에 저장되게 한다.
    """
    local_hub = MODELS_DIR / "hub"
    local_hub.mkdir(parents=True, exist_ok=True)

    hf_default_cache = Path.home() / ".cache" / "huggingface" / "hub"

    linked = 0
    for model_id in REQUIRED_HF_MODELS:
        dir_name = _hf_cache_dir_name(model_id)
        local_path = local_hub / dir_name
        hf_cache_path = hf_default_cache / dir_name

        if local_path.exists():
            continue

        if hf_cache_path.exists():
            local_path.symlink_to(hf_cache_path)
            linked += 1

    os.environ["HF_HOME"] = str(MODELS_DIR)

    if linked > 0:
        print(f"[모델 캐시] {linked}개 모델 심링크 생성 → {MODELS_DIR}")


def ensure_dirs() -> None:
    """필수 디렉토리 생성."""
    for d in (DOCS_DIR, MARKDOWN_DIR, PARENT_STORE_PATH):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 환경변수 오버라이드 (Colab 등 외부 환경에서 경로 변경)
# import 시점에 환경변수가 설정되어 있으면 자동으로 반영된다.
# ---------------------------------------------------------------------------
_raw = os.environ.get("RAG_BENCH_DATA_DIR")
if _raw:
    BENCH_DATA_DIR = Path(_raw)

_raw = os.environ.get("RAG_BENCH_DOCS_DIR")
if _raw:
    BENCH_DOCS_DIR = Path(_raw)

_raw = os.environ.get("RAG_BENCH_DOCS_SRC")
if _raw:
    DOCS_DIR = Path(_raw)

del _raw
