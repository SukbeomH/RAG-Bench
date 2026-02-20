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
BENCH_DATA_DIR = PACKAGE_ROOT / "_benchdata"   # 벤치마크 중간 산출물 (기준점)
BENCH_DOCS_DIR = PACKAGE_ROOT / "docs"         # 벤치마크 대상 markdown 문서
MODELS_DIR = PACKAGE_ROOT / "_models"          # 로컬 모델 캐시
MARKDOWN_DIR = BENCH_DATA_DIR / "markdown"     # 변환된 마크다운 저장 경로
PARENT_STORE_PATH = BENCH_DATA_DIR / "parent_store"  # Parent 청크 저장 경로

# ---------------------------------------------------------------------------
# 레거시 참조용 별칭 (하위 호환)
# ---------------------------------------------------------------------------
DOCS_DIR = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Qdrant 인덱스 경로 접두사
# ---------------------------------------------------------------------------
# _benchdata/ 하위 Qdrant 인덱스 디렉토리 이름 공통 접두사.
# "qdrant_db_" 문자열을 코드 전역에서 하드코딩하는 대신 이 상수를 사용한다.
QDRANT_DB_PREFIX = "qdrant_db_"

# ---------------------------------------------------------------------------
# HuggingFace 모델 레지스트리
# ---------------------------------------------------------------------------
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
# LLM 기본 설정
# ---------------------------------------------------------------------------
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_TEMPERATURE = 0

# 용도별 LLM 모델 상수
# 이 상수를 수정하면 모든 관련 컴포넌트에 일괄 적용된다.
DEFAULT_ANSWER_LLM = "gpt-4o-mini"      # 답변 생성용 (BenchmarkRunner)
DEFAULT_EVAL_LLM = "gpt-4o-mini"        # RAGAS 평가용 (ExtendedRAGEvaluator)
DEFAULT_CONTEXTUAL_LLM = "gpt-4o-mini"  # Contextual Retrieval 압축용


def setup_ssl_bypass() -> None:
    """기업 네트워크/프록시 환경을 위한 SSL 우회 설정."""
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
