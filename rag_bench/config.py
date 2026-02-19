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
# 디렉토리 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
MARKDOWN_DIR = PROJECT_ROOT / "markdown"
PARENT_STORE_PATH = PROJECT_ROOT / "parent_store"
QDRANT_BASE_PATH = PROJECT_ROOT  # qdrant_db_<strategy> 접두사로 사용

# ---------------------------------------------------------------------------
# 패키지 내부 경로 (rag_bench/ 패키지를 독립 공유할 때 사용)
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).parent
BENCH_DOCS_DIR = PACKAGE_ROOT / "docs"      # 벤치마크 대상 markdown 문서
BENCH_DATA_DIR = PACKAGE_ROOT / "_benchdata"  # 벤치마크 중간 산출물
MODELS_DIR = PACKAGE_ROOT / "_models"         # 로컬 모델 캐시

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

    # MPS OOM 방지: CPU 강제 사용
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.set_default_device("cpu")
            # 기존 MPS 캐시 해제
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
