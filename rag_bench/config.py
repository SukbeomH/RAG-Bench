"""
전역 설정 모듈

프로젝트 디렉토리, LLM, SSL 등 전역 설정을 관리한다.
"""

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

# .env 파일 로드
from dotenv import load_dotenv

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

    import requests

    _original_request = requests.Session.request

    def _patched_request(self, *args, **kwargs):
        kwargs["verify"] = False
        return _original_request(self, *args, **kwargs)

    requests.Session.request = _patched_request  # type: ignore[method-assign]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["CURL_CA_BUNDLE"] = ""
    # os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    # SSL_CERT_FILE 동기화 (requests와 동일하게)
    if "REQUESTS_CA_BUNDLE" in os.environ:
        os.environ["SSL_CERT_FILE"] = os.environ["REQUESTS_CA_BUNDLE"]


def ensure_dirs() -> None:
    """필수 디렉토리 생성."""
    for d in (DOCS_DIR, MARKDOWN_DIR, PARENT_STORE_PATH):
        d.mkdir(parents=True, exist_ok=True)
