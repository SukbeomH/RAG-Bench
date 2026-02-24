"""
로컬 Jupyter 환경 설정 + rag_bench config 패치.

로컬 환경에서 rag_bench를 실행하기 위한 경로/환경 설정.
(Google Colab의 colab_config.py에서 마이그레이션)
"""

import gc
import os
import warnings
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------

# 프로젝트 루트 (autorag/)
PROJECT_ROOT = Path(__file__).parent.parent
# rag_bench 패키지 경로
RAG_BENCH_ROOT = PROJECT_ROOT / "rag_bench"

# 로컬 데이터 경로
LOCAL_DATA_DIR = Path(__file__).parent / "data"
LOCAL_DOCS_DIR = LOCAL_DATA_DIR / "docs"  # 벤치마크용 .md 파일 위치
PDF_DIR = PROJECT_ROOT / "docs"  # PDF 원본 파일 위치 (샘플링 소스)

# 벤치마크 데이터 디렉토리 (_benchdata)
BENCHDATA_DIR = RAG_BENCH_ROOT / "_benchdata"
MODELS_DIR = RAG_BENCH_ROOT / "_models"
CHECKPOINTS_DIR = BENCHDATA_DIR / "checkpoints"
RESULTS_DIR = BENCHDATA_DIR / "results"

# Qdrant 로컬 저장 경로
QDRANT_LOCAL_BASE = BENCHDATA_DIR

# ---------------------------------------------------------------------------
# LLM 상수
# ---------------------------------------------------------------------------

DEFAULT_ANSWER_LLM = "gpt-4o-mini"  # 답변 생성용
DEFAULT_EVAL_LLM = "gpt-4o-mini"  # RAGAS 평가용


# ---------------------------------------------------------------------------
# 환경 감지
# ---------------------------------------------------------------------------


def is_notebook() -> bool:
    """Jupyter/IPython 노트북 환경 여부."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        shell_name = type(shell).__name__
        # ZMQInteractiveShell = Jupyter, TerminalInteractiveShell = IPython
        return shell_name == "ZMQInteractiveShell"
    except (ImportError, NameError):
        return False


def get_device() -> str:
    """사용 가능한 최적 디바이스를 반환한다 (CUDA > CPU, MPS 제외)."""
    try:
        from rag_bench.utils.device import detect_device

        device = detect_device()
        print(f"[Device] {device.upper()}")
        return device
    except ImportError:
        pass

    # fallback: 직접 감지
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[Device] CUDA: {name}")
            return "cuda"
    except ImportError:
        pass
    print("[Device] CPU")
    return "cpu"


# ---------------------------------------------------------------------------
# 로컬 환경 초기화
# ---------------------------------------------------------------------------


def setup_local_env() -> dict:
    """로컬 Jupyter 환경을 초기화한다.

    1. .env 파일에서 API 키 로드
    2. HF_HOME 설정
    3. 필수 디렉토리 생성

    Returns:
        설정 요약 dict.
    """
    info = {"is_notebook": is_notebook(), "device": get_device()}

    # 1. .env 파일에서 API Key 로드
    try:
        from dotenv import load_dotenv

        # 프로젝트 루트의 .env 파일 로드
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
            print(f"[API Key] .env 파일 로드 완료: {env_path}")
        else:
            print(f"[Warning] .env 파일 없음: {env_path}")
            print("  → .env 파일을 생성하고 OPENAI_API_KEY=sk-... 를 설정하세요.")
    except ImportError:
        print("[Warning] python-dotenv 미설치. pip install python-dotenv")

    info["api_key_loaded"] = "OPENAI_API_KEY" in os.environ
    info["upstage_api_key_loaded"] = "UPSTAGE_API_KEY" in os.environ

    if info["api_key_loaded"]:
        print("[API Key] OPENAI_API_KEY 로드 완료")
    if info["upstage_api_key_loaded"]:
        print("[API Key] UPSTAGE_API_KEY 로드 완료")

    # 2. HF_HOME 설정 (기존 rag_bench/_models 또는 시스템 기본값 사용)
    if "HF_HOME" not in os.environ:
        if MODELS_DIR.exists():
            os.environ["HF_HOME"] = str(MODELS_DIR)
            info["hf_home"] = str(MODELS_DIR)
        else:
            info["hf_home"] = "system default"
    else:
        info["hf_home"] = os.environ["HF_HOME"]

    # 3. 디렉토리 생성
    for d in [BENCHDATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 4. 공통 환경 변수
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")

    print("\n[Setup] 환경 설정 완료:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    return info


# ---------------------------------------------------------------------------
# rag_bench config 패치
# ---------------------------------------------------------------------------


def patch_rag_bench_config(qdrant_mode: str = "local") -> None:
    """rag_bench.config 모듈의 경로를 로컬 환경에 맞게 설정.

    Args:
        qdrant_mode: 'local' (로컬 파일 저장), 'memory' (인메모리)
    """
    import rag_bench.config as cfg

    # BENCH_DOCS_DIR: 노트북 전용 docs가 있으면 사용, 없으면 rag_bench 기본값 유지
    if LOCAL_DOCS_DIR.exists() and list(LOCAL_DOCS_DIR.glob("*.md")):
        cfg.BENCH_DOCS_DIR = LOCAL_DOCS_DIR

    # 나머지 경로는 rag_bench.config 기본값을 그대로 활용
    # (로컬에서는 rag_bench/_benchdata/ 를 기준으로 동작)

    # SSL bypass는 로컬에서도 기업 프록시 환경에서 필요할 수 있으므로 유지
    # (config.py의 setup_ssl_bypass 기본 구현 사용)

    # run_all_combos / generate_qa 모듈 패치 (import-time 복사 변수 대응)
    try:
        import rag_bench.scripts.run_all_combos as rac

        rac.BENCH_DATA_DIR = cfg.BENCH_DATA_DIR
        rac.BENCH_DOCS_DIR = cfg.BENCH_DOCS_DIR
    except ImportError:
        pass

    try:
        import rag_bench.scripts.generate_qa as gqa

        gqa.BENCH_DATA_DIR = cfg.BENCH_DATA_DIR
        gqa.BENCH_DOCS_DIR = cfg.BENCH_DOCS_DIR
        gqa.DOCS_DIR = cfg.DOCS_DIR
        gqa.KG_SAVE_PATH = cfg.BENCH_DATA_DIR / "ragas_knowledge_graph.json"
    except ImportError:
        pass

    print("[Patch] rag_bench.config 설정:")
    print(f"  DOCS_DIR       → {cfg.DOCS_DIR}")
    print(f"  BENCH_DOCS_DIR → {cfg.BENCH_DOCS_DIR}")
    print(f"  BENCH_DATA_DIR → {cfg.BENCH_DATA_DIR}")
    print(f"  Qdrant mode    → {qdrant_mode}")


# ---------------------------------------------------------------------------
# Qdrant 경로/모드 헬퍼
# ---------------------------------------------------------------------------


def get_qdrant_path(
    dense: str, sparse: str, qdrant_mode: str = "local", contextual: bool = False
) -> str:
    """Qdrant 모드별 저장 경로를 반환한다.

    Args:
        dense: Dense 모델 키 (e.g., 'bge-m3')
        sparse: Sparse 모델 키 (e.g., 'korean_bm25')
        qdrant_mode: 'local' (로컬 파일), 'memory' (인메모리)
        contextual: contextual retrieval 인덱스 여부

    Returns:
        Qdrant path 문자열. ':memory:' 인 경우 인메모리 모드.
    """
    if qdrant_mode == "memory":
        return ":memory:"

    prefix = "qdrant_db_ctx_" if contextual else "qdrant_db_"
    dir_name = f"{prefix}{dense}_{sparse}"
    return str(QDRANT_LOCAL_BASE / dir_name)


# ---------------------------------------------------------------------------
# 한글 폰트 설정
# ---------------------------------------------------------------------------


def _setup_korean_font() -> None:
    """matplotlib 한글 폰트 설정 (koreanize-matplotlib 사용)."""
    try:
        import koreanize_matplotlib  # noqa: F401

        print("[Font] 한글 폰트 설정 완료 (NanumGothic)")
    except ImportError:
        print("[Warning] koreanize-matplotlib 미설치 — 한글 그래프가 깨질 수 있습니다.")
        print("  → pip install koreanize-matplotlib")


# ---------------------------------------------------------------------------
# HuggingFace Hub 패치
# ---------------------------------------------------------------------------


def _patch_hf_hub() -> None:
    """HuggingFace Hub의 additional_chat_templates 404 에러를 억제한다.

    sentence-transformers 3.x가 구형 모델에 없는 additional_chat_templates를
    조회하면서 EntryNotFoundError를 발생시키는 문제를 방지한다.
    """
    try:
        import huggingface_hub
        from huggingface_hub.errors import EntryNotFoundError

        _orig = huggingface_hub.list_repo_tree

        def _safe_list_repo_tree(repo_id, path_in_repo="", **kwargs):
            try:
                return _orig(repo_id, path_in_repo, **kwargs)
            except EntryNotFoundError:
                if "additional_chat_templates" in str(path_in_repo):
                    return iter([])
                raise

        huggingface_hub.list_repo_tree = _safe_list_repo_tree
        print(
            "[Patch] huggingface_hub.list_repo_tree → additional_chat_templates 404 억제"
        )
    except Exception as e:
        print(f"[Warning] HF Hub 패치 실패 (무시): {e}")


# ---------------------------------------------------------------------------
# CacheConfig 헬퍼
# ---------------------------------------------------------------------------


def get_cache_config(device: str = "cpu") -> "CacheConfig":
    """로컬 환경에 맞는 CacheConfig를 반환한다.

    Args:
        device: ColBERT 및 Dense 임베딩 디바이스 ('cuda' 또는 'cpu').

    Returns:
        CacheConfig 인스턴스.
    """
    from rag_bench.combo import CacheConfig

    return CacheConfig(colbert_device=device, dense_device=device)


# ---------------------------------------------------------------------------
# 메모리 관리
# ---------------------------------------------------------------------------


def release_memory():
    """GPU/CPU 메모리 강제 해제."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"[Memory] CUDA cache cleared. Allocated: {allocated:.0f}MB")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS: 명시적 cache clear API 없음, GC만 수행
            pass
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 통합 초기화
# ---------------------------------------------------------------------------


def init_local(
    qdrant_mode: str = "local",
    device: Optional[str] = None,
) -> dict:
    """로컬 Jupyter 환경 전체 초기화 — setup + patch를 한 번에 실행.

    Args:
        qdrant_mode: 'local' (로컬 파일), 'memory' (인메모리)
        device: 'cuda', 'cpu', None (자동 감지)

    Returns:
        설정 요약 dict.
    """
    info = setup_local_env()

    if device is None:
        device = info["device"]

    patch_rag_bench_config(qdrant_mode=qdrant_mode)

    # HF Hub 패치 (additional_chat_templates 404 억제)
    _patch_hf_hub()

    # 한글 폰트 설정
    _setup_korean_font()

    info["qdrant_mode"] = qdrant_mode
    info["patched"] = True

    print(f"\n{'=' * 60}")
    print(" 로컬 환경 초기화 완료")
    print(f"{'=' * 60}")

    return info
