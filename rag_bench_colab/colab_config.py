"""
Colab 환경 설정 + rag_bench config 패치.

Google Colab에서 rag_bench를 실행하기 위한 경로/환경 오버라이드.
"""

import gc
import os
import warnings
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------

COLAB_PROJECT_ROOT = Path("/content/RAG-Bench")
COLAB_DATA_DIR = COLAB_PROJECT_ROOT / "rag_bench_colab" / "data"
COLAB_DOCS_DIR = COLAB_DATA_DIR / "docs"   # 벤치마크용 .md 파일 위치
COLAB_PDF_DIR = COLAB_PROJECT_ROOT / "docs"  # PDF 원본 파일 위치 (샘플링 소스)

# Google Drive 경로 (영속 저장)
DRIVE_BASE = Path("/content/drive/MyDrive/rag_bench_colab")
DRIVE_BENCHDATA_DIR = DRIVE_BASE / "_benchdata"
DRIVE_MODELS_DIR = DRIVE_BASE / "models"
DRIVE_CHECKPOINTS_DIR = DRIVE_BASE / "checkpoints"
DRIVE_RESULTS_DIR = DRIVE_BASE / "results"

# Qdrant: 에페메럴 (세션 내 로컬) 또는 Drive
QDRANT_EPHEMERAL_BASE = Path("/content/qdrant_workspace")

# ---------------------------------------------------------------------------
# Colab 전용 LLM 상수
# ---------------------------------------------------------------------------

# Colab 환경에서는 비용 절감을 위해 경량 모델을 기본으로 사용한다.
# rag_bench.config의 DEFAULT_*_LLM을 오버라이드하는 역할.
DEFAULT_COLAB_ANSWER_LLM = "gpt-4o-mini"    # 답변 생성용 (colab_runner)
DEFAULT_COLAB_EVAL_LLM = "gpt-4o-mini"      # RAGAS 평가용 (colab_runner)


# ---------------------------------------------------------------------------
# 환경 감지
# ---------------------------------------------------------------------------


def is_colab() -> bool:
    """Google Colab 환경 여부."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def get_device() -> str:
    """CUDA 사용 가능 시 'cuda', 아니면 'cpu'."""
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
# Colab 환경 초기화
# ---------------------------------------------------------------------------


def setup_colab_env(mount_drive: bool = True) -> dict:
    """Colab 환경을 초기화한다.

    1. Google Drive 마운트
    2. Colab Secrets에서 OPENAI_API_KEY 로드
    3. HF_HOME을 Drive로 설정
    4. 필수 디렉토리 생성

    Returns:
        설정 요약 dict.
    """
    info = {"is_colab": is_colab(), "device": get_device()}

    # 1. Google Drive 마운트
    if mount_drive and is_colab():
        try:
            from google.colab import drive

            drive.mount("/content/drive")
            info["drive_mounted"] = True
        except Exception as e:
            print(f"[Warning] Drive 마운트 실패: {e}")
            info["drive_mounted"] = False
    else:
        info["drive_mounted"] = False

    # 2. API Key 로드 (Colab Secrets)
    if is_colab():
        try:
            from google.colab import userdata

            api_key = userdata.get("OPENAI_API_KEY")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                print("[API Key] Colab Secrets에서 OPENAI_API_KEY 로드 완료")

            upstage_key = userdata.get("UPSTAGE_API_KEY")
            if upstage_key:
                os.environ["UPSTAGE_API_KEY"] = upstage_key
                print("[API Key] Colab Secrets에서 UPSTAGE_API_KEY 로드 완료")
        except Exception:
            print("[Warning] Colab Secrets 접근 실패. Cell 1.4~1.5에서 수동 입력하세요.")

    info["api_key_loaded"] = "OPENAI_API_KEY" in os.environ
    info["upstage_api_key_loaded"] = "UPSTAGE_API_KEY" in os.environ

    # 3. HF_HOME 설정 (Drive 사용 시 영속 캐시)
    if info["drive_mounted"]:
        DRIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(DRIVE_MODELS_DIR)
        info["hf_home"] = str(DRIVE_MODELS_DIR)
    else:
        os.environ["HF_HOME"] = "/content/hf_cache"
        Path("/content/hf_cache").mkdir(parents=True, exist_ok=True)
        info["hf_home"] = "/content/hf_cache"

    # 4. 디렉토리 생성
    for d in [
        DRIVE_BENCHDATA_DIR,
        DRIVE_CHECKPOINTS_DIR,
        DRIVE_RESULTS_DIR,
        QDRANT_EPHEMERAL_BASE,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # 5. 공통 환경 변수
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")

    print("\n[Setup] 환경 설정 완료:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    return info


# ---------------------------------------------------------------------------
# rag_bench config 패치
# ---------------------------------------------------------------------------


def patch_rag_bench_config(qdrant_mode: str = "ephemeral") -> None:
    """rag_bench.config 모듈의 경로를 Colab 환경으로 오버라이드.

    Args:
        qdrant_mode: 'ephemeral' (세션 내 /content), 'drive' (Google Drive), 'memory' (인메모리)
    """
    # 환경변수 설정 (config 모듈이 아직 import되지 않은 경우에도 자동 반영)
    os.environ["RAG_BENCH_DATA_DIR"] = str(DRIVE_BENCHDATA_DIR)
    os.environ["RAG_BENCH_DOCS_DIR"] = str(COLAB_DOCS_DIR)
    os.environ["RAG_BENCH_DOCS_SRC"] = str(COLAB_PDF_DIR)

    import rag_bench.config as cfg

    # 이미 import된 모듈의 속성도 직접 업데이트 (하위 호환)
    cfg.DOCS_DIR = COLAB_PDF_DIR          # PDF 원본 경로 (샘플링 소스)
    cfg.BENCH_DOCS_DIR = COLAB_DOCS_DIR   # 벤치마크용 .md 파일
    cfg.BENCH_DATA_DIR = DRIVE_BENCHDATA_DIR
    cfg.MODELS_DIR = DRIVE_MODELS_DIR

    # parent_store는 _benchdata 하위
    parent_store = DRIVE_BENCHDATA_DIR / "parent_store"
    parent_store.mkdir(parents=True, exist_ok=True)

    # ensure_model_cache를 no-op으로 대체 (Drive가 HF_HOME)
    cfg.ensure_model_cache = lambda: None

    # SSL bypass는 Colab에서 불필요하지만 호환성 유지
    def _colab_ssl_bypass():
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        # MPS 관련 로직은 Colab에서 불필요 (CUDA 환경)

    cfg.setup_ssl_bypass = _colab_ssl_bypass

    # run_all_combos 모듈의 값 복사된 변수도 패치 (import 시 값이 복사되므로)
    try:
        import rag_bench.scripts.run_all_combos as rac

        rac.BENCH_DATA_DIR = DRIVE_BENCHDATA_DIR
        rac.BENCH_DOCS_DIR = COLAB_DOCS_DIR
    except ImportError:
        pass  # 아직 로드되지 않았으면 무시 (이후 import 시 cfg 값 사용)

    # generate_qa 모듈의 import-time 복사 변수도 패치
    try:
        import rag_bench.scripts.generate_qa as gqa

        gqa.BENCH_DATA_DIR = DRIVE_BENCHDATA_DIR
        gqa.BENCH_DOCS_DIR = COLAB_DOCS_DIR
        gqa.DOCS_DIR = COLAB_PDF_DIR
        gqa.KG_SAVE_PATH = DRIVE_BENCHDATA_DIR / "ragas_knowledge_graph.json"
    except ImportError:
        pass  # 이후 import 시 cfg 값 사용

    print("[Patch] rag_bench.config 패치 완료:")
    print(f"  DOCS_DIR       → {cfg.DOCS_DIR}")
    print(f"  BENCH_DOCS_DIR → {cfg.BENCH_DOCS_DIR}")
    print(f"  BENCH_DATA_DIR → {cfg.BENCH_DATA_DIR}")
    print(f"  MODELS_DIR     → {cfg.MODELS_DIR}")
    print(f"  Qdrant mode    → {qdrant_mode}")


# ---------------------------------------------------------------------------
# Qdrant 경로/모드 헬퍼
# ---------------------------------------------------------------------------


def get_qdrant_path(
    dense: str, sparse: str, qdrant_mode: str = "ephemeral", contextual: bool = False
) -> str:
    """Qdrant 모드별 저장 경로를 반환한다.

    Args:
        dense: Dense 모델 키 (e.g., 'bge-m3')
        sparse: Sparse 모델 키 (e.g., 'korean_bm25')
        qdrant_mode: 'ephemeral', 'drive', 'memory'
        contextual: contextual retrieval 인덱스 여부

    Returns:
        Qdrant path 문자열. ':memory:' 인 경우 인메모리 모드.
    """
    if qdrant_mode == "memory":
        return ":memory:"

    prefix = "qdrant_db_ctx_" if contextual else "qdrant_db_"
    dir_name = f"{prefix}{dense}_{sparse}"

    if qdrant_mode == "drive":
        return str(DRIVE_BENCHDATA_DIR / dir_name)
    else:  # ephemeral
        return str(QDRANT_EPHEMERAL_BASE / dir_name)


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


# ---------------------------------------------------------------------------
# Dense 모델 디바이스 패치
# ---------------------------------------------------------------------------


def _patch_hf_hub_for_colab() -> None:
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
        print("[Patch] huggingface_hub.list_repo_tree → additional_chat_templates 404 억제")
    except Exception as e:
        print(f"[Warning] HF Hub 패치 실패 (무시): {e}")


def patch_dense_device(device: str = "cuda") -> None:
    """[DEPRECATED] DenseSparseStrategy(device=) 파라미터를 직접 사용하세요.

    이 함수는 이전 버전 호환성을 위해 유지되지만 더 이상 monkey-patch를 수행하지 않습니다.
    CacheConfig(dense_device=device) 또는 DenseSparseStrategy(device=device)를 사용하세요.
    """
    import warnings
    warnings.warn(
        "patch_dense_device()는 deprecated입니다. "
        "DenseSparseStrategy(device=) 파라미터를 직접 사용하세요.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(f"[Deprecated] patch_dense_device('{device}') — DI 패턴으로 대체됨. device='{device}' 파라미터를 직접 전달하세요.")


# ---------------------------------------------------------------------------
# Qdrant 인메모리 패치
# ---------------------------------------------------------------------------


def patch_qdrant_memory_mode() -> None:
    """[DEPRECATED] DenseSparseStrategy(qdrant_path=':memory:')가 자동으로 지원됩니다.

    이 함수는 이전 버전 호환성을 위해 유지되지만 더 이상 monkey-patch를 수행하지 않습니다.
    """
    import warnings
    warnings.warn(
        "patch_qdrant_memory_mode()는 deprecated입니다. "
        "DenseSparseStrategy(qdrant_path=':memory:')가 자동으로 지원됩니다.",
        DeprecationWarning,
        stacklevel=2,
    )
    print("[Deprecated] patch_qdrant_memory_mode() — _init_qdrant() 내장 처리됨.")


# ---------------------------------------------------------------------------
# CacheConfig 헬퍼
# ---------------------------------------------------------------------------


def get_cache_config(device: str = "cuda") -> "CacheConfig":
    """Colab 환경에 맞는 CacheConfig를 반환한다.

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
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 통합 초기화
# ---------------------------------------------------------------------------


def init_colab(
    qdrant_mode: str = "ephemeral",
    device: Optional[str] = None,
    mount_drive: bool = True,
) -> dict:
    """Colab 환경 전체 초기화 — setup + patch를 한 번에 실행.

    Args:
        qdrant_mode: 'ephemeral', 'drive', 'memory'
        device: 'cuda', 'cpu', None (자동 감지)
        mount_drive: Google Drive 마운트 여부

    Returns:
        설정 요약 dict.
    """
    info = setup_colab_env(mount_drive=mount_drive)

    if device is None:
        device = info["device"]

    patch_rag_bench_config(qdrant_mode=qdrant_mode)

    # HF Hub 패치 (additional_chat_templates 404 억제)
    _patch_hf_hub_for_colab()

    # patch_dense_device(device)  — 제거됨 (DenseSparseStrategy 생성자 device= 파라미터로 대체)
    # patch_qdrant_memory_mode()  — 제거됨 (_init_qdrant 내장 처리)

    # 한글 폰트 설정
    _setup_korean_font()

    info["qdrant_mode"] = qdrant_mode
    info["patched"] = True

    print(f"\n{'=' * 60}")
    print(" Colab 환경 초기화 완료")
    print(f"{'=' * 60}")

    return info
