"""
HuggingFace 모델 사전 다운로드 스크립트.

프로젝트에서 사용하는 모든 HF 모델을 로컬 캐시(_models/)에 미리 다운로드한다.
~/.cache/huggingface/hub에 이미 있는 모델은 심링크로 재사용한다.

Usage:
    python -m rag_bench.scripts.prefetch_models           # 심링크 우선, 없으면 다운로드
    python -m rag_bench.scripts.prefetch_models --force    # 심링크 무시, 로컬에 직접 다운로드
    python -m rag_bench.scripts.prefetch_models --status   # 현재 캐시 상태만 출력
"""

import argparse
import sys
import time
from pathlib import Path

from rag_bench.config import (
    MODELS_DIR,
    REQUIRED_HF_MODELS,
    _hf_cache_dir_name,
    ensure_model_cache,
    setup_ssl_bypass,
)


def _model_size_mb(model_path: Path) -> float:
    """디렉토리 내 파일 총 크기 (MB)."""
    if not model_path.exists():
        return 0.0
    total = 0
    target = model_path.resolve() if model_path.is_symlink() else model_path
    for f in target.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def print_status():
    """현재 모델 캐시 상태 출력."""
    local_hub = MODELS_DIR / "hub"
    hf_default = Path.home() / ".cache" / "huggingface" / "hub"

    print(f"\n{'═' * 70}")
    print(" 모델 캐시 상태")
    print(f"{'═' * 70}")
    print(f"  로컬 캐시: {MODELS_DIR}")
    print(f"  HF 기본:   {hf_default}")
    print(f"{'─' * 70}")
    print(f"  {'모델 ID':<45} {'상태':<12} {'크기':>8}")
    print(f"{'─' * 70}")

    for model_id in REQUIRED_HF_MODELS:
        dir_name = _hf_cache_dir_name(model_id)
        local_path = local_hub / dir_name
        hf_path = hf_default / dir_name

        if local_path.is_symlink():
            size = _model_size_mb(local_path)
            status = "심링크"
        elif local_path.exists():
            size = _model_size_mb(local_path)
            status = "로컬"
        elif hf_path.exists():
            size = _model_size_mb(hf_path)
            status = "HF캐시만"
        else:
            size = 0.0
            status = "미다운로드"

        print(f"  {model_id:<45} {status:<12} {size:>7.1f}MB")

    print(f"{'─' * 70}")


def prefetch(force: bool = False):
    """모든 모델 다운로드."""
    setup_ssl_bypass()

    if not force:
        ensure_model_cache()

    local_hub = MODELS_DIR / "hub"
    local_hub.mkdir(parents=True, exist_ok=True)

    if force:
        import os
        os.environ["HF_HOME"] = str(MODELS_DIR)

    from huggingface_hub import snapshot_download

    print(f"\n{'═' * 60}")
    print(f" 모델 프리페치 — {'강제 로컬 다운로드' if force else '심링크 우선'}")
    print(f"{'═' * 60}")

    for i, model_id in enumerate(REQUIRED_HF_MODELS, 1):
        dir_name = _hf_cache_dir_name(model_id)
        local_path = local_hub / dir_name

        if not force and local_path.exists():
            print(f"  [{i}/{len(REQUIRED_HF_MODELS)}] {model_id} — 이미 존재 (건너뜀)")
            continue

        if force and local_path.is_symlink():
            local_path.unlink()

        print(f"  [{i}/{len(REQUIRED_HF_MODELS)}] {model_id} — 다운로드 중...")
        t0 = time.time()
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(local_hub),
            )
            elapsed = time.time() - t0
            print(f"    ✓ 완료 ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ✗ 실패 ({elapsed:.1f}s): {e}")

    print(f"\n{'═' * 60}")
    print_status()


def main():
    parser = argparse.ArgumentParser(description="HuggingFace 모델 사전 다운로드")
    parser.add_argument("--force", action="store_true",
                        help="심링크 무시, 로컬에 직접 다운로드")
    parser.add_argument("--status", action="store_true",
                        help="현재 캐시 상태만 출력")
    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    prefetch(force=args.force)


if __name__ == "__main__":
    main()
