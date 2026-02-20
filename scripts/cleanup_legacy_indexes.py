"""
레거시 Qdrant 인덱스 정리 스크립트.

rag_bench/_benchdata/ 하위에 존재하는 레거시 combo_id 기반 인덱스
(qdrant_db_combo1, qdrant_db_combo2, qdrant_db_combo3, qdrant_db_combo4)를
삭제한다.

이 디렉토리들은 이전의 하위 호환(combo_id) 방식에서 생성된 것으로,
현재의 (dense_model + sparse_type) 독립 파라미터 방식에서는 사용되지 않는다.

기본 실행은 dry-run 모드로, 삭제 대상만 출력하고 실제 삭제는 수행하지 않는다.
실제 삭제를 수행하려면 --execute 플래그를 명시적으로 지정해야 한다.

Usage:
    # dry-run (기본): 삭제 대상 목록만 출력
    python scripts/cleanup_legacy_indexes.py

    # 실제 삭제
    python scripts/cleanup_legacy_indexes.py --execute
"""

import argparse
import shutil
import sys
from pathlib import Path

# 프로젝트 루트 기준 경로
_SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = _SCRIPT_DIR.parent
BENCH_DATA_DIR = PROJECT_ROOT / "rag_bench" / "_benchdata"

# 삭제 대상 패턴: qdrant_db_combo1 ~ qdrant_db_combo4
LEGACY_COMBO_IDS = [1, 2, 3, 4]
LEGACY_PREFIX = "qdrant_db_combo"


def find_legacy_dirs() -> list[Path]:
    """레거시 combo 인덱스 디렉토리 목록을 반환한다."""
    targets = []
    for combo_id in LEGACY_COMBO_IDS:
        dir_path = BENCH_DATA_DIR / f"{LEGACY_PREFIX}{combo_id}"
        if dir_path.exists():
            targets.append(dir_path)
    return targets


def get_dir_size_mb(path: Path) -> float:
    """디렉토리 크기(MB)를 반환한다."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="레거시 Qdrant combo 인덱스 정리 (기본: dry-run)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="실제 삭제를 수행한다 (이 플래그 없이는 dry-run만 실행)",
    )
    args = parser.parse_args()

    print(f"벤치데이터 디렉토리: {BENCH_DATA_DIR}")
    print(f"레거시 대상: {[f'{LEGACY_PREFIX}{i}' for i in LEGACY_COMBO_IDS]}")
    print()

    if not BENCH_DATA_DIR.exists():
        print(f"오류: 벤치데이터 디렉토리가 존재하지 않습니다: {BENCH_DATA_DIR}")
        sys.exit(1)

    targets = find_legacy_dirs()

    if not targets:
        print("레거시 인덱스 디렉토리가 존재하지 않습니다. 정리 불필요.")
        sys.exit(0)

    total_size_mb = 0.0
    print(f"{'=' * 60}")
    print(f" 삭제 대상 ({len(targets)}개)")
    print(f"{'=' * 60}")
    for dir_path in targets:
        size_mb = get_dir_size_mb(dir_path)
        total_size_mb += size_mb
        print(f"  {dir_path.name:<30} ({size_mb:.1f} MB)")
    print(f"{'─' * 60}")
    print(f"  합계:                          {total_size_mb:.1f} MB")
    print(f"{'=' * 60}")
    print()

    if not args.execute:
        print("[DRY-RUN] 실제 삭제는 수행하지 않습니다.")
        print("  실제 삭제를 원하면: python scripts/cleanup_legacy_indexes.py --execute")
        return

    # 실제 삭제
    print("[EXECUTE] 레거시 인덱스 삭제 시작...")
    deleted = []
    failed = []
    for dir_path in targets:
        try:
            shutil.rmtree(dir_path)
            print(f"  삭제 완료: {dir_path.name}")
            deleted.append(dir_path)
        except Exception as e:
            print(f"  삭제 실패: {dir_path.name} — {e}")
            failed.append((dir_path, str(e)))

    print()
    print(f"{'=' * 60}")
    print(f" 결과: {len(deleted)}개 삭제 완료, {len(failed)}개 실패")
    print(f"  절약 공간: 약 {total_size_mb:.1f} MB")
    print(f"{'=' * 60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
