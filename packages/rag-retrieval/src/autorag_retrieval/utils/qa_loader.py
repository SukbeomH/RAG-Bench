"""QA 데이터셋 로드 유틸리티."""
import json
import sys
from pathlib import Path


def load_qa_dataset(data_dir: Path) -> dict:
    """qa_dataset.json을 로드한다.

    Args:
        data_dir: _benchdata 디렉터리 경로 (BENCH_DATA_DIR).

    Returns:
        { 'num_qa': int, 'qa_pairs': List[dict] } 형태의 dict.
    """
    qa_path = data_dir / "qa_dataset.json"
    if not qa_path.exists():
        print(f"Error: QA 데이터셋이 없습니다: {qa_path}")
        print("  먼저 실행: python -m rag_bench.scripts.generate_qa")
        sys.exit(1)
    dataset = json.loads(qa_path.read_text(encoding="utf-8"))
    print(f"QA 데이터셋 로드: {dataset['num_qa']}개 QA")
    return dataset
