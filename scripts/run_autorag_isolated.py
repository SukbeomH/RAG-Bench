"""
AutoRAG 벤치마크 실행 — 별도 venv 독립 스크립트.

rag_bench과 의존성 충돌을 피하기 위해 /tmp/autorag-env venv에서 실행.
이 스크립트는 rag_bench에 의존하지 않으며, parquet 파일과 YAML 설정만 사용.

Usage:
    /tmp/autorag-env/.venv/bin/python scripts/run_autorag_isolated.py \
        --config autorag_benchmark/config/benchmark_config.yaml \
        --qa autorag_benchmark/data_ragbench/qa.parquet \
        --corpus autorag_benchmark/data_ragbench/corpus.parquet \
        --project_dir autorag_benchmark/results
"""

import argparse
import os
import sys
import time
from pathlib import Path

# .env 파일 로드
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def check_prerequisites(config_path: Path) -> None:
    """AutoRAG import, API key, Qdrant, 설정 파일 확인."""
    try:
        import autorag  # noqa: F401
        print("[OK] AutoRAG 임포트 성공")
    except ImportError:
        print("[ERROR] AutoRAG가 설치되지 않았습니다.")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)
    print("[OK] OPENAI_API_KEY 확인")

    if not config_path.exists():
        print(f"[ERROR] 설정 파일이 없습니다: {config_path}")
        sys.exit(1)
    print(f"[OK] 설정 파일: {config_path}")

    try:
        import requests
        resp = requests.get("http://localhost:6333/healthz", timeout=3)
        if resp.status_code == 200:
            print("[OK] Qdrant Docker 연결 확인 (localhost:6333)")
        else:
            print("[WARN] Qdrant 응답 이상")
    except Exception:
        print("[WARN] Qdrant에 연결할 수 없습니다.")
        print("  docker compose up -d")


def run_benchmark(config_path: Path, qa_path: Path, corpus_path: Path, project_dir: Path) -> Path:
    """AutoRAG 벤치마크 실행."""
    from autorag.evaluator import Evaluator

    evaluator = Evaluator(
        qa_data_path=str(qa_path),
        corpus_data_path=str(corpus_path),
        project_dir=str(project_dir),
    )

    print(f"\n  설정: {config_path.name}")
    print(f"  QA: {qa_path}")
    print(f"  Corpus: {corpus_path}")
    print(f"  결과: {project_dir}")

    evaluator.start_trial(str(config_path))

    trial_dirs = sorted(project_dir.glob("[0-9]*"))
    if trial_dirs:
        return trial_dirs[-1]
    return project_dir


def analyze_results(trial_dir: Path) -> None:
    """결과 분석."""
    import pandas as pd

    summary_path = trial_dir / "summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        print(f"\n{'=' * 70}")
        print(" 노드별 최적 모듈 (summary.csv)")
        print(f"{'=' * 70}")
        print(df.to_string(index=False))
    else:
        print(f"[WARN] summary.csv 없음: {summary_path}")

    # retrieval 상세
    for pattern in ["**/semantic_retrieval/summary.csv", "**/retrieve_node_line/*/summary.csv"]:
        for p in trial_dir.glob(pattern):
            rdf = pd.read_csv(p)
            print(f"\n  -- {p.relative_to(trial_dir)} --")
            print(f"  {rdf.to_string(index=False)}")

    print(f"\n[TIP] 대시보드: autorag dashboard --trial_dir {trial_dir}")


def main():
    parser = argparse.ArgumentParser(description="AutoRAG 벤치마크 (독립 venv)")
    parser.add_argument("--config", type=str, required=True, help="YAML 설정 파일 경로")
    parser.add_argument("--qa", type=str, required=True, help="qa.parquet 경로")
    parser.add_argument("--corpus", type=str, required=True, help="corpus.parquet 경로")
    parser.add_argument("--project_dir", type=str, required=True, help="결과 저장 디렉토리")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    qa_path = Path(args.qa).resolve()
    corpus_path = Path(args.corpus).resolve()
    project_dir = Path(args.project_dir).resolve()

    print(f"{'=' * 60}")
    print(f" AutoRAG 벤치마크 (독립 venv)")
    print(f"{'=' * 60}")

    check_prerequisites(config_path)

    if not qa_path.exists() or not corpus_path.exists():
        print(f"[ERROR] parquet 파일이 없습니다.")
        print(f"  qa: {qa_path} (exists={qa_path.exists()})")
        print(f"  corpus: {corpus_path} (exists={corpus_path.exists()})")
        print(f"\n  먼저 현재 환경에서 데이터 변환을 실행하세요:")
        print(f"  uv run python -m rag_bench.scripts.run_autorag --skip_benchmark")
        sys.exit(1)

    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f" 벤치마크 실행")
    print(f"{'=' * 60}")
    t0 = time.time()
    trial_dir = run_benchmark(config_path, qa_path, corpus_path, project_dir)
    elapsed = time.time() - t0
    print(f"\n  벤치마크 완료: {elapsed:.1f}s")
    print(f"  결과: {trial_dir}")

    print(f"\n{'=' * 60}")
    print(f" 결과 분석")
    print(f"{'=' * 60}")
    analyze_results(trial_dir)

    print(f"\n{'=' * 60}")
    print(f" 완료")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
