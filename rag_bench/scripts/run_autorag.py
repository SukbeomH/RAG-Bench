"""
AutoRAG 크로스 프레임워크 벤치마크 — rag_bench 동일 데이터로 AutoRAG 평가.

rag_bench의 QA 20개 + child_chunks를 AutoRAG parquet 포맷으로 변환하여
동일 데이터 기반 크로스 프레임워크 비교를 수행한다.

Usage:
    python -m rag_bench.scripts.run_autorag [--config dense|hybrid|PATH] [--skip_convert] [--compare]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from rag_bench.config import (
    BENCH_DATA_DIR,
    BENCH_DOCS_DIR,
    setup_ssl_bypass,
)
from rag_bench.indexing.chunker import create_parent_child_chunks

# ── 경로 상수 ──
PROJECT_ROOT = Path(__file__).parent.parent.parent
AUTORAG_DIR = PROJECT_ROOT / "autorag_benchmark"
AUTORAG_CONFIG_DIR = AUTORAG_DIR / "config"
AUTORAG_DATA_DIR = AUTORAG_DIR / "data_ragbench"  # 기존 data/ 보존
AUTORAG_RESULTS_DIR = AUTORAG_DIR / "results"

# ── 내장 설정 매핑 ──
CONFIG_PRESETS = {
    "dense": AUTORAG_CONFIG_DIR / "benchmark_config.yaml",
    "hybrid": AUTORAG_CONFIG_DIR / "hybrid_benchmark_config.yaml",
}


def _check_prerequisites(config_path: Path) -> None:
    """Step 1: AutoRAG import, API key, Qdrant, 설정 파일 확인."""
    errors = []

    # AutoRAG import
    try:
        import autorag  # noqa: F401
    except ImportError:
        print("Error: AutoRAG가 설치되지 않았습니다.")
        print("  설치: uv pip install -e '.[autorag]'")
        sys.exit(1)

    # OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    # 설정 파일
    if not config_path.exists():
        errors.append(f"설정 파일이 없습니다: {config_path}")

    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        sys.exit(1)

    # Qdrant Docker (경고만)
    try:
        import requests

        resp = requests.get("http://localhost:6333/healthz", timeout=3)
        if resp.status_code == 200:
            print("  [OK] Qdrant Docker 연결 확인 (localhost:6333)")
        else:
            print("  [WARN] Qdrant 응답 이상. Docker 컨테이너를 확인해주세요.")
    except Exception:
        print("  [WARN] Qdrant에 연결할 수 없습니다.")
        print("    docker compose up -d  또는")
        print("    docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant")


def _load_qa_dataset() -> dict:
    """QA 데이터셋 로드."""
    qa_path = BENCH_DATA_DIR / "qa_dataset.json"
    if not qa_path.exists():
        print(f"Error: QA 데이터셋이 없습니다: {qa_path}")
        print("  먼저 실행: python -m rag_bench.scripts.generate_qa")
        sys.exit(1)
    dataset = json.loads(qa_path.read_text(encoding="utf-8"))
    print(f"  QA 데이터셋 로드: {dataset['num_qa']}개 QA")
    return dataset


def _compute_conversion_hash(docs_hash: str, num_qa: int) -> str:
    """데이터 변환 해시 계산 (캐싱용)."""
    key = f"{docs_hash}:{num_qa}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _convert_data(dataset: dict) -> None:
    """Step 2: rag_bench 데이터 → AutoRAG parquet 변환."""
    import pandas as pd

    qa_pairs = dataset["qa_pairs"]
    docs_hash = dataset.get("docs_hash", "")
    num_qa = dataset["num_qa"]

    # 해시 기반 캐싱
    hash_file = AUTORAG_DATA_DIR / ".conversion_hash"
    new_hash = _compute_conversion_hash(docs_hash, num_qa)
    if hash_file.exists() and hash_file.read_text().strip() == new_hash:
        qa_parquet = AUTORAG_DATA_DIR / "qa.parquet"
        corpus_parquet = AUTORAG_DATA_DIR / "corpus.parquet"
        if qa_parquet.exists() and corpus_parquet.exists():
            print("  [캐시] 기존 변환 결과 재사용 (해시 일치)")
            return

    # 청킹
    print("  문서 청킹 중...")
    parent_store_path = BENCH_DATA_DIR / "parent_store"
    parent_pairs, child_chunks = create_parent_child_chunks(
        markdown_dir=str(BENCH_DOCS_DIR),
        parent_store_path=str(parent_store_path),
    )
    if not child_chunks:
        print("Error: Child 청크가 생성되지 않았습니다.")
        sys.exit(1)

    # ── corpus.parquet ──
    print(f"  corpus.parquet 생성 중 ({len(child_chunks)}개 청크)...")
    corpus_rows = []
    # child doc_id 매핑: parent_id → [doc_ids]
    parent_to_docids: dict[str, list[str]] = {}
    for i, doc in enumerate(child_chunks):
        doc_id = f"ragbench_{i:06d}"
        metadata = dict(doc.metadata)
        metadata["last_modified_datetime"] = datetime.now().isoformat()
        corpus_rows.append({
            "doc_id": doc_id,
            "contents": doc.page_content,
            "metadata": metadata,
        })
        pid = doc.metadata.get("parent_id", "")
        parent_to_docids.setdefault(pid, []).append(doc_id)

    corpus_df = pd.DataFrame(corpus_rows)

    # ── qa.parquet ──
    print(f"  qa.parquet 생성 중 ({len(qa_pairs)}개 QA)...")
    qa_rows = []
    for i, qa in enumerate(qa_pairs):
        pid = qa["parent_id"]
        # retrieval_gt: 해당 parent_id의 모든 child doc_ids
        gt_doc_ids = parent_to_docids.get(pid, [])
        qa_rows.append({
            "qid": f"q_{i:04d}",
            "query": qa["question"],
            "retrieval_gt": [gt_doc_ids],
            "generation_gt": [qa["ground_truth"]],
        })

    qa_df = pd.DataFrame(qa_rows)

    # 저장
    AUTORAG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = AUTORAG_DATA_DIR / "corpus.parquet"
    qa_path = AUTORAG_DATA_DIR / "qa.parquet"
    corpus_df.to_parquet(corpus_path, index=False)
    qa_df.to_parquet(qa_path, index=False)

    # 해시 저장
    hash_file.write_text(new_hash)

    print(f"  저장 완료:")
    print(f"    corpus: {corpus_path} ({len(corpus_df)}행)")
    print(f"    qa:     {qa_path} ({len(qa_df)}행)")


def _run_benchmark(config_path: Path) -> Path:
    """Step 3: AutoRAG 벤치마크 실행."""
    from autorag.evaluator import Evaluator

    qa_path = str(AUTORAG_DATA_DIR / "qa.parquet")
    corpus_path = str(AUTORAG_DATA_DIR / "corpus.parquet")

    evaluator = Evaluator(
        qa_data_path=qa_path,
        corpus_data_path=corpus_path,
        project_dir=str(AUTORAG_RESULTS_DIR),
    )

    print(f"  설정: {config_path.name}")
    print(f"  데이터: {AUTORAG_DATA_DIR}")
    print(f"  결과: {AUTORAG_RESULTS_DIR}")

    evaluator.start_trial(str(config_path))

    # 가장 최근 trial 디렉토리
    trial_dirs = sorted(AUTORAG_RESULTS_DIR.glob("[0-9]*"))
    if trial_dirs:
        return trial_dirs[-1]
    return AUTORAG_RESULTS_DIR


def _analyze_results(trial_dir: Path) -> None:
    """Step 4: 결과 분석."""
    import pandas as pd

    # summary.csv
    summary_path = trial_dir / "summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        print(f"\n  ── 노드별 최적 모듈 (summary.csv) ──")
        print(f"  {df.to_string(index=False)}")
    else:
        print(f"  [WARN] summary.csv를 찾을 수 없습니다: {summary_path}")

    # semantic_retrieval 상세
    retrieval_summary = trial_dir / "semantic_retrieval" / "summary.csv"
    if not retrieval_summary.exists():
        # 대체 경로 탐색
        candidates = list(trial_dir.glob("**/semantic_retrieval/summary.csv"))
        if candidates:
            retrieval_summary = candidates[0]

    if retrieval_summary.exists():
        rdf = pd.read_csv(retrieval_summary)
        print(f"\n  ── Dense 모델별 비교 (semantic_retrieval) ──")
        print(f"  {rdf.to_string(index=False)}")

    # 대시보드 안내
    print(f"\n  [TIP] 대시보드: autorag dashboard --trial_dir {trial_dir}")


def _compare_with_ragbench() -> None:
    """Step 5: rag_bench 결과와 비교."""
    import pandas as pd

    ragas_path = BENCH_DATA_DIR / "all_combos_ragas.csv"
    if not ragas_path.exists():
        print("  [WARN] rag_bench RAGAS 결과가 없습니다.")
        print(f"    먼저 실행: python -m rag_bench.scripts.run_all_combos")
        return

    ragas_df = pd.read_csv(ragas_path)
    print(f"\n  ── rag_bench RAGAS 결과 ──")
    print(f"  {ragas_df.to_string(index=False)}")

    # AutoRAG 최신 trial에서 retrieval 결과 로드
    trial_dirs = sorted(AUTORAG_RESULTS_DIR.glob("[0-9]*"))
    if not trial_dirs:
        print("  [WARN] AutoRAG 결과가 없습니다.")
        return

    latest = trial_dirs[-1]
    summary_path = latest / "summary.csv"
    if not summary_path.exists():
        print(f"  [WARN] AutoRAG summary 없음: {summary_path}")
        return

    autorag_df = pd.read_csv(summary_path)

    # 임베딩 모델 매핑 테이블
    print(f"\n  ── 크로스 프레임워크 비교 ──")
    print(f"  {'프레임워크':<20} {'데이터':<15} {'QA 수':<8} {'비고'}")
    print(f"  {'─' * 20} {'─' * 15} {'─' * 8} {'─' * 30}")
    print(f"  {'rag_bench':<20} {'child_chunks':<15} {'20':<8} {'RAGAS 평가 (6종 + ColBERT + Rerank)'}")
    print(f"  {'AutoRAG':<20} {'data_ragbench':<15} {'20':<8} {'AutoRAG 파이프라인 탐색'}")

    print(f"\n  ── AutoRAG 노드별 최적 결과 ──")
    print(f"  {autorag_df.to_string(index=False)}")


def main():
    parser = argparse.ArgumentParser(
        description="AutoRAG 크로스 프레임워크 벤치마크 — rag_bench 동일 데이터 기반"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dense",
        help="YAML 설정: dense, hybrid, 또는 커스텀 경로 (기본: dense)",
    )
    parser.add_argument(
        "--skip_convert",
        action="store_true",
        help="데이터 변환 건너뛰기 (기존 parquet 사용)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="rag_bench 결과와 비교",
    )
    parser.add_argument(
        "--convert_only",
        action="store_true",
        help="데이터 변환만 수행 (벤치마크 실행 건너뛰기). 별도 venv에서 실행 시 사용.",
    )
    args = parser.parse_args()

    setup_ssl_bypass()

    # 설정 파일 결정
    if args.config in CONFIG_PRESETS:
        config_path = CONFIG_PRESETS[args.config]
    else:
        config_path = Path(args.config)

    print(f"{'═' * 60}")
    print(f" AutoRAG 크로스 프레임워크 벤치마크")
    print(f"{'═' * 60}")
    print(f"  설정: {args.config} → {config_path}")
    print(f"  데이터 변환: {'건너뛰기' if args.skip_convert else '실행'}")
    print(f"  비교 모드: {'ON' if args.compare else 'OFF'}")

    # ── Step 1: Prerequisites ──
    if not args.convert_only:
        print(f"\n{'=' * 60}")
        print("Step 1: Prerequisites 확인")
        print(f"{'=' * 60}")
        _check_prerequisites(config_path)

    # ── Step 2: 데이터 변환 ──
    print(f"\n{'=' * 60}")
    print("Step 2: 데이터 변환 (rag_bench → AutoRAG parquet)")
    print(f"{'=' * 60}")
    dataset = _load_qa_dataset()
    if args.skip_convert:
        qa_parquet = AUTORAG_DATA_DIR / "qa.parquet"
        corpus_parquet = AUTORAG_DATA_DIR / "corpus.parquet"
        if not qa_parquet.exists() or not corpus_parquet.exists():
            print("  [WARN] parquet 파일이 없어 변환을 실행합니다.")
            _convert_data(dataset)
        else:
            print("  [건너뛰기] 기존 parquet 사용")
    else:
        t0 = time.time()
        _convert_data(dataset)
        print(f"  변환 소요: {time.time() - t0:.1f}s")

    if args.convert_only:
        print(f"\n{'═' * 60}")
        print(f" 데이터 변환 완료 (--convert_only)")
        print(f"{'═' * 60}")
        print(f"  QA:     {AUTORAG_DATA_DIR / 'qa.parquet'}")
        print(f"  Corpus: {AUTORAG_DATA_DIR / 'corpus.parquet'}")
        print(f"\n  별도 venv에서 벤치마크 실행:")
        print(f"  /tmp/autorag-env/.venv/bin/python scripts/run_autorag_isolated.py \\")
        print(f"    --config {config_path} \\")
        print(f"    --qa {AUTORAG_DATA_DIR / 'qa.parquet'} \\")
        print(f"    --corpus {AUTORAG_DATA_DIR / 'corpus.parquet'} \\")
        print(f"    --project_dir {AUTORAG_RESULTS_DIR}")
        return

    # ── Step 3: AutoRAG 벤치마크 ──
    print(f"\n{'=' * 60}")
    print("Step 3: AutoRAG 벤치마크 실행")
    print(f"{'=' * 60}")
    t0 = time.time()
    trial_dir = _run_benchmark(config_path)
    elapsed = time.time() - t0
    print(f"  벤치마크 완료: {elapsed:.1f}s")
    print(f"  결과 디렉토리: {trial_dir}")

    # ── Step 4: 결과 분석 ──
    print(f"\n{'=' * 60}")
    print("Step 4: 결과 분석")
    print(f"{'=' * 60}")
    _analyze_results(trial_dir)

    # ── Step 5: rag_bench 비교 ──
    if args.compare:
        print(f"\n{'=' * 60}")
        print("Step 5: rag_bench 결과 비교")
        print(f"{'=' * 60}")
        _compare_with_ragbench()

    print(f"\n{'═' * 60}")
    print(f" 벤치마크 완료")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
