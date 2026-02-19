"""
통합 벤치마크 실행 + RAGAS 평가 스크립트.

QA 로드 → 청킹 → 3종 전략 인덱싱 → BenchmarkRunner.run() → ExtendedRAGEvaluator → 비교 테이블

전략 3종:
  1. DenseSparse (combo_id=4, MiniLM + BM25)  → _benchdata/qdrant_db_bench
  2. ColBERT (all-MiniLM-L6-v2, brute-force)  → in-memory
  3. ColBERTRerank (DenseSparse base)          → _benchdata/qdrant_db_rerank

Usage:
    python -m rag_bench.scripts.run_bench [--k 3] [--skip_colbert]
"""

import argparse
import json
import sys

from rag_bench.config import (
    BENCH_DATA_DIR,
    BENCH_DOCS_DIR,
    setup_ssl_bypass,
)
from rag_bench.evaluation import ExtendedRAGEvaluator
from rag_bench.indexing.chunker import create_parent_child_chunks
from rag_bench.runner import BenchmarkRunner


def _load_qa_dataset() -> dict:
    """_benchdata/qa_dataset.json을 로드한다."""
    qa_path = BENCH_DATA_DIR / "qa_dataset.json"
    if not qa_path.exists():
        print(f"Error: QA 데이터셋이 없습니다: {qa_path}")
        print("  먼저 실행: python -m rag_bench.scripts.generate_qa")
        sys.exit(1)

    dataset = json.loads(qa_path.read_text(encoding="utf-8"))
    print(f"QA 데이터셋 로드: {dataset['num_qa']}개 QA")
    return dataset


def _build_strategies(child_chunks, skip_colbert: bool) -> list:
    """벤치마크 대상 전략 3종을 생성한다."""
    from rag_bench.strategies.dense_sparse import DenseSparseStrategy

    strategies = []

    # 1. DenseSparse (combo_id=4, 경량/빠른)
    qdrant_bench = str(BENCH_DATA_DIR / "qdrant_db_bench")
    ds = DenseSparseStrategy(combo_id=4, qdrant_path=qdrant_bench)
    strategies.append(ds)

    # 2. ColBERT (brute-force, in-memory)
    if not skip_colbert:
        from rag_bench.strategies.colbert import ColBERTStrategy

        colbert = ColBERTStrategy(
            model_name="jinaai/jina-colbert-v2",
            use_index=False,
        )
        strategies.append(colbert)

    # 3. ColBERTRerank (DenseSparse base)
    from rag_bench.strategies.colbert_rerank import ColBERTRerankStrategy

    qdrant_rerank = str(BENCH_DATA_DIR / "qdrant_db_rerank")
    ds_rerank_base = DenseSparseStrategy(combo_id=4, qdrant_path=qdrant_rerank)
    reranker = ColBERTRerankStrategy(
        base_strategy=ds_rerank_base,
        model_name="jinaai/jina-colbert-v2",
        rerank_n=20,
    )
    strategies.append(reranker)

    return strategies


def _print_ragas_table(scores_df):
    """RAGAS 평가 결과를 포맷팅하여 출력한다."""
    if scores_df is None or scores_df.empty:
        print("RAGAS 평가 결과가 없습니다.")
        return

    print(f"\n{'=' * 80}")
    print("RAGAS 평가 결과 비교")
    print(f"{'=' * 80}")

    # 컬럼 정렬
    metric_cols = [c for c in scores_df.columns if c != "strategy"]
    header = f"{'전략':<45}"
    for col in metric_cols:
        header += f" {col:>15}"
    print(header)
    print("-" * 80)

    for _, row in scores_df.iterrows():
        line = f"{row['strategy']:<45}"
        for col in metric_cols:
            val = row.get(col, "N/A")
            if isinstance(val, float):
                line += f" {val:>15.4f}"
            else:
                line += f" {str(val):>15}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="통합 벤치마크 + RAGAS 평가")
    parser.add_argument("--k", type=int, default=3, help="검색 결과 수 (기본: 3)")
    parser.add_argument(
        "--skip_colbert",
        action="store_true",
        help="ColBERT 전략 건너뛰기 (DenseSparse + Rerank만 실행)",
    )
    args = parser.parse_args()

    setup_ssl_bypass()

    # 1. QA 데이터셋 로드
    print("\n=== Step 1: QA 데이터셋 로드 ===")
    dataset = _load_qa_dataset()
    qa_pairs = dataset["qa_pairs"]
    queries = [qa["question"] for qa in qa_pairs]
    ground_truths = [qa["ground_truth"] for qa in qa_pairs]

    # 2. 문서 청킹
    print("\n=== Step 2: 문서 청킹 ===")
    parent_store_path = BENCH_DATA_DIR / "parent_store"
    parent_pairs, child_chunks = create_parent_child_chunks(
        markdown_dir=str(BENCH_DOCS_DIR),
        parent_store_path=str(parent_store_path),
    )

    if not child_chunks:
        print("Error: Child 청크가 생성되지 않았습니다.")
        sys.exit(1)

    # 3. 전략 생성 및 인덱싱
    print("\n=== Step 3: 전략 생성 및 인덱싱 ===")
    strategies = _build_strategies(child_chunks, skip_colbert=args.skip_colbert)

    for strategy in strategies:
        print(f"\n--- 인덱싱: {strategy.name} ---")
        strategy.index(child_chunks)

    # 4. 벤치마크 실행
    print("\n=== Step 4: 벤치마크 실행 ===")
    evaluator = ExtendedRAGEvaluator()
    runner = BenchmarkRunner(
        strategies=strategies,
        queries=queries,
        k=args.k,
        evaluator=evaluator,
    )
    runner.run()
    runner.compare()

    # 5. RAGAS 평가
    print("\n=== Step 5: RAGAS 평가 ===")
    scores_df = runner.evaluate(ground_truths=ground_truths)
    _print_ragas_table(scores_df)

    # 6. 결과 DataFrame 저장
    results_df = runner.to_dataframe()
    if results_df is not None:
        results_path = BENCH_DATA_DIR / "bench_results.csv"
        results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장: {results_path}")

    if scores_df is not None:
        scores_path = BENCH_DATA_DIR / "ragas_scores.csv"
        scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")
        print(f"RAGAS 점수 저장: {scores_path}")

    # 7. 클린업
    print("\n=== 클린업 ===")
    for strategy in strategies:
        try:
            strategy.cleanup()
        except Exception as e:
            print(f"  cleanup 실패 ({strategy.name}): {e}")

    print("\n=== 벤치마크 완료 ===")


if __name__ == "__main__":
    main()
