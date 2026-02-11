"""
전체 조합 벤치마크 — DenseSparse 6종 + ColBERT + ColBERTRerank 완전 비교.

DenseSparse 조합:
  1. 한국어 최적 (KoSimCSE + BM25/OKt)      — konlpy 필요
  2. 다국어 균형 (E5 + SPLADE)               — SPLADE 모델 필요
  3. 올인원 통합 (BGE-M3)                    — fastembed
  4. 경량/빠른 속도 (MiniLM + BM25)          — fastembed
  5. 고성능 API (OpenAI Large + SPLADE)      — 유료 API
  6. 한국어 API (Upstage Solar + BM25/OKt)   — 유료 API + konlpy

추가 전략:
  7. ColBERT (jina-colbert-v2, brute-force)
  8~13. ColBERTRerank (각 DenseSparse 위에 리랭킹)

초기화/인덱싱 실패 전략은 건너뛰고, 성공한 전략만 평가한다.

Usage:
    python -m rag_bench.scripts.run_all_combos [--k 3] [--combos 1,3,4] [--skip_paid] [--skip_colbert] [--skip_rerank] [--no_ragas]
"""

import argparse
import json
import sys
import time
import traceback
from typing import List, Optional, Tuple

from rag_bench.config import (
    BENCH_DATA_DIR,
    BENCH_DOCS_DIR,
    setup_ssl_bypass,
)
from rag_bench.indexing.chunker import create_parent_child_chunks
from rag_bench.runner import BenchmarkRunner

ALL_COMBO_IDS = [1, 2, 3, 4, 5, 6]
PAID_COMBO_IDS = {5, 6}


def _load_qa_dataset() -> dict:
    qa_path = BENCH_DATA_DIR / "qa_dataset.json"
    if not qa_path.exists():
        print(f"Error: QA 데이터셋이 없습니다: {qa_path}")
        print("  먼저 실행: python -m rag_bench.scripts.generate_qa")
        sys.exit(1)
    dataset = json.loads(qa_path.read_text(encoding="utf-8"))
    print(f"QA 데이터셋 로드: {dataset['num_qa']}개 QA")
    return dataset


def _try_build_dense_sparse(combo_id: int, child_chunks, qdrant_suffix: str):
    """DenseSparseStrategy를 생성·인덱싱하고, 실패하면 (None, error_msg)를 반환한다."""
    from rag_bench.strategies.dense_sparse import DenseSparseStrategy

    qdrant_path = str(BENCH_DATA_DIR / f"qdrant_db_{qdrant_suffix}")
    strategy = DenseSparseStrategy(combo_id=combo_id, qdrant_path=qdrant_path)
    strategy.index(child_chunks)
    return strategy, None


def _try_build_colbert(child_chunks):
    from rag_bench.strategies.colbert import ColBERTStrategy

    strategy = ColBERTStrategy(
        model_name="jinaai/jina-colbert-v2",
        use_index=False,
    )
    strategy.index(child_chunks)
    return strategy, None


def _try_build_rerank(base_strategy, child_chunks):
    from rag_bench.strategies.colbert_rerank import ColBERTRerankStrategy

    strategy = ColBERTRerankStrategy(
        base_strategy=base_strategy,
        model_name="jinaai/jina-colbert-v2",
        rerank_n=20,
    )
    # base는 이미 인덱싱 됨 → ColBERT 모델만 로드
    strategy._base_strategy = base_strategy  # 이미 인덱싱된 base 재사용
    strategy._ensure_initialized()
    strategy._is_ready = True
    return strategy, None


def _safe_build(label: str, build_fn, *args) -> Tuple[Optional[object], Optional[str]]:
    """전략 생성을 시도하고, 실패 시 에러 메시지를 반환한다."""
    print(f"\n{'─' * 60}")
    print(f"▶ 생성 중: {label}")
    print(f"{'─' * 60}")
    t0 = time.time()
    try:
        strategy, _ = build_fn(*args)
        elapsed = time.time() - t0
        print(f"  ✓ 성공 ({elapsed:.1f}s)")
        return strategy, None
    except Exception as e:
        elapsed = time.time() - t0
        err = f"{type(e).__name__}: {e}"
        print(f"  ✗ 실패 ({elapsed:.1f}s): {err}")
        traceback.print_exc()
        return None, err


def _print_ragas_table(scores_df):
    if scores_df is None or scores_df.empty:
        print("RAGAS 평가 결과가 없습니다.")
        return

    print(f"\n{'═' * 90}")
    print(" RAGAS 평가 결과 비교")
    print(f"{'═' * 90}")

    metric_cols = [c for c in scores_df.columns if c != "strategy"]
    header = f"  {'전략':<45}"
    for col in metric_cols:
        header += f" {col:>14}"
    print(header)
    print(f"  {'─' * 45} " + " ".join("─" * 14 for _ in metric_cols))

    for _, row in scores_df.iterrows():
        line = f"  {row['strategy']:<45}"
        for col in metric_cols:
            val = row.get(col, "N/A")
            if isinstance(val, float):
                line += f" {val:>14.4f}"
            else:
                line += f" {str(val):>14}"
        print(line)


def _print_init_summary(results: list):
    """전략 초기화 결과 요약표를 출력한다."""
    print(f"\n{'═' * 60}")
    print(" 전략 초기화 결과")
    print(f"{'═' * 60}")
    ok = sum(1 for _, s, _ in results if s is not None)
    fail = sum(1 for _, s, _ in results if s is None)
    print(f"  성공: {ok}개 / 실패: {fail}개 / 전체: {len(results)}개\n")

    for label, strategy, err in results:
        if strategy is not None:
            print(f"  ✓ {label}")
        else:
            print(f"  ✗ {label} — {err}")


def main():
    parser = argparse.ArgumentParser(
        description="전체 조합 벤치마크 — DenseSparse 6종 + ColBERT + ColBERTRerank"
    )
    parser.add_argument("--k", type=int, default=3, help="검색 결과 수 (기본: 3)")
    parser.add_argument(
        "--combos",
        type=str,
        default=None,
        help="DenseSparse 조합 ID (쉼표 구분, 예: 1,3,4). 미지정 시 전체.",
    )
    parser.add_argument(
        "--skip_paid",
        action="store_true",
        help="유료 API 조합(5, 6) 건너뛰기",
    )
    parser.add_argument(
        "--skip_colbert",
        action="store_true",
        help="ColBERT 단독 전략 건너뛰기",
    )
    parser.add_argument(
        "--skip_rerank",
        action="store_true",
        help="ColBERTRerank 전략 건너뛰기",
    )
    parser.add_argument(
        "--no_ragas",
        action="store_true",
        help="RAGAS 평가 건너뛰기 (검색 성능만 측정)",
    )
    args = parser.parse_args()

    setup_ssl_bypass()

    # ── 조합 ID 결정 ──
    if args.combos:
        combo_ids = [int(x.strip()) for x in args.combos.split(",")]
    else:
        combo_ids = list(ALL_COMBO_IDS)

    if args.skip_paid:
        combo_ids = [c for c in combo_ids if c not in PAID_COMBO_IDS]

    print(f"대상 DenseSparse 조합: {combo_ids}")
    print(f"ColBERT: {'OFF' if args.skip_colbert else 'ON'}")
    print(f"ColBERTRerank: {'OFF' if args.skip_rerank else 'ON'}")
    print(f"RAGAS 평가: {'OFF' if args.no_ragas else 'ON'}")

    # ── Step 1: QA 로드 ──
    print(f"\n{'=' * 60}")
    print("Step 1: QA 데이터셋 로드")
    print(f"{'=' * 60}")
    dataset = _load_qa_dataset()
    qa_pairs = dataset["qa_pairs"]
    queries = [qa["question"] for qa in qa_pairs]
    ground_truths = [qa["ground_truth"] for qa in qa_pairs]

    # ── Step 2: 문서 청킹 ──
    print(f"\n{'=' * 60}")
    print("Step 2: 문서 청킹")
    print(f"{'=' * 60}")
    parent_store_path = BENCH_DATA_DIR / "parent_store"
    parent_pairs, child_chunks = create_parent_child_chunks(
        markdown_dir=str(BENCH_DOCS_DIR),
        parent_store_path=str(parent_store_path),
    )
    if not child_chunks:
        print("Error: Child 청크가 생성되지 않았습니다.")
        sys.exit(1)

    # ── Step 3: 전략 생성 ──
    print(f"\n{'=' * 60}")
    print("Step 3: 전략 생성 및 인덱싱")
    print(f"{'=' * 60}")

    # (label, strategy_or_None, error_or_None)
    build_results: List[Tuple[str, object, Optional[str]]] = []

    # 3-a. DenseSparse 6종
    ds_strategies = {}  # combo_id → strategy (성공한 것만)
    for cid in combo_ids:
        label = f"DenseSparse combo={cid}"
        strategy, err = _safe_build(
            label, _try_build_dense_sparse, cid, child_chunks, f"combo{cid}"
        )
        build_results.append((label, strategy, err))
        if strategy is not None:
            ds_strategies[cid] = strategy

    # 3-b. ColBERT 단독
    colbert_strategy = None
    if not args.skip_colbert:
        label = "ColBERT (jina-colbert-v2, brute-force)"
        strategy, err = _safe_build(label, _try_build_colbert, child_chunks)
        build_results.append((label, strategy, err))
        colbert_strategy = strategy

    # 3-c. ColBERTRerank (각 성공한 DenseSparse 위에)
    rerank_strategies = {}
    if not args.skip_rerank:
        for cid, ds in ds_strategies.items():
            label = f"ColBERTRerank (base=combo{cid})"
            strategy, err = _safe_build(label, _try_build_rerank, ds, child_chunks)
            build_results.append((label, strategy, err))
            if strategy is not None:
                rerank_strategies[cid] = strategy

    _print_init_summary(build_results)

    # ── 성공한 전략만 수집 ──
    active_strategies = []
    for cid in combo_ids:
        if cid in ds_strategies:
            active_strategies.append(ds_strategies[cid])
    if colbert_strategy is not None:
        active_strategies.append(colbert_strategy)
    for cid in combo_ids:
        if cid in rerank_strategies:
            active_strategies.append(rerank_strategies[cid])

    if not active_strategies:
        print("\n성공한 전략이 없습니다. 종료합니다.")
        sys.exit(1)

    print(f"\n벤치마크 대상 전략: {len(active_strategies)}개")

    # ── Step 4: 벤치마크 실행 ──
    print(f"\n{'=' * 60}")
    print("Step 4: 벤치마크 실행")
    print(f"{'=' * 60}")

    evaluator = None
    if not args.no_ragas:
        from rag_bench.evaluation import RAGEvaluator
        try:
            evaluator = RAGEvaluator()
        except Exception as e:
            print(f"RAGEvaluator 초기화 실패 (RAGAS 평가 건너뜀): {e}")

    runner = BenchmarkRunner(
        strategies=active_strategies,
        queries=queries,
        k=args.k,
        evaluator=evaluator,
    )
    runner.run()
    runner.compare()

    # ── Step 5: RAGAS 평가 ──
    scores_df = None
    if evaluator is not None:
        print(f"\n{'=' * 60}")
        print("Step 5: RAGAS 평가")
        print(f"{'=' * 60}")
        scores_df = runner.evaluate(ground_truths=ground_truths)
        _print_ragas_table(scores_df)

    # ── Step 6: 결과 저장 ──
    print(f"\n{'=' * 60}")
    print("Step 6: 결과 저장")
    print(f"{'=' * 60}")

    results_df = runner.to_dataframe()
    if results_df is not None:
        results_path = BENCH_DATA_DIR / "all_combos_results.csv"
        results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
        print(f"  검색 결과: {results_path}")

    if scores_df is not None:
        scores_path = BENCH_DATA_DIR / "all_combos_ragas.csv"
        scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")
        print(f"  RAGAS 점수: {scores_path}")

    # ── Step 7: 클린업 ──
    print(f"\n{'=' * 60}")
    print("Step 7: 클린업")
    print(f"{'=' * 60}")
    for strategy in active_strategies:
        try:
            strategy.cleanup()
            print(f"  ✓ {strategy.name}")
        except Exception as e:
            print(f"  ✗ {strategy.name}: {e}")

    print(f"\n{'═' * 60}")
    print(f" 벤치마크 완료 — 전략 {len(active_strategies)}개 비교")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
