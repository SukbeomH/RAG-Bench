"""
전체 조합 벤치마크 — 4-Layer 교차 조합 + 2-Pass 실행.

4-Layer 설계:
  Layer 1: Dense Model   (kosimcse, e5, bge-m3 / openai-large, upstage)
  Layer 2: Sparse Model  (korean_bm25, splade)
  Layer 3: Reranker      (none, colbert, flashrank)
  Layer 4: Contextual    (none, contextual) — 인덱싱 시 LLM 문맥 부착, 독립 적용 가능

총 유효 조합 (full): 5 × 2 × 3 × 2 = 60개

옵션:
  --preset quick|standard|full  프리셋 기반 조합 생성 (필수)
  --pass1-only                  레이턴시만 측정 (RAGAS 없음)
  --top_n N                     Pass 1 후 상위 N만 RAGAS
  --dry-run                     조합 목록만 출력
  --layers                      레이어별 기여도 분석

Usage:
    python -m rag_bench.scripts.run_all_combos --preset full --dry-run
    python -m rag_bench.scripts.run_all_combos --preset quick --pass1-only
    python -m rag_bench.scripts.run_all_combos --preset standard --top_n 10
"""

import argparse
import gc
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from rag_bench.config import (
    BENCH_DATA_DIR,
    BENCH_DOCS_DIR,
    DOCS_DIR,
    setup_ssl_bypass,
)
from rag_bench.combo import (
    ComboSpec, PRESETS, generate_valid_combinations,
    CacheConfig, IndexCacheManager, build_strategy_from_spec,
)
from rag_bench.indexing.chunker import create_parent_child_chunks
from rag_bench.indexing.pdf_converter import pdfs_to_markdowns
from rag_bench.run_tracker import RunTracker, track_openai_tokens
from rag_bench.runner import BenchmarkRunner
from rag_bench.utils.qa_loader import load_qa_dataset
from rag_bench.utils.report import print_ragas_table


def _safe_build(
    label: str, build_fn, *args, progress: str = "",
    tracker: Optional["RunTracker"] = None,
    spec: Optional["ComboSpec"] = None,
) -> Tuple[Optional[object], Optional[str]]:
    print(f"\n{'─' * 60}")
    prefix = f"{progress} " if progress else ""
    print(f"{prefix}▶ 생성 중: {label}")
    print(f"{'─' * 60}")

    timing = None
    if tracker and spec:
        timing = tracker.start_build(
            label=label,
            dense=spec.dense,
            sparse=spec.sparse,
            reranker=spec.reranker,
            llm_support=spec.llm_support,
            retrieval_mode=spec.retrieval_mode,
        )
    elif tracker:
        timing = tracker.start_build(label=label)

    t0 = time.time()
    try:
        with track_openai_tokens() as token_usage:
            strategy, _ = build_fn(*args)
        elapsed = time.time() - t0
        token_info = ""
        if token_usage.total_tokens > 0:
            token_info = f", tokens: {token_usage.total_tokens:,}"
        print(f"  ✓ 성공 ({elapsed:.1f}s{token_info})")
        if tracker and timing:
            tracker.end_build(timing, success=True, tokens=token_usage)
        _release_memory()
        return strategy, None
    except Exception as e:
        elapsed = time.time() - t0
        err = f"{type(e).__name__}: {e}"
        print(f"  ✗ 실패 ({elapsed:.1f}s): {err}")
        traceback.print_exc()
        if tracker and timing:
            tracker.end_build(timing, success=False, error=err)
        _release_memory()
        return None, err


def _print_init_summary(results: list):
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


# ===========================================================================
# 새 모드: 4-Layer 조합 실행 — 파이프라인 Step 헬퍼 함수
# ===========================================================================


def _step0_sample_pdfs(args) -> None:
    """Step 0: PDF 페이지 샘플링 → Markdown 재생성."""
    print(f"\n{'=' * 60}")
    print("Step 0: PDF 페이지 샘플링 → Markdown 변환")
    print(f"{'=' * 60}")
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[Warning] {DOCS_DIR}에 PDF 없음 — 기존 .md 파일을 사용합니다.")
        return
    ratio = getattr(args, "page_sample_ratio", 0.1)
    max_p = getattr(args, "max_sample_pages", 5)
    print(f"  PDF: {len(pdf_files)}개  비율: {ratio:.0%}  최대: {max_p}페이지")
    BENCH_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    pdfs_to_markdowns(
        docs_dir=str(DOCS_DIR),
        output_dir=str(BENCH_DOCS_DIR),
        sample_pages=True,
        page_sample_ratio=ratio,
        max_sample_pages=max_p,
    )
    print(f"  샘플링된 .md → {BENCH_DOCS_DIR}")


def _step1_load_qa(args, tracker) -> Tuple[List[str], List[str]]:
    """Step 1: QA 데이터셋 로드 또는 재생성.

    Returns:
        (queries, ground_truths)
    """
    print(f"\n{'=' * 60}")
    print("Step 1: QA 데이터셋 로드")
    print(f"{'=' * 60}")
    with tracker.phase("qa_dataset_load"):
        if getattr(args, "regenerate_qa", False) or getattr(args, "sample_pages", False):
            from rag_bench.scripts.generate_qa import (
                compute_effective_num_qa,
                generate_qa_ragas,
            )
            import argparse as _ap
            _qa_args = _ap.Namespace(
                sample_pages=getattr(args, "sample_pages", False),
                max_qa_per_page=getattr(args, "max_qa_per_page", 2),
            )
            _tmp_parent_pairs, _ = create_parent_child_chunks(
                markdown_dir=str(BENCH_DOCS_DIR),
                parent_store_path=str(BENCH_DATA_DIR / "parent_store"),
            )
            effective_num_qa = compute_effective_num_qa(_qa_args, _tmp_parent_pairs)
            print(f"  QA 재생성: {effective_num_qa}개 (청크 {len(_tmp_parent_pairs)}개 × {_qa_args.max_qa_per_page})")
            with track_openai_tokens() as _qa_tokens:
                qa_pairs_raw = generate_qa_ragas(
                    parent_pairs=_tmp_parent_pairs,
                    num_qa=effective_num_qa,
                    reuse_kg=False,
                )
            if _qa_tokens.total_tokens > 0:
                tracker.add_tokens_breakdown(_qa_tokens, "qa_generation", "openai")
            if qa_pairs_raw:
                import json as _json, hashlib as _hl
                _docs_hash = _hl.md5(
                    "".join(sorted(str(p) for p in BENCH_DOCS_DIR.glob("*.md"))).encode()
                ).hexdigest()[:8]
                _qa_out = {
                    "docs_hash": _docs_hash,
                    "num_qa": len(qa_pairs_raw),
                    "qa_pairs": qa_pairs_raw,
                }
                (BENCH_DATA_DIR / "qa_dataset.json").write_text(
                    _json.dumps(_qa_out, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                qa_pairs = qa_pairs_raw
            else:
                print("  [Warning] QA 재생성 실패 — 기존 qa_dataset.json 사용")
                dataset = load_qa_dataset(BENCH_DATA_DIR)
                qa_pairs = dataset["qa_pairs"]
        else:
            dataset = load_qa_dataset(BENCH_DATA_DIR)
            qa_pairs = dataset["qa_pairs"]
        queries = [qa["question"] for qa in qa_pairs]
        ground_truths = [qa["ground_truth"] for qa in qa_pairs]
    return queries, ground_truths


def _step2_chunk_docs(tracker) -> Tuple[List, List]:
    """Step 2: 문서 청킹.

    Returns:
        (parent_pairs, child_chunks)
    """
    print(f"\n{'=' * 60}")
    print("Step 2: 문서 청킹")
    print(f"{'=' * 60}")
    with tracker.phase("chunking"):
        parent_store_path = BENCH_DATA_DIR / "parent_store"
        parent_pairs, child_chunks = create_parent_child_chunks(
            markdown_dir=str(BENCH_DOCS_DIR),
            parent_store_path=str(parent_store_path),
        )
        if not child_chunks:
            print("Error: Child 청크가 생성되지 않았습니다.")
            sys.exit(1)
    return parent_pairs, child_chunks


def _step25_enrich_contextual(combos, parent_pairs, child_chunks) -> Optional[List]:
    """Step 2.5: Contextual 청크 사전 생성 (해당 조합이 있을 때만).

    Returns:
        pre_enriched_chunks 또는 None (contextual 조합 없을 때)
    """
    if not any(s.llm_support == "contextual" for s in combos):
        return None
    print(f"\n{'=' * 60}")
    print("Step 2.5: Contextual Retrieval 청크 사전 생성")
    print(f"{'=' * 60}")
    from rag_bench.strategies.contextual_retrieval import ContextualRetrievalStrategy
    _ctx_prep = ContextualRetrievalStrategy(
        base_strategy=None,
        parent_pairs=parent_pairs,
    )
    pre_enriched_chunks = _ctx_prep.enrich_only(child_chunks)
    print(f"  사전 생성 완료: {len(pre_enriched_chunks)}개 enriched 청크")
    return pre_enriched_chunks


def _step3_build_strategies(
    args, combos, index_cache, child_chunks, parent_pairs, pre_enriched_chunks, tracker
) -> Tuple[List[Tuple], List]:
    """Step 3: 전략 생성 및 인덱싱.

    Returns:
        (strategies, active_strategies)
        strategies: List[Tuple[ComboSpec, strategy]]
        active_strategies: List[strategy] (전략 객체만)
    """
    print(f"\n{'=' * 60}")
    print("Step 3: 전략 생성 및 인덱싱")
    print(f"{'=' * 60}")

    strategies: List[Tuple[ComboSpec, Any]] = []
    build_results: List[Tuple[str, object, Optional[str]]] = []
    reindex = args.reindex

    with tracker.phase("strategy_build_and_indexing"):
        for i, spec in enumerate(combos, 1):
            progress = f"[{i}/{len(combos)}]"
            label = spec.label
            strategy, err = _safe_build(
                label,
                lambda s=spec: (
                    build_strategy_from_spec(
                        s, index_cache, child_chunks, parent_pairs, reindex,
                        pre_enriched=pre_enriched_chunks,
                    ),
                    None,
                ),
                progress=progress,
                tracker=tracker,
                spec=spec,
            )
            build_results.append((label, strategy, err))
            if strategy is not None:
                strategies.append((spec, strategy))
            _release_memory()

    _print_init_summary(build_results)

    if not strategies:
        print("\n성공한 전략이 없습니다. 종료합니다.")
        sys.exit(1)

    active_strategies = [s for _, s in strategies]
    print(f"\n벤치마크 대상 전략: {len(active_strategies)}개")
    return strategies, active_strategies


def _step4_pass1(
    args, active_strategies, strategies, queries, tracker
) -> Tuple[Any, Optional[Any], bool]:
    """Step 4: Pass 1 — 레이턴시 측정.

    Returns:
        (runner, summary_df, pass1_done)
        pass1_done=True이면 --pass1-only 조기 종료 신호.
    """
    print(f"\n{'=' * 60}")
    print("Step 4: Pass 1 — 레이턴시 측정")
    print(f"{'=' * 60}")
    print(f"  총 검색: {len(active_strategies)}개 전략 x {len(queries)}개 쿼리 = {len(active_strategies) * len(queries)}회")

    pass1_workers = getattr(args, "pass1_workers", 0)
    if pass1_workers > 1:
        print(f"  [병렬 모드] pass1-workers={pass1_workers}")
    runner = BenchmarkRunner(
        strategies=active_strategies,
        queries=queries,
        k=args.k,
        evaluator=None,
        parallel_strategies=pass1_workers,
    )
    with tracker.phase("pass1_latency"):
        runner.run()
    runner.compare()

    latency_df = runner.to_dataframe()
    summary_df = None
    if latency_df is not None:
        latency_path = BENCH_DATA_DIR / "all_combos_latency.csv"
        if getattr(args, "append_results", False) and latency_path.exists():
            import pandas as pd
            existing = pd.read_csv(latency_path)
            new_strategies = latency_df["strategy"].unique()
            existing = existing[~existing["strategy"].isin(new_strategies)]
            merged = pd.concat([existing, latency_df], ignore_index=True)
            merged.to_csv(latency_path, index=False, encoding="utf-8-sig")
            print(f"  레이턴시 결과 병합(append): {latency_path} (기존 {len(existing)}행 + 신규 {len(latency_df)}행)")
            latency_df = merged
        else:
            latency_df.to_csv(latency_path, index=False, encoding="utf-8-sig")
            print(f"  레이턴시 결과: {latency_path}")
        summary_df = _build_latency_summary(latency_df)

        for spec, strat in strategies:
            timing = tracker.find_timing(spec.label)
            if timing is None:
                continue
            mask = latency_df["strategy"] == strat.name
            strat_rows = latency_df[mask]
            if strat_rows.empty:
                continue
            valid_lats = strat_rows.loc[strat_rows["error"].isna(), "latency_ms"].tolist()
            error_count = int(strat_rows["error"].notna().sum())
            tracker.record_query_stats(timing, valid_lats, error_count)

    if args.layers and summary_df is not None:
        _timing_for_layers = None
        _timing_csv = BENCH_DATA_DIR / "combo_timing.csv"
        if _timing_csv.exists():
            try:
                import pandas as _pd_tmp
                _timing_for_layers = _pd_tmp.read_csv(_timing_csv)
            except Exception:
                pass
        _print_layer_contribution(strategies, summary_df, timing_df=_timing_for_layers)

    if args.pass1_only:
        print(f"\n{'═' * 60}")
        print(f" Pass 1 완료 — {len(active_strategies)}개 전략 레이턴시 측정")
        print(f"{'═' * 60}")
        return runner, summary_df, True

    return runner, summary_df, False


def _select_eval_strategies(args, strategies, summary_df) -> List:
    """Pass 1 결과를 기반으로 RAGAS 평가 대상 전략을 선택한다.

    top_n이 전체 전략 수보다 작으면 평균 레이턴시 기준 상위 top_n개만 반환한다.
    """
    top_n = args.top_n or len(strategies)
    if top_n < len(strategies):
        print(f"\n{'=' * 60}")
        print(f"Step 5: Pass 2 — 상위 {top_n}개 RAGAS 평가")
        print(f"{'=' * 60}")
        if summary_df is not None and "avg_latency" in summary_df.columns:
            strategy_latencies = []
            for spec, strat in strategies:
                mask = summary_df["strategy"] == strat.name
                avg_lat = summary_df.loc[mask, "avg_latency"].values[0] if mask.any() else float("inf")
                strategy_latencies.append((spec, strat, avg_lat))
            strategy_latencies.sort(key=lambda x: x[2])
            return [(sp, st) for sp, st, _ in strategy_latencies[:top_n]]
        return strategies[:top_n]

    print(f"\n{'=' * 60}")
    print(f"Step 5: Pass 2 — 전체 {len(strategies)}개 RAGAS 평가")
    print(f"{'=' * 60}")
    return strategies


def _save_ragas_results(args, scores_df, eval_strategies, eval_runner, tracker) -> None:
    """RAGAS 평가 결과를 per-sample CSV, 집계 CSV, RunTracker에 저장한다."""
    if eval_runner.reports:
        per_sample_dir = BENCH_DATA_DIR / "per_sample"
        per_sample_dir.mkdir(parents=True, exist_ok=True)
        for strat_name, report in eval_runner.reports.items():
            if not report.per_sample_df.empty:
                safe_name = strat_name.replace("/", "_").replace(" ", "_")
                report.per_sample_df.to_csv(
                    per_sample_dir / f"{safe_name}.csv", index=False, encoding="utf-8-sig"
                )
        print(f"  per-sample 결과: {per_sample_dir}/")

    scores_path = BENCH_DATA_DIR / "all_combos_ragas.csv"
    if getattr(args, "append_results", False) and scores_path.exists():
        import pandas as pd
        existing_ragas = pd.read_csv(scores_path)
        new_strategies = scores_df["strategy"].unique()
        existing_ragas = existing_ragas[~existing_ragas["strategy"].isin(new_strategies)]
        scores_df = pd.concat([existing_ragas, scores_df], ignore_index=True)
        scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")
        print(f"  RAGAS 점수 병합(append): {scores_path}")
    else:
        scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")
        print(f"  RAGAS 점수: {scores_path}")

    for _, row in scores_df.iterrows():
        strat_name = row["strategy"]
        for spec, strat in eval_strategies:
            if strat.name == strat_name:
                timing = tracker.find_timing(spec.label)
                if timing:
                    metric_cols = [c for c in scores_df.columns if c != "strategy"]
                    scores = {
                        c: round(float(row[c]), 4)
                        for c in metric_cols
                        if isinstance(row[c], (int, float))
                    }
                    tracker.record_ragas(timing, scores)
                break


def _step5_pass2(
    args, strategies, queries, ground_truths, runner, summary_df, tracker
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], List]:
    """Step 5: Pass 2 — RAGAS 평가 (상위 N 또는 전체).

    Returns:
        (scores_df, evaluator, eval_runner, eval_strategies)
    """
    eval_strategies = _select_eval_strategies(args, strategies, summary_df)

    evaluator = None
    if not args.no_ragas:
        from rag_bench.evaluation import ExtendedRAGEvaluator
        from rag_bench.evaluation.metrics import MetricPreset
        try:
            preset_enum = MetricPreset(args.metric_preset)
            evaluator = ExtendedRAGEvaluator(preset=preset_enum)
            print(f"  Evaluator: ExtendedRAGEvaluator (preset={args.metric_preset}, profile={args.scoring_profile})")
        except Exception as e:
            print(f"ExtendedRAGEvaluator 초기화 실패 (RAGAS 평가 건너뜀): {e}")

    scores_df = None
    eval_runner = None
    if evaluator is not None:
        pass2_workers = getattr(args, "pass2_workers", 0)
        if pass2_workers > 1:
            print(f"  [병렬 평가] pass2-workers={pass2_workers}")
        eval_runner = BenchmarkRunner(
            strategies=[s for _, s in eval_strategies],
            queries=queries,
            k=args.k,
            evaluator=evaluator,
            parallel_eval=pass2_workers,
        )
        eval_runner.inject_results(runner._results)
        with tracker.phase("pass2_ragas"):
            with track_openai_tokens() as ragas_tokens:
                scores_df = eval_runner.evaluate(ground_truths=ground_truths)
        if ragas_tokens.total_tokens > 0:
            tracker.record_ragas_tokens(ragas_tokens)
        print_ragas_table(scores_df, scoring_profile=args.scoring_profile)

        if scores_df is not None:
            _save_ragas_results(args, scores_df, eval_strategies, eval_runner, tracker)

        if args.layers and scores_df is not None:
            _print_layer_contribution_ragas(eval_strategies, scores_df)

    return scores_df, evaluator, eval_runner, eval_strategies


def _step6_timing(
    args, strategies, eval_strategies, tracker, summary_df, evaluator, eval_runner, queries
) -> Optional[Any]:
    """Step 6: 조합별 전체 소요 시간 집계.

    Returns:
        timing_df 또는 None
    """
    timing_df = _build_combo_timing_df(
        strategies=strategies,
        tracker=tracker,
        summary_df=summary_df,
        eval_runner=eval_runner if evaluator else None,
        n_queries=len(queries),
    )
    if timing_df is not None:
        timing_path = BENCH_DATA_DIR / "combo_timing.csv"
        if getattr(args, "append_results", False) and timing_path.exists():
            import pandas as _pd
            existing_t = _pd.read_csv(timing_path)
            existing_t = existing_t[~existing_t["label"].isin(timing_df["label"].unique())]
            timing_df = _pd.concat([existing_t, timing_df], ignore_index=True)
        timing_df.to_csv(timing_path, index=False, encoding="utf-8-sig")
        print(f"  조합 타이밍: {timing_path}")
        from rag_bench.utils.report import print_combo_timing_table, print_qa_scaling_table
        print_combo_timing_table(timing_df)
        print_qa_scaling_table(
            timing_df=timing_df,
            n_strategies=len(strategies),
            n_eval_strategies=len(eval_strategies),
        )
    return timing_df


def _step7_report(args, summary_df, scores_df, combos, evaluator, tracker, timing_df) -> None:
    """Step 7: Markdown + HTML 리포트 생성."""
    ragas_df = scores_df if evaluator else None

    if summary_df is not None:
        _generate_report(
            summary_df, ragas_df,
            combos, BENCH_DATA_DIR, tracker=tracker, timing_df=timing_df,
        )

    # HTML 보고서 자동 생성
    try:
        from rag_bench.scripts.generate_html_report import generate_html_report
        from dataclasses import asdict as _asdict

        run_record = None
        if tracker and hasattr(tracker, "_record"):
            try:
                run_record = _asdict(tracker._record)
            except Exception:
                pass

        session_id = ""
        if tracker and hasattr(tracker, "_record"):
            session_id = getattr(tracker._record, "run_id", "")

        html_path = BENCH_DATA_DIR / "benchmark_report.html"
        generate_html_report(
            latency_df=summary_df,
            ragas_df=ragas_df,
            output_path=str(html_path),
            session_id=session_id,
            run_record=run_record,
            timing_df=timing_df,
        )
    except Exception as _html_err:
        print(f"  [HTML 보고서 생성 실패]: {_html_err}")


def _collect_upstage_tokens(strategies, tracker) -> None:
    """Upstage 임베딩 토큰 breakdown 집계."""
    try:
        from rag_bench.strategies.upstage_embed import UpstageEmbedStrategy
    except ImportError:
        return
    for item in strategies:
        # (spec, strat) 튜플이 아닌 경우(flat list 등) 방어
        if isinstance(item, tuple) and len(item) == 2:
            _spec, _strat = item
        else:
            continue
        _base = getattr(_strat, "_base_strategy", _strat)
        if isinstance(_base, UpstageEmbedStrategy):
            _idx, _qry = _base.get_raw_token_usages()
            if _idx.total_tokens > 0:
                tracker.add_tokens_breakdown(_idx, "embedding_indexing", "upstage")
            if _qry.total_tokens > 0:
                tracker.add_tokens_breakdown(_qry, "embedding_query", "upstage")


# ===========================================================================
# 새 모드: 4-Layer 조합 실행 — 오케스트레이터
# ===========================================================================


def _run_preset_mode(args):
    """--preset 기반 새 4-Layer 조합 실행."""
    setup_ssl_bypass()

    preset_name = args.preset
    if preset_name not in PRESETS:
        print(f"Error: 알 수 없는 프리셋: {preset_name}")
        print(f"  사용 가능: {list(PRESETS.keys())}")
        sys.exit(1)

    config = PRESETS[preset_name]
    combos = generate_valid_combinations(config)

    if getattr(args, "dense_filter", None):
        filter_models = [m.strip() for m in args.dense_filter.split(",")]
        combos = [c for c in combos if c.dense in filter_models]
        if not combos:
            print(f"Error: --dense-filter '{args.dense_filter}'에 해당하는 조합이 없습니다.")
            sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f" 4-Layer 조합 벤치마크 — 프리셋: {preset_name}")
    print(f"{'═' * 60}")
    print(f"  Dense Models: {config['dense_models']}")
    print(f"  Sparse Models: {config['sparse_models']}")
    print(f"  Rerankers: {config['rerankers']}")
    print(f"  LLM Support: {config['llm_support']}")
    print(f"  총 조합: {len(combos)}개")

    if args.dry_run:
        print(f"\n{'─' * 80}")
        print(f" {'#':>3}  {'Label':<40} {'Retrieval Mode':<50}")
        print(f"{'─' * 80}")
        for i, spec in enumerate(combos, 1):
            print(f" {i:>3}  {spec.label:<40} {spec.retrieval_mode:<50}")
        print(f"{'─' * 80}")
        print(f" 합계: {len(combos)}개 유효 조합")
        unique_keys = set(spec.index_key for spec in combos)
        print(f" 고유 인덱스: {len(unique_keys)}개 (실제 인덱싱 횟수)")
        if args.layers:
            _print_layer_analysis_preview(combos, config)
        return

    tracker = RunTracker(output_dir=BENCH_DATA_DIR)

    if getattr(args, "sample_pages", False):
        _step0_sample_pdfs(args)

    queries, ground_truths = _step1_load_qa(args, tracker)
    parent_pairs, child_chunks = _step2_chunk_docs(tracker)

    tracker.set_config(
        preset=preset_name,
        k=args.k,
        top_n=args.top_n,
        pass1_only=args.pass1_only,
        layers=args.layers,
        num_combos=len(combos),
        num_queries=len(queries),
        num_docs=len(child_chunks),
    )

    pre_enriched_chunks = _step25_enrich_contextual(combos, parent_pairs, child_chunks)

    index_cache = IndexCacheManager()
    strategies, active_strategies = _step3_build_strategies(
        args, combos, index_cache, child_chunks, parent_pairs, pre_enriched_chunks, tracker
    )

    runner, summary_df, pass1_done = _step4_pass1(
        args, active_strategies, strategies, queries, tracker
    )
    if pass1_done:
        tracker.finalize()
        _cleanup_strategies(active_strategies)
        return

    scores_df, evaluator, eval_runner, eval_strategies = _step5_pass2(
        args, strategies, queries, ground_truths, runner, summary_df, tracker
    )

    timing_df = _step6_timing(
        args, strategies, eval_strategies, tracker, summary_df, evaluator, eval_runner, queries
    )

    _step7_report(args, summary_df, scores_df, combos, evaluator, tracker, timing_df)
    try:
        _collect_upstage_tokens(strategies, tracker)
    except Exception as _e:
        print(f"  [토큰 집계 오류 무시]: {_e}")
    tracker.finalize()
    _cleanup_strategies(active_strategies)

    print(f"\n{'═' * 60}")
    print(f" 벤치마크 완료 — {len(active_strategies)}개 전략 비교")
    print(f"{'═' * 60}")


# ===========================================================================
# 레이어 기여도 분석
# ===========================================================================

# 4-Layer 정의 — (레이어 이름, ComboSpec 필드 접근 람다) 공통 상수.
# _print_layer_analysis_preview / _print_layer_contribution /
# _print_layer_contribution_ragas 에서 공유하여 레이어 명칭 중복을 방지한다.
_LAYER_DEFINITIONS = [
    ("Layer 1 — Dense Model", lambda s: s.dense),
    ("Layer 2 — Sparse Model", lambda s: s.sparse),
    ("Layer 3 — Reranker", lambda s: s.reranker or "none"),
    ("Layer 4 — Contextual", lambda s: s.llm_support or "none"),
]


def _print_layer_analysis_preview(combos: List[ComboSpec], config: dict):
    """dry-run 시 레이어 분석 미리보기."""
    print(f"\n{'═' * 60}")
    print(" 레이어별 조합 분포")
    print(f"{'═' * 60}")

    config_values = [
        config["dense_models"],
        config["sparse_models"],
        config["rerankers"],
        config["llm_support"],
    ]
    for (layer_name, get_val), values in zip(_LAYER_DEFINITIONS, config_values):
        print(f"\n  {layer_name}:")
        for val in values:
            count = sum(1 for c in combos if get_val(c) == val)
            print(f"    {str(val) or 'None':<20} → {count}개 조합")


def _build_latency_summary(latency_df):
    """쿼리별 raw DataFrame → 전략별 요약 DataFrame (avg_latency 등)."""

    valid = latency_df[latency_df["error"].isna()].copy()
    if valid.empty:
        return None
    summary = (
        valid.groupby("strategy")["latency_ms"]
        .agg(avg_latency="mean", min_latency="min", max_latency="max",
             p50_latency="median", query_count="count")
        .reset_index()
    )
    # ms → s 변환 (avg_latency)
    summary["avg_latency"] = summary["avg_latency"] / 1000.0
    summary["min_latency"] = summary["min_latency"] / 1000.0
    summary["max_latency"] = summary["max_latency"] / 1000.0
    summary["p50_latency"] = summary["p50_latency"] / 1000.0
    return summary


def _print_layer_contribution(strategies: List[Tuple[ComboSpec, Any]], summary_df, timing_df=None):
    """레이턴시 기반 레이어 기여도 출력.

    Layer 1~3: 쿼리 평균 레이턴시(avg_latency) 기준
    Layer 4 (Contextual): 인덱싱 소요시간(build_s) 기준 — 쿼리 레이턴시에 영향 없음
    """
    print(f"\n{'═' * 60}")
    print(" 레이어별 기여도 분석")
    print(f"{'═' * 60}")

    if summary_df is None or summary_df.empty:
        print("  (레이턴시 데이터 없음)")
        return

    # 전략명 → 평균 레이턴시(s) 매핑
    lat_map = {}
    for _, row in summary_df.iterrows():
        lat_map[row["strategy"]] = row["avg_latency"]

    # Layer 1~3: 쿼리 레이턴시 기준
    print("\n  [지표: 쿼리 평균 레이턴시 (s)]")
    for layer_name, get_val in _LAYER_DEFINITIONS[:3]:
        print(f"\n  {layer_name}:")
        val_lats: Dict[str, List[float]] = {}
        for spec, strat in strategies:
            val = get_val(spec)
            lat = lat_map.get(strat.name, None)
            if lat is not None:
                val_lats.setdefault(val, []).append(lat)

        for val, lats in sorted(val_lats.items()):
            avg = sum(lats) / len(lats) if lats else 0
            print(f"    {val:<20} → {avg:.3f}s (n={len(lats)})")

    # Layer 4: 인덱싱 소요시간 기준
    print(f"\n  Layer 4 — Contextual [지표: 인덱싱 소요시간 build_s]:")
    if timing_df is not None and "build_s" in timing_df.columns and "llm_support" in timing_df.columns:
        grp = timing_df.groupby("llm_support")["build_s"]
        for val, group in grp:
            avg_build = group.mean()
            label = val if val and val != "nan" else "none"
            print(f"    {label:<20} → {avg_build:.1f}s avg build (n={len(group)})")
    else:
        # timing_df 없으면 쿼리 레이턴시로 대체 출력
        val_lats = {}
        for spec, strat in strategies:
            val = spec.llm_support or "none"
            lat = lat_map.get(strat.name, None)
            if lat is not None:
                val_lats.setdefault(val, []).append(lat)
        for val, lats in sorted(val_lats.items()):
            avg = sum(lats) / len(lats) if lats else 0
            print(f"    {val:<20} → {avg:.3f}s query latency (n={len(lats)}) [build_s 데이터 없음]")


def _print_layer_contribution_ragas(strategies: List[Tuple[ComboSpec, Any]], scores_df):
    """RAGAS 기반 레이어 기여도 출력."""
    print(f"\n{'═' * 60}")
    print(" 레이어별 RAGAS 점수 기여도")
    print(f"{'═' * 60}")

    metric_cols = [c for c in scores_df.columns if c not in ("strategy",)]

    # 전략명 → 점수 매핑
    score_map = {}
    for _, row in scores_df.iterrows():
        score_map[row["strategy"]] = {col: row[col] for col in metric_cols if isinstance(row[col], float)}

    for layer_name, get_val in _LAYER_DEFINITIONS:
        print(f"\n  {layer_name}:")
        val_scores: Dict[str, List[Dict[str, float]]] = {}
        for spec, strat in strategies:
            val = get_val(spec)
            scores = score_map.get(strat.name, None)
            if scores:
                val_scores.setdefault(val, []).append(scores)

        for val, score_list in sorted(val_scores.items()):
            if not score_list:
                continue
            # 각 메트릭의 평균
            avg_parts = []
            for mc in metric_cols:
                vals = [s.get(mc, 0) for s in score_list if mc in s]
                if vals:
                    avg_parts.append(f"{mc}={sum(vals)/len(vals):.3f}")
            print(f"    {val:<20} → {', '.join(avg_parts[:4])} (n={len(score_list)})")


# ===========================================================================
# 리포트 생성
# ===========================================================================


def _report_env_section(lines: list, tracker) -> None:
    """리포트에 실행 환경 섹션을 추가한다."""
    if not (tracker and hasattr(tracker, '_record')):
        return
    rec = tracker._record
    pf = rec.platform_info
    lines.append("## 실행 환경")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("|------|-----|")
    lines.append(f"| Run ID | {rec.run_id} |")
    lines.append(f"| Preset | {rec.preset} |")
    lines.append(f"| Platform | {pf.get('os', '')} {pf.get('os_release', '')} |")
    chip = pf.get("apple_chip", pf.get("processor", "N/A"))
    lines.append(f"| Chip / CPU | {chip} ({pf.get('cpu_count_logical', '?')} cores) |")
    lines.append(f"| RAM | {pf.get('ram_total_gb', '?')} GB |")
    lines.append(f"| GPU | {pf.get('gpu') or 'None'} |")
    lines.append(f"| Python | {pf.get('python_version', '')} |")
    lines.append(f"| Git Commit | {pf.get('git_commit', '')} |")
    lines.append("")

    if tracker._phases:
        total_s = rec.duration_s or 1
        lines.append("## 단계별 소요 시간")
        lines.append("")
        lines.append("| 단계 | 소요 시간 | 비중 | 토큰 |")
        lines.append("|------|----------|:----:|------|")
        for p in tracker._phases:
            if p.duration_s <= 0:
                continue
            pct = p.duration_s / total_s * 100
            tok_str = ""
            if p.tokens and p.tokens.get("total_tokens", 0) > 0:
                tok_str = f"{p.tokens['total_tokens']:,}"
            lines.append(f"| {p.phase} | {p.duration_s:.1f}s | {pct:.1f}% | {tok_str} |")
        lines.append("")

    tt = tracker._token_total
    if tt.total_tokens > 0:
        lines.append("## 토큰 사용량")
        lines.append("")
        lines.append("| 항목 | 값 |")
        lines.append("|------|-----|")
        lines.append(f"| Total Tokens | {tt.total_tokens:,} |")
        lines.append(f"| Prompt | {tt.prompt_tokens:,} |")
        lines.append(f"| Completion | {tt.completion_tokens:,} |")
        lines.append(f"| API Cost | ${tt.total_cost_usd:.4f} |")
        lines.append(f"| LLM Calls | {tt.num_calls} |")
        lines.append("")


def _report_latency_section(lines: list, latency_summary_df) -> None:
    """리포트에 레이턴시 결과 섹션을 추가한다."""
    lines.extend(["---", "", "## 레이턴시 결과 (Top 10)", ""])
    if latency_summary_df is not None and "strategy" in latency_summary_df.columns:
        if "avg_latency" in latency_summary_df.columns:
            sorted_df = latency_summary_df.sort_values("avg_latency")
            lines.append("| # | 전략 | 평균 레이턴시 |")
            lines.append("|---|------|:----------:|")
            for i, (_, row) in enumerate(sorted_df.head(10).iterrows(), 1):
                lines.append(f"| {i} | {row['strategy']} | {row['avg_latency']:.3f}s |")
        lines.append("")


def _report_timing_section(lines: list, timing_df) -> None:
    """리포트에 조합별 타이밍 섹션을 추가한다."""
    if timing_df is None or timing_df.empty:
        return
    lines.append("## 조합별 전체 소요 시간")
    lines.append("")
    lines.append("| 조합 | Dense | Sparse | Reranker | LLM | 빌드(s) | Pass1(s) | Pass2(s) | 합계(s) |")
    lines.append("|------|-------|--------|----------|-----|:-------:|:--------:|:--------:|:-------:|")
    for _, row in timing_df.sort_values("total_s", ascending=False).iterrows():
        lines.append(
            f"| {row['label']} | {row['dense']} | {row['sparse']} | {row['reranker']} | {row['llm_support']}"
            f" | {row['build_s']:.1f} | {row['pass1_s']:.1f} | {row['pass2_s']:.1f} | **{row['total_s']:.1f}** |"
        )
    lines.append("")

    lines.append("### 레이어별 평균 소요 시간")
    lines.append("")
    for layer_col, layer_name in [
        ("dense", "Dense Model"), ("sparse", "Sparse"),
        ("reranker", "Reranker"), ("llm_support", "LLM Support"),
    ]:
        lines.append(f"**{layer_name}**")
        lines.append("")
        lines.append("| 값 | 빌드(s) | Pass1(s) | Pass2(s) | 합계(s) |")
        lines.append("|---|:-------:|:--------:|:--------:|:-------:|")
        for val, grp in timing_df.groupby(layer_col):
            lines.append(
                f"| {val} | {grp['build_s'].mean():.1f} | {grp['pass1_s'].mean():.1f}"
                f" | {grp['pass2_s'].mean():.1f} | {grp['total_s'].mean():.1f} |"
            )
        lines.append("")


def _report_ragas_section(lines: list, ragas_df) -> None:
    """리포트에 RAGAS 평가 결과 + 가중 점수 섹션을 추가한다."""
    if ragas_df is None or ragas_df.empty:
        return
    lines.append("## RAGAS 평가 결과")
    lines.append("")
    metric_cols = [c for c in ragas_df.columns if c != "strategy"]
    header = "| 전략 | " + " | ".join(metric_cols) + " |"
    sep = "|------|" + "|".join(":---:" for _ in metric_cols) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in ragas_df.iterrows():
        vals = " | ".join(
            f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
            for c in metric_cols
        )
        lines.append(f"| {row['strategy']} | {vals} |")
    lines.append("")

    from rag_bench.evaluation.evaluator import SCORING_PROFILES
    lines.append("## 가중 점수 (Scoring Profiles)")
    lines.append("")
    profile_names = list(SCORING_PROFILES.keys())
    header = "| 전략 | " + " | ".join(profile_names) + " |"
    sep = "|------|" + "|".join(":---:" for _ in profile_names) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in ragas_df.iterrows():
        vals = []
        for pname in profile_names:
            weights = SCORING_PROFILES[pname]
            ws = sum(
                row.get(metric, 0.0) * weight
                for metric, weight in weights.items()
                if isinstance(row.get(metric, 0.0), (int, float))
            )
            vals.append(f"{ws:.4f}")
        lines.append(f"| {row['strategy']} | " + " | ".join(vals) + " |")
    lines.append("")


def _generate_report(latency_summary_df, ragas_df, combo_specs, output_dir, tracker=None, timing_df=None):
    """Markdown 리포트 생성. latency_summary_df는 전략별 요약 DataFrame."""
    report_path = output_dir / "e2e_report.md"

    lines = [
        "# E2E 4-Layer 조합 벤치마크 리포트",
        "",
        f"**조합 수**: {len(combo_specs)}개",
        f"**생성 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    _report_env_section(lines, tracker)
    _report_latency_section(lines, latency_summary_df)
    _report_timing_section(lines, timing_df)
    _report_ragas_section(lines, ragas_df)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  리포트: {report_path}")


# ===========================================================================
# 공통 유틸리티
# ===========================================================================


def _build_combo_timing_df(
    strategies: List[Tuple[ComboSpec, Any]],
    tracker,
    summary_df,
    eval_runner=None,
    n_queries: int = 0,
):
    """조합별 전체 소요 시간 DataFrame을 생성한다.

    컬럼:
        label, dense, sparse, reranker, llm_support,
        build_s           — 인덱싱(빌드) 소요 시간
        pass1_s           — Pass 1 검색 총 소요 시간 (avg_latency × n_queries)
        pass1_s_per_qa    — 조합당 QA 1개 평균 검색 시간
        pass2_s           — Pass 2 RAGAS 평가 소요 시간 (전략별 실측, 없으면 0)
        pass2_s_per_qa    — 조합당 QA 1개 평균 RAGAS 시간 (0이면 미평가)
        total_s           — build_s + pass1_s + pass2_s
        n_queries         — 실행에 사용된 QA 수
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    rows = []
    for spec, strat in strategies:
        timing = tracker.find_timing(spec.label)
        build_s = timing.build_time_s if timing else 0.0

        # Pass 1: avg_latency(s) × query_count
        pass1_s = 0.0
        n_q_actual = n_queries
        if summary_df is not None and "strategy" in summary_df.columns:
            mask = summary_df["strategy"] == strat.name
            if mask.any():
                avg_lat_s = summary_df.loc[mask, "avg_latency"].values[0]  # 이미 초 단위
                n_q_actual = int(summary_df.loc[mask, "query_count"].values[0])
                pass1_s = round(avg_lat_s * n_q_actual, 2)

        pass1_per_qa = round(pass1_s / n_q_actual, 3) if n_q_actual > 0 else 0.0

        # Pass 2: eval_runner._eval_times 에서 전략별 실측
        pass2_s = 0.0
        if eval_runner is not None and hasattr(eval_runner, "_eval_times"):
            pass2_s = eval_runner._eval_times.get(strat.name, 0.0)

        pass2_per_qa = round(pass2_s / n_q_actual, 3) if (n_q_actual > 0 and pass2_s > 0) else 0.0

        rows.append({
            "label": spec.label,
            "dense": spec.dense,
            "sparse": spec.sparse,
            "reranker": spec.reranker or "none",
            "llm_support": spec.llm_support or "none",
            "build_s": round(build_s, 2),
            "pass1_s": pass1_s,
            "pass1_s_per_qa": pass1_per_qa,
            "pass2_s": round(pass2_s, 2),
            "pass2_s_per_qa": pass2_per_qa,
            "total_s": round(build_s + pass1_s + pass2_s, 2),
            "n_queries": n_q_actual,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def _release_memory():
    """PyTorch 캐시 + 가비지 컬렉션 강제 해제."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cleanup_strategies(strategies):
    """전략 클린업."""
    print(f"\n{'=' * 60}")
    print("클린업")
    print(f"{'=' * 60}")
    for strategy in strategies:
        try:
            strategy.cleanup()
            print(f"  ✓ {strategy.name}")
        except Exception as e:
            print(f"  ✗ {strategy.name}: {e}")
    _release_memory()
    print("  ✓ 메모리 캐시 해제 완료")


# ===========================================================================
# main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="전체 조합 벤치마크 — 4-Layer 교차 조합 + 2-Pass 실행"
    )

    parser.add_argument("--k", type=int, default=3, help="검색 결과 수 (기본: 3)")
    parser.add_argument("--no_ragas", action="store_true", help="RAGAS 평가 건너뛰기")
    parser.add_argument("--reindex", action="store_true", help="기존 인덱스 삭제 후 재인덱싱")
    parser.add_argument("--preset", type=str, required=True,
                        help="프리셋 선택: quick|standard|full")
    parser.add_argument("--pass1-only", action="store_true",
                        help="레이턴시만 측정 (RAGAS 없음)")
    parser.add_argument("--pass1-workers", type=int, default=0,
                        help="Pass 1 전략 병렬 워커 수 (기본: 0=순차). 예: --pass1-workers 4")
    parser.add_argument("--pass2-workers", type=int, default=0,
                        help="Pass 2 전략 병렬 평가 워커 수 (기본: 0=순차). 예: --pass2-workers 4")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Pass 1 후 상위 N 조합만 RAGAS 평가")
    parser.add_argument("--dry-run", action="store_true",
                        help="조합 목록만 출력 (실행 안 함)")
    parser.add_argument("--layers", action="store_true",
                        help="레이어별 기여도 분석 출력")
    parser.add_argument("--metric-preset", type=str, default="core_only",
                        choices=["core_only", "full", "reference_free", "comprehensive"],
                        help="메트릭 프리셋 (기본: core_only)")
    parser.add_argument("--scoring-profile", type=str, default="balanced",
                        choices=["balanced", "precision_critical", "speed_critical", "comprehensive"],
                        help="스코어링 프로파일 (기본: balanced)")
    parser.add_argument("--dense-filter", type=str, default=None,
                        help="실행할 dense 모델 필터 (쉼표 구분). 예: --dense-filter upstage,openai-large")
    parser.add_argument("--append-results", action="store_true",
                        help="기존 latency/RAGAS CSV에 결과를 병합(append)하여 저장")
    # PDF 페이지 샘플링
    parser.add_argument("--sample-pages", action="store_true",
                        help="docs/*.pdf를 페이지 샘플링하여 rag_bench/docs/*.md 재생성 후 인덱싱")
    parser.add_argument("--page-sample-ratio", type=float, default=0.1,
                        help="페이지 샘플링 비율 (기본: 0.1 = 10%%)")
    parser.add_argument("--max-sample-pages", type=int, default=5,
                        help="최대 샘플 페이지 수 (기본: 5)")
    parser.add_argument("--max-qa-per-page", type=int, default=2,
                        help="청크당 QA 수 — QA 재생성 시 총 QA = 청크 수 × 이 값 (기본: 2)")
    parser.add_argument("--regenerate-qa", action="store_true",
                        help="기존 qa_dataset.json 무시하고 현재 문서에서 QA 재생성")

    args = parser.parse_args()
    _run_preset_mode(args)


if __name__ == "__main__":
    main()
