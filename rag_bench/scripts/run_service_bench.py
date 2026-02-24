"""
서비스 벤치마크 오케스트레이터 — 문서 종류별 최적 RAG 모델 선정.

두 가지 실행 모드:
  --mode hf   : HuggingFace 표준 데이터셋 사용 (권장)
  --mode docs : 사용자 문서 디렉토리 기반

고정 파이프라인 (service 프리셋):
  [Dense Model] × [Sparse Model] + ColBERT Reranker + Contextual Retrieval
  → 4 Dense × 2 Sparse = 8개 조합 비교

사용 예:
    # HuggingFace 표준 데이터셋 모드
    python -m rag_bench.scripts.run_service_bench \\
        --mode hf \\
        --categories general,legal,business,medical

    # 사용자 문서 모드
    python -m rag_bench.scripts.run_service_bench \\
        --mode docs \\
        --docs_dir /path/to/user/docs \\
        --num_qa 20

    # 빠른 테스트 (RAGAS 없이 레이턴시만)
    python -m rag_bench.scripts.run_service_bench \\
        --mode hf --categories general --pass1_only

    # 재실행 (완료된 카테고리 스킵)
    python -m rag_bench.scripts.run_service_bench \\
        --mode hf --categories general,legal,business,medical
"""

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_bench.config import (
    BENCH_DATA_DIR,
    BENCH_DOCS_DIR,
    setup_ssl_bypass,
)
from rag_bench.combo import (
    CacheConfig, ComboSpec, IndexCacheManager, PRESETS,
    build_strategy_from_spec, generate_valid_combinations,
)
from rag_bench.document_types.types import DocType, DOC_TYPE_METADATA
from rag_bench.runner import BenchmarkRunner
from rag_bench.utils.report import print_ragas_table


# ---------------------------------------------------------------------------
# 출력 디렉토리 기본값
# ---------------------------------------------------------------------------
SERVICE_RUN_DIR = BENCH_DATA_DIR / "service_run"


# ---------------------------------------------------------------------------
# 체크포인트 시스템
# ---------------------------------------------------------------------------

class CheckpointManager:
    """카테고리별 벤치마크 진행 상태를 JSON으로 영속화한다."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = run_dir / "checkpoint.json"
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        if self._state_file.exists():
            try:
                return json.loads(self._state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"completed": [], "started": {}, "meta": {}}

    def _save(self) -> None:
        self._state_file.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def is_completed(self, category: str) -> bool:
        return category in self._state["completed"]

    def mark_started(self, category: str, meta: Dict) -> None:
        self._state["started"][category] = {"t": time.time(), **meta}
        self._save()

    def mark_completed(self, category: str, result_path: str) -> None:
        if category not in self._state["completed"]:
            self._state["completed"].append(category)
        self._state.setdefault("results", {})[category] = result_path
        self._save()
        print(f"  [체크포인트] {category} 완료 저장 → {self._state_file}")

    def summary(self) -> Dict:
        return {
            "completed": self._state.get("completed", []),
            "result_files": self._state.get("results", {}),
        }


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------

def _save_category_result(
    category: str,
    run_dir: Path,
    latency_df,
    ragas_df,
    qa_pairs: List[Dict],
) -> str:
    """카테고리별 결과를 JSON + CSV로 저장한다."""
    cat_dir = run_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    # QA 쌍 저장
    qa_path = cat_dir / "qa_pairs.json"
    qa_path.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    # 레이턴시 CSV
    if latency_df is not None:
        latency_df.to_csv(cat_dir / "latency.csv", index=False, encoding="utf-8-sig")

    # RAGAS 결과 JSON + CSV
    result: Dict[str, Any] = {
        "category": category,
        "n_qa": len(qa_pairs),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if ragas_df is not None:
        result["ragas"] = ragas_df.to_dict(orient="records")
        ragas_df.to_csv(cat_dir / "ragas.csv", index=False, encoding="utf-8-sig")

    result_path = cat_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(result_path)


# ---------------------------------------------------------------------------
# 데이터 준비 — HF 모드
# ---------------------------------------------------------------------------

def _prepare_hf_data(
    doc_type: DocType,
    cat_dir: Path,
    max_corpus: int,
    max_queries: int,
    use_cache: bool = True,
) -> Tuple[List, List, List[Dict]]:
    """HuggingFace 데이터셋을 로드하고 parent-child 청크로 변환한다.

    Returns:
        (parent_pairs, child_chunks, qa_pairs)
    """
    from rag_bench.datasets.hf_loader import HFDatasetLoader, beir_to_parent_child_chunks

    loader = HFDatasetLoader(max_corpus=max_corpus, max_queries=max_queries)
    cache_dir = cat_dir / "hf_cache"

    # 캐시 확인
    source_names = {
        DocType.GENERAL:  "miracl-ko",
        DocType.LEGAL:    "markers_bm-law",
        DocType.BUSINESS: "markers_bm-finance+public+commerce",
        DocType.MEDICAL:  "publichealth-qa-ko",
    }
    source_name = source_names.get(doc_type, "unknown")

    dataset = None
    if use_cache:
        dataset = HFDatasetLoader.load_cache(cache_dir, source_name)
        if dataset:
            print(f"  [캐시 로드] {source_name} ({dataset.n_docs:,}docs / {dataset.n_queries:,}queries)")

    if dataset is None:
        print(f"\n  HuggingFace 데이터셋 로드 중: {doc_type.value}")
        dataset = loader.load(doc_type)
        if use_cache:
            loader.save_cache(dataset, cache_dir)

    if dataset.n_docs == 0:
        raise ValueError(f"{doc_type.value} 데이터셋 코퍼스가 비어 있습니다.")

    print(f"  코퍼스: {dataset.n_docs:,}개 | 쿼리: {dataset.n_queries:,}개")

    # BeIR → Parent-Child 청크
    print("  Parent-Child 청킹 중...")
    parent_pairs, child_chunks = beir_to_parent_child_chunks(dataset)
    print(f"  청크: {len(parent_pairs):,} parents / {len(child_chunks):,} children")

    if not child_chunks:
        raise ValueError(f"{doc_type.value} child 청크가 생성되지 않았습니다.")

    # QA 쌍 (BeIR qrels 기반)
    qa_pairs = dataset.get_qa_pairs()
    if not qa_pairs:
        raise ValueError(f"{doc_type.value} QA 쌍이 없습니다.")

    return parent_pairs, child_chunks, qa_pairs


# ---------------------------------------------------------------------------
# 데이터 준비 — 사용자 문서 모드
# ---------------------------------------------------------------------------

def _prepare_docs_data(
    doc_type: DocType,
    docs_dir: Path,
    cat_dir: Path,
    num_qa: int,
    llm_model: str = "gpt-4o-mini",
) -> Tuple[List, List, List[Dict]]:
    """사용자 문서를 파싱하고 QA를 LLM으로 생성한다.

    Returns:
        (parent_pairs, child_chunks, qa_pairs)
    """
    from rag_bench.indexing.multi_parser import parse_directory
    from rag_bench.document_types.classifier import classify_document
    from rag_bench.document_types.sampler import sample_text
    from rag_bench.indexing.chunker import create_parent_child_chunks

    # 해당 카테고리에 속하는 파일 필터링
    print(f"  {docs_dir} 에서 {doc_type.value} 문서 검색 중...")
    all_files = list(parse_directory(docs_dir, skip_errors=True))

    cat_files = []
    for fpath, text in all_files:
        if not text.strip():
            continue
        classified = classify_document(text, min_score=1)
        if classified == doc_type:
            cat_files.append((fpath, text))

    if not cat_files:
        raise ValueError(
            f"{doc_type.value} 카테고리 문서가 {docs_dir}에서 발견되지 않았습니다. "
            f"--doc_type 옵션으로 수동 분류를 사용하거나 해당 카테고리 문서를 추가하세요."
        )

    print(f"  {doc_type.value} 문서: {len(cat_files)}개")

    # 샘플링 적용
    md_dir = cat_dir / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)
    for fpath, text in cat_files:
        sampled = sample_text(text, doc_type)
        out_path = md_dir / (fpath.stem + ".md")
        out_path.write_text(sampled, encoding="utf-8")

    # Parent-Child 청킹
    parent_pairs, child_chunks = create_parent_child_chunks(
        markdown_dir=str(md_dir),
        parent_store_path=str(cat_dir / "parent_store"),
    )

    if not child_chunks:
        raise ValueError(f"{doc_type.value} child 청크가 생성되지 않았습니다.")

    # QA 생성 (LLM)
    print(f"  QA 생성 중 ({num_qa}개)...")
    from rag_bench.scripts.generate_qa import generate_qa_ragas
    qa_pairs_raw = generate_qa_ragas(
        parent_pairs=parent_pairs,
        num_qa=num_qa,
        reuse_kg=False,
    )
    if not qa_pairs_raw:
        raise ValueError(f"{doc_type.value} QA 생성 실패.")

    return parent_pairs, child_chunks, qa_pairs_raw


# ---------------------------------------------------------------------------
# 벤치마크 실행 (카테고리별 공통 로직)
# ---------------------------------------------------------------------------

def _run_category_bench(
    doc_type: DocType,
    parent_pairs: List,
    child_chunks: List,
    qa_pairs: List[Dict],
    index_cache: IndexCacheManager,
    combos: List[ComboSpec],
    run_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Optional[Any], Optional[Any]]:
    """단일 카테고리에 대해 서비스 벤치마크를 실행한다.

    Returns:
        (latency_df, ragas_df)
    """
    cat_name = doc_type.value
    cat_qdrant_dir = run_dir / cat_name / "qdrant"
    cat_qdrant_dir.mkdir(parents=True, exist_ok=True)

    # 카테고리별 Qdrant 경로로 캐시 키 재설정
    # 각 카테고리가 독립적인 인덱스를 사용하도록 index_cache를 새로 생성
    cat_cache_cfg = CacheConfig(
        colbert_model=index_cache.config.colbert_model,
        colbert_device=index_cache.config.colbert_device,
        dense_device=index_cache.config.dense_device,
        flashrank_model=index_cache.config.flashrank_model,
        flashrank_max_length=index_cache.config.flashrank_max_length,
        contextual_llm=index_cache.config.contextual_llm,
        rerank_n=index_cache.config.rerank_n,
    )
    cat_cache = IndexCacheManager(cat_cache_cfg)
    # ColBERT 모델은 전역 캐시에서 공유 (재로드 방지)
    cat_cache._colbert_model = index_cache._colbert_model
    cat_cache._colbert_lock = index_cache._colbert_lock
    cat_cache._flashrank_ranker = index_cache._flashrank_ranker

    queries = [qa["question"] for qa in qa_pairs]
    ground_truths = [qa.get("ground_truth", "") for qa in qa_pairs]

    # Contextual 청크 사전 생성
    pre_enriched: Optional[List] = None
    if any(c.llm_support == "contextual" for c in combos):
        print(f"\n  [{cat_name}] Contextual 청크 사전 생성 중...")
        from rag_bench.strategies.contextual_retrieval import ContextualRetrievalStrategy
        _ctx_prep = ContextualRetrievalStrategy(
            base_strategy=None,
            parent_pairs=parent_pairs,
            llm_model=args.contextual_llm,
        )
        pre_enriched = _ctx_prep.enrich_only(child_chunks)
        print(f"  [{cat_name}] Enriched 청크: {len(pre_enriched):,}개")

    # 전략 빌드 (카테고리별 Qdrant 경로 = run_dir/cat_name/qdrant/...)
    strategies = []
    for i, spec in enumerate(combos, 1):
        label = spec.label
        progress = f"[{i}/{len(combos)}]"
        print(f"\n  {progress} [{cat_name}] {label} 빌드 중...")

        current = build_strategy_from_spec(
            spec=spec,
            index_cache=cat_cache,
            child_chunks=child_chunks,
            parent_pairs=parent_pairs,
            reindex=args.reindex,
            pre_enriched=pre_enriched,
            qdrant_base_dir=cat_qdrant_dir,
        )

        strategies.append(current)
        _release_memory()

    # ColBERT/FlashRank 모델을 전역 캐시에 공유 (다른 카테고리에서 재사용)
    if cat_cache._colbert_model is not None:
        index_cache._colbert_model = cat_cache._colbert_model
    if cat_cache._flashrank_ranker is not None:
        index_cache._flashrank_ranker = cat_cache._flashrank_ranker

    if not strategies:
        print(f"  [{cat_name}] 생성된 전략이 없습니다.")
        return None, None

    # Pass 1: 레이턴시 측정
    print(f"\n  [{cat_name}] Pass 1 — 레이턴시 측정 ({len(strategies)}개 전략 × {len(queries)}개 쿼리)")
    runner = BenchmarkRunner(
        strategies=strategies,
        queries=queries,
        k=args.k,
        evaluator=None,
    )
    runner.run()
    runner.compare()
    latency_df = runner.to_dataframe()

    if args.pass1_only:
        return latency_df, None

    # Pass 2: RAGAS 평가
    print(f"\n  [{cat_name}] Pass 2 — RAGAS 평가")
    ragas_df = None
    if not args.no_ragas:
        try:
            from rag_bench.evaluation import ExtendedRAGEvaluator
            from rag_bench.evaluation.metrics import MetricPreset
            evaluator = ExtendedRAGEvaluator(preset=MetricPreset("standard"))
            eval_runner = BenchmarkRunner(
                strategies=strategies,
                queries=queries,
                k=args.k,
                evaluator=evaluator,
            )
            eval_runner.inject_results(runner._results)
            ragas_df = eval_runner.evaluate(ground_truths=ground_truths)
            if ragas_df is not None:
                print_ragas_table(ragas_df)
        except Exception as e:
            print(f"  [{cat_name}] RAGAS 평가 실패: {e}")
            traceback.print_exc()

    return latency_df, ragas_df


def _release_memory():
    """GC + GPU 캐시 정리."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="서비스 벤치마크 — 문서 종류별 최적 RAG 모델 선정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 모드
    parser.add_argument(
        "--mode", choices=["hf", "docs"], default="hf",
        help="hf: HuggingFace 표준 데이터셋 / docs: 사용자 문서 디렉토리 (기본: hf)"
    )

    # 카테고리 선택
    all_cats = [dt.value for dt in DocType]
    parser.add_argument(
        "--categories", type=str,
        default="general,legal,business,medical",
        help=f"실행할 카테고리 (쉼표 구분, 기본: general,legal,business,medical). 가능: {','.join(all_cats)}"
    )

    # docs 모드 전용
    parser.add_argument("--docs_dir", type=str, default=None, help="사용자 문서 디렉토리 (--mode docs 필수)")
    parser.add_argument("--num_qa", type=int, default=10, help="카테고리당 LLM 생성 QA 수 (docs 모드, 기본: 10)")

    # HF 모드 전용
    parser.add_argument("--max_corpus", type=int, default=10_000, help="HF 코퍼스 샘플링 크기 (기본: 10000)")
    parser.add_argument("--max_queries", type=int, default=100, help="HF 쿼리 샘플링 크기 (기본: 100)")
    parser.add_argument("--no_hf_cache", action="store_true", help="HF 데이터셋 캐시 무시 (재다운로드)")

    # 벤치마크 공통
    parser.add_argument("--k", type=int, default=3, help="검색 결과 수 (기본: 3)")
    parser.add_argument("--pass1_only", action="store_true", help="레이턴시만 측정 (RAGAS 없음)")
    parser.add_argument("--no_ragas", action="store_true", help="RAGAS 평가 건너뜀")
    parser.add_argument("--reindex", action="store_true", help="기존 인덱스 무시하고 재인덱싱")

    # 모델 설정
    parser.add_argument("--contextual_llm", type=str, default="gpt-4o-mini", help="Contextual Retrieval LLM (기본: gpt-4o-mini)")
    parser.add_argument("--colbert_model", type=str, default="jinaai/jina-colbert-v2", help="ColBERT 모델")

    # 출력
    parser.add_argument("--output_dir", type=str, default=None, help="결과 저장 디렉토리 (기본: _benchdata/service_run/)")
    parser.add_argument("--dry_run", action="store_true", help="조합 목록만 출력 (실행 없음)")

    return parser.parse_args()


def main():
    setup_ssl_bypass()
    args = parse_args()

    # --- 카테고리 파싱
    cat_names = [c.strip() for c in args.categories.split(",") if c.strip()]
    try:
        categories = [DocType(c) for c in cat_names]
    except ValueError as e:
        valid = [dt.value for dt in DocType]
        print(f"오류: 유효하지 않은 카테고리 — {e}\n허용 값: {valid}")
        sys.exit(1)

    # --- docs 모드 검증
    if args.mode == "docs":
        if not args.docs_dir:
            print("오류: --mode docs 일 때 --docs_dir 을 지정해야 합니다.")
            sys.exit(1)
        docs_dir = Path(args.docs_dir)
        if not docs_dir.is_dir():
            print(f"오류: docs_dir 가 존재하지 않습니다: {docs_dir}")
            sys.exit(1)
        # docs 모드에서 TECHNICAL 포함 가능
    else:
        # hf 모드에서 TECHNICAL은 제외
        if DocType.TECHNICAL in categories:
            print("[경고] TECHNICAL 카테고리는 HF 데이터셋이 없어 hf 모드에서 제외됩니다.")
            categories = [c for c in categories if c != DocType.TECHNICAL]
        if not categories:
            print("실행할 카테고리가 없습니다.")
            sys.exit(0)

    # --- 출력 디렉토리
    run_dir = Path(args.output_dir) if args.output_dir else SERVICE_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- service 프리셋 (ColBERT + Contextual 고정)
    combos = generate_valid_combinations(PRESETS["service"])
    print(f"\n{'═' * 60}")
    print(" 서비스 벤치마크 시작")
    print(f"{'═' * 60}")
    print(f"  모드     : {args.mode}")
    print(f"  카테고리 : {[c.value for c in categories]}")
    print(f"  조합 수  : {len(combos)}개 (service 프리셋)")
    print(f"  출력 경로: {run_dir}")
    print()
    for i, c in enumerate(combos, 1):
        print(f"  조합 {i:2d}: {c.label}")

    if args.dry_run:
        print("\n[dry-run] 실제 실행 없이 종료합니다.")
        return

    # --- 전역 캐시 (ColBERT 싱글톤 공유)
    cfg = CacheConfig(
        colbert_model=args.colbert_model,
        colbert_device="cpu",
        contextual_llm=args.contextual_llm,
    )
    global_cache = IndexCacheManager(cfg)

    # --- 체크포인트
    checkpoint = CheckpointManager(run_dir)

    # --- 카테고리별 실행
    all_results: Dict[str, Dict] = {}
    for doc_type in categories:
        cat_name = doc_type.value
        print(f"\n{'═' * 60}")
        print(f" [{cat_name.upper()}] 카테고리 시작")
        print(f"{'═' * 60}")

        if checkpoint.is_completed(cat_name) and not args.reindex:
            print(f"  [체크포인트] {cat_name} 이미 완료됨 — 건너뜀")
            continue

        checkpoint.mark_started(cat_name, {"mode": args.mode, "n_combos": len(combos)})
        cat_dir = run_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 데이터 준비
            if args.mode == "hf":
                parent_pairs, child_chunks, qa_pairs = _prepare_hf_data(
                    doc_type=doc_type,
                    cat_dir=cat_dir,
                    max_corpus=args.max_corpus,
                    max_queries=args.max_queries,
                    use_cache=not args.no_hf_cache,
                )
            else:
                parent_pairs, child_chunks, qa_pairs = _prepare_docs_data(
                    doc_type=doc_type,
                    docs_dir=docs_dir,
                    cat_dir=cat_dir,
                    num_qa=args.num_qa,
                    llm_model=args.contextual_llm,
                )

            print(f"\n  데이터 준비 완료: {len(child_chunks):,} children / {len(qa_pairs):,} QA")

            # 벤치마크 실행
            t0 = time.time()
            latency_df, ragas_df = _run_category_bench(
                doc_type=doc_type,
                parent_pairs=parent_pairs,
                child_chunks=child_chunks,
                qa_pairs=qa_pairs,
                index_cache=global_cache,
                combos=combos,
                run_dir=run_dir,
                args=args,
            )
            elapsed = time.time() - t0
            print(f"\n  [{cat_name}] 완료 ({elapsed:.0f}s)")

            # 결과 저장
            result_path = _save_category_result(cat_name, run_dir, latency_df, ragas_df, qa_pairs)
            checkpoint.mark_completed(cat_name, result_path)

            all_results[cat_name] = {
                "latency_df": latency_df,
                "ragas_df": ragas_df,
                "n_qa": len(qa_pairs),
            }

        except Exception as e:
            print(f"\n  [{cat_name}] 오류 발생: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"  [{cat_name}] 건너뛰고 다음 카테고리 진행...")

        finally:
            _release_memory()

    # --- 최종 요약
    print(f"\n{'═' * 60}")
    print(" 서비스 벤치마크 완료")
    print(f"{'═' * 60}")
    summary = checkpoint.summary()
    print(f"  완료된 카테고리: {summary['completed']}")
    print(f"  결과 디렉토리  : {run_dir}")

    if all_results:
        print("\n  카테고리별 결과 요약:")
        for cat, res in all_results.items():
            n_qa = res["n_qa"]
            ragas_note = ""
            if res["ragas_df"] is not None:
                cols = [c for c in res["ragas_df"].columns if c != "strategy"]
                ragas_note = f" | RAGAS {len(cols)}개 지표"
            print(f"    {cat:12s}: {n_qa}개 QA{ragas_note}")

    print(f"\n  다음 단계: python -m rag_bench.analysis.reporter --run_dir {run_dir}")


if __name__ == "__main__":
    main()
