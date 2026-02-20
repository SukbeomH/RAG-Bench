"""
전체 조합 벤치마크 — 3-Layer 교차 조합 + 2-Pass 실행.

3-Layer 설계:
  Layer 1: Dense Model   (kosimcse, e5, bge-m3, minilm)
  Layer 2: Sparse Model  (korean_bm25, splade, fastembed_bm25)
  Layer 3: Retrieval Mode (hybrid × reranker × llm_support = 6종)

총 유효 조합: 4 × 3 × 6 = 72개

레거시 모드:
  --combos / --skip_* 플래그 사용 시 기존 방식으로 동작.

새 모드:
  --preset quick|standard|full  프리셋 기반 조합 생성
  --pass1-only                  레이턴시만 측정 (RAGAS 없음)
  --top_n N                     Pass 1 후 상위 N만 RAGAS
  --dry-run                     조합 목록만 출력
  --layers                      레이어별 기여도 분석

Usage:
    # 레거시 모드
    python -m rag_bench.scripts.run_all_combos [--k 3] [--combos 1,3,4] [--skip_colbert] [--skip_rerank] [--skip_contextual] [--skip_flashrank] [--no_ragas] [--reindex] [--contextual_base 3]

    # 새 모드
    python -m rag_bench.scripts.run_all_combos --preset full --dry-run
    python -m rag_bench.scripts.run_all_combos --preset quick --pass1-only
    python -m rag_bench.scripts.run_all_combos --preset standard --top_n 10
"""

import argparse
import gc
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_bench.config import (
    BENCH_DATA_DIR,
    BENCH_DOCS_DIR,
    setup_ssl_bypass,
)
from rag_bench.indexing.chunker import create_parent_child_chunks
from rag_bench.run_tracker import RunTracker, track_openai_tokens
from rag_bench.runner import BenchmarkRunner
from rag_bench.strategies.dense_sparse import DENSE_MODELS, SPARSE_TYPES

ALL_COMBO_IDS = [1, 2, 3, 4]


# ===========================================================================
# ComboSpec + 조합 생성기
# ===========================================================================


@dataclass
class ComboSpec:
    """3-Layer 조합 명세."""

    dense: str = ""                   # DENSE_MODELS 키 (예: "kosimcse")
    sparse: str = ""                  # SPARSE_TYPES 값 (예: "splade")
    reranker: Optional[str] = None    # None | "colbert" | "flashrank"
    llm_support: Optional[str] = None # None | "contextual"

    @property
    def label(self) -> str:
        parts = [self.dense, self.sparse]
        if self.reranker:
            parts.append(self.reranker)
        if self.llm_support:
            parts.append(self.llm_support)
        return "+".join(parts)

    @property
    def retrieval_mode(self) -> str:
        mode = "hybrid"
        suffixes = []
        if self.reranker:
            suffixes.append(self.reranker + "_rerank")
        if self.llm_support:
            suffixes.append("llm_support")
        if suffixes:
            mode += "_with_" + "_and_".join(suffixes)
        return mode

    @property
    def index_key(self) -> str:
        """인덱스 캐싱 키. (dense, sparse) 쌍으로 결정."""
        return f"{self.dense}:{self.sparse}"


# ---------------------------------------------------------------------------
# 프리셋
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, list]] = {
    "quick": {
        "dense_models": ["bge-m3", "minilm"],
        "sparse_models": ["fastembed_bm25"],
        "rerankers": [None, "flashrank"],
        "llm_support": [None],
    },
    "standard": {
        "dense_models": list(DENSE_MODELS.keys()),
        "sparse_models": list(SPARSE_TYPES),
        "rerankers": [None, "flashrank"],
        "llm_support": [None],
    },
    "full": {
        "dense_models": list(DENSE_MODELS.keys()),
        "sparse_models": list(SPARSE_TYPES),
        "rerankers": [None, "colbert", "flashrank"],
        "llm_support": [None, "contextual"],
    },
}


def generate_valid_combinations(config: Dict[str, list]) -> List[ComboSpec]:
    """3-Layer 카테시안 곱으로 유효 조합 생성.

    Args:
        config: PRESETS 딕셔너리 항목.
    """
    combos = []
    for d in config["dense_models"]:
        for s in config["sparse_models"]:
            for r in config["rerankers"]:
                for llm_sup in config["llm_support"]:
                    combos.append(ComboSpec(dense=d, sparse=s, reranker=r, llm_support=llm_sup))
    return combos


# ===========================================================================
# CacheConfig — IndexCacheManager 하드코딩 설정 외부화
# ===========================================================================


@dataclass
class CacheConfig:
    """IndexCacheManager에서 사용하는 모델/리소스 설정."""

    colbert_model: str = "jinaai/jina-colbert-v2"
    colbert_device: str = "cpu"
    flashrank_model: str = "ms-marco-MultiBERT-L-12"
    flashrank_max_length: int = 512
    contextual_llm: str = "gpt-4o-mini"
    rerank_n: int = 20


# ===========================================================================
# IndexCacheManager
# ===========================================================================


@dataclass
class IndexCacheManager:
    """동일 (dense, sparse) 쌍은 같은 Qdrant 인덱스를 재사용."""

    config: CacheConfig = field(default_factory=CacheConfig)
    cache: Dict[str, Tuple[Any, str]] = field(default_factory=dict)
    ctx_cache: Dict[str, Any] = field(default_factory=dict)  # contextual 전략 캐시
    _colbert_model: Any = field(default=None, repr=False)     # ColBERT 싱글톤
    _flashrank_ranker: Any = field(default=None, repr=False)  # FlashRank 싱글톤

    def get_colbert_model(self):
        """ColBERT 모델을 1회만 로드하고 이후 공유."""
        if self._colbert_model is not None:
            return self._colbert_model
        from pylate import models
        print(f"[ColBERT 캐시] 모델 최초 로드 중 (device={self.config.colbert_device})...")
        self._colbert_model = models.ColBERT(
            model_name_or_path=self.config.colbert_model,
            device=self.config.colbert_device,
            trust_remote_code=True,
        )
        print("[ColBERT 캐시] 모델 로드 완료.")
        return self._colbert_model

    def get_flashrank_ranker(self):
        """FlashRank Ranker를 1회만 로드하고 이후 공유."""
        if self._flashrank_ranker is not None:
            return self._flashrank_ranker
        from flashrank import Ranker
        print("[FlashRank 캐시] Ranker 최초 로드 중 (이후 공유)...")
        self._flashrank_ranker = Ranker(
            model_name=self.config.flashrank_model,
            max_length=self.config.flashrank_max_length,
        )
        print("[FlashRank 캐시] Ranker 로드 완료.")
        return self._flashrank_ranker

    def get_or_build(self, spec: ComboSpec, child_chunks, reindex=False):
        """base DenseSparseStrategy를 캐시에서 가져오거나 새로 빌드.

        디스크에 기존 Qdrant 인덱스가 있고 reindex=False면 재인덱싱 없이 연결만 한다.
        """
        from rag_bench.strategies.dense_sparse import DenseSparseStrategy

        key = spec.index_key
        qdrant_path = str(BENCH_DATA_DIR / f"qdrant_db_{spec.dense}_{spec.sparse}")

        if key in self.cache and not reindex:
            cached_strategy, _ = self.cache[key]
            return cached_strategy

        strategy = DenseSparseStrategy(
            dense_model=spec.dense, sparse_type=spec.sparse, qdrant_path=qdrant_path
        )

        qdrant_dir = Path(qdrant_path)
        index_exists = qdrant_dir.exists() and any(qdrant_dir.iterdir())

        if index_exists and not reindex:
            print(f"  [기존 인덱스 재사용] {spec.dense}+{spec.sparse} — {qdrant_path}")
            strategy._ensure_initialized()
            # BM25 어휘는 디스크에 영속화되지 않으므로 동일 문서로 재fit
            if hasattr(strategy._sparse_embeddings, "fit"):
                texts = [doc.page_content for doc in child_chunks]
                strategy._sparse_embeddings.fit(texts)
            strategy._is_ready = True
        else:
            strategy.index(child_chunks)

        self.cache[key] = (strategy, qdrant_path)
        return strategy

    def get_or_build_contextual(self, spec: ComboSpec, child_chunks, parent_pairs, reindex=False):
        """contextual 전략을 캐시에서 가져오거나 새로 빌드.

        디스크에 기존 Contextual Qdrant 인덱스가 있고 reindex=False면
        LLM 문맥 생성 및 재인덱싱 없이 연결만 한다.
        """
        from rag_bench.strategies.contextual_retrieval import ContextualRetrievalStrategy
        from rag_bench.strategies.dense_sparse import DenseSparseStrategy

        key = f"ctx:{spec.index_key}"

        if key in self.ctx_cache and not reindex:
            return self.ctx_cache[key]

        ctx_qdrant_path = str(
            BENCH_DATA_DIR / f"qdrant_db_ctx_{spec.dense}_{spec.sparse}"
        )
        ctx_base = DenseSparseStrategy(
            dense_model=spec.dense, sparse_type=spec.sparse, qdrant_path=ctx_qdrant_path
        )

        # 이미 캐시된 base 전략에서 Dense/Sparse 모델 객체 공유 (재로드 방지)
        base_key = spec.index_key
        if base_key in self.cache:
            cached_base, _ = self.cache[base_key]
            if cached_base._dense_embeddings is not None:
                ctx_base.share_embeddings(
                    dense_embeddings=cached_base._dense_embeddings,
                    sparse_embeddings=cached_base._sparse_embeddings,
                    embedding_dim=cached_base._embedding_dim,
                    use_langchain_sparse=cached_base._use_langchain_sparse,
                )

        strategy = ContextualRetrievalStrategy(
            base_strategy=ctx_base,
            parent_pairs=parent_pairs,
            llm_model=self.config.contextual_llm,
        )

        ctx_qdrant_dir = Path(ctx_qdrant_path)
        ctx_index_exists = ctx_qdrant_dir.exists() and any(ctx_qdrant_dir.iterdir())

        if ctx_index_exists and not reindex:
            print(f"  [기존 Contextual 인덱스 재사용] {spec.dense}+{spec.sparse} — {ctx_qdrant_path}")
            ctx_base._ensure_initialized()
            if hasattr(ctx_base._sparse_embeddings, "fit"):
                texts = [doc.page_content for doc in child_chunks]
                ctx_base._sparse_embeddings.fit(texts)
            ctx_base._is_ready = True
            strategy._is_ready = True
        else:
            strategy.index(child_chunks)

        self.ctx_cache[key] = strategy
        return strategy


# ===========================================================================
# 전략 빌드 로직
# ===========================================================================


def build_strategy_from_spec(
    spec: ComboSpec,
    index_cache: IndexCacheManager,
    child_chunks,
    parent_pairs,
    reindex: bool = False,
):
    """ComboSpec에서 전략 인스턴스 생성."""
    # 1. Base: DenseSparse (인덱스 캐시 활용)
    base = index_cache.get_or_build(spec, child_chunks, reindex=reindex)

    # 2. LLM Support 적용 (contextual 캐시 활용)
    if spec.llm_support == "contextual":
        base = index_cache.get_or_build_contextual(spec, child_chunks, parent_pairs, reindex)

    # 3. Reranker 적용 (Decorator)
    cfg = index_cache.config
    if spec.reranker == "colbert":
        from rag_bench.strategies.colbert_rerank import ColBERTRerankStrategy

        shared = index_cache.get_colbert_model()
        return ColBERTRerankStrategy(
            base_strategy=base,
            model_name=cfg.colbert_model,
            rerank_n=cfg.rerank_n,
            device=cfg.colbert_device,
            shared_model=shared,
        )
    elif spec.reranker == "flashrank":
        from rag_bench.strategies.flashrank_rerank import FlashRankRerankStrategy

        shared = index_cache.get_flashrank_ranker()
        return FlashRankRerankStrategy(
            base_strategy=base,
            model_name=cfg.flashrank_model,
            rerank_n=cfg.rerank_n,
            max_length=cfg.flashrank_max_length,
            shared_ranker=shared,
        )

    return base


# ===========================================================================
# 레거시 빌드 함수들 (기존 --combos / --skip_* 모드)
# ===========================================================================


def _load_qa_dataset() -> dict:
    qa_path = BENCH_DATA_DIR / "qa_dataset.json"
    if not qa_path.exists():
        print(f"Error: QA 데이터셋이 없습니다: {qa_path}")
        print("  먼저 실행: python -m rag_bench.scripts.generate_qa")
        sys.exit(1)
    dataset = json.loads(qa_path.read_text(encoding="utf-8"))
    print(f"QA 데이터셋 로드: {dataset['num_qa']}개 QA")
    return dataset


def _try_build_dense_sparse(combo_id: int, child_chunks, qdrant_suffix: str, reindex=True):
    from rag_bench.strategies.dense_sparse import DenseSparseStrategy

    qdrant_path = str(BENCH_DATA_DIR / f"qdrant_db_{qdrant_suffix}")
    strategy = DenseSparseStrategy(combo_id=combo_id, qdrant_path=qdrant_path)

    if reindex:
        print("  [재인덱싱] 임베딩 모델 로드 + Qdrant 인덱싱...")
        strategy.index(child_chunks)
    else:
        qdrant_dir = Path(qdrant_path)
        if not qdrant_dir.exists() or not any(qdrant_dir.iterdir()):
            print("  [기존 인덱스 없음] 재인덱싱으로 자동 전환")
            strategy.index(child_chunks)
        else:
            print("  [기존 로드] 임베딩 모델 로드 중...")
            strategy._ensure_initialized()
            print("  [기존 로드] Qdrant 컬렉션 연결 완료")
            from rag_bench.strategies.dense_sparse import KoreanBM25Encoder
            if isinstance(strategy._sparse_embeddings, KoreanBM25Encoder):
                print("  [기존 로드] BM25 어휘 구축 중...")
                texts = [doc.page_content for doc in child_chunks]
                strategy._sparse_embeddings.fit(texts)
            strategy._is_ready = True
            print("  [기존 로드] 준비 완료")
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
    strategy._base_strategy = base_strategy
    strategy._ensure_initialized()
    strategy._is_ready = True
    return strategy, None


def _try_build_contextual(
    base_combo_id: int, child_chunks, parent_pairs, qdrant_suffix: str, reindex=True
):
    from rag_bench.strategies.dense_sparse import DenseSparseStrategy
    from rag_bench.strategies.contextual_retrieval import ContextualRetrievalStrategy

    qdrant_path = str(BENCH_DATA_DIR / f"qdrant_db_{qdrant_suffix}")
    base = DenseSparseStrategy(combo_id=base_combo_id, qdrant_path=qdrant_path)
    strategy = ContextualRetrievalStrategy(
        base_strategy=base,
        parent_pairs=parent_pairs,
        llm_model="gpt-4o-mini",
    )
    strategy.index(child_chunks)
    return strategy, None


def _try_build_flashrank_rerank(base_strategy, child_chunks):
    from rag_bench.strategies.flashrank_rerank import FlashRankRerankStrategy

    strategy = FlashRankRerankStrategy(
        base_strategy=base_strategy,
        model_name="ms-marco-MultiBERT-L-12",
        rerank_n=20,
    )
    strategy._base_strategy = base_strategy
    strategy._ensure_initialized()
    strategy._is_ready = True
    return strategy, None



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


def _print_ragas_table(scores_df, scoring_profile="balanced"):
    if scores_df is None or scores_df.empty:
        print("RAGAS 평가 결과가 없습니다.")
        return

    from rag_bench.evaluation.evaluator import SCORING_PROFILES

    print(f"\n{'═' * 100}")
    print(f" RAGAS 평가 결과 비교 (scoring: {scoring_profile})")
    print(f"{'═' * 100}")

    metric_cols = [c for c in scores_df.columns if c != "strategy"]

    # 가중 점수 계산
    weights = SCORING_PROFILES.get(scoring_profile, SCORING_PROFILES["balanced"])
    weighted_scores = []
    for _, row in scores_df.iterrows():
        ws = 0.0
        for metric, weight in weights.items():
            val = row.get(metric, 0.0)
            if isinstance(val, (int, float)):
                ws += val * weight
        weighted_scores.append(round(ws, 4))

    display_cols = metric_cols + ["weighted"]
    header = f"  {'전략':<45}"
    for col in display_cols:
        header += f" {col:>14}"
    print(header)
    print(f"  {'─' * 45} " + " ".join("─" * 14 for _ in display_cols))

    for i, (_, row) in enumerate(scores_df.iterrows()):
        line = f"  {row['strategy']:<45}"
        for col in metric_cols:
            val = row.get(col, "N/A")
            if isinstance(val, float):
                line += f" {val:>14.4f}"
            else:
                line += f" {str(val):>14}"
        line += f" {weighted_scores[i]:>14.4f}"
        print(line)


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
# 새 모드: 3-Layer 조합 실행
# ===========================================================================


def _run_preset_mode(args):
    """--preset 기반 새 3-Layer 조합 실행."""
    setup_ssl_bypass()

    preset_name = args.preset
    if preset_name not in PRESETS:
        print(f"Error: 알 수 없는 프리셋: {preset_name}")
        print(f"  사용 가능: {list(PRESETS.keys())}")
        sys.exit(1)

    config = PRESETS[preset_name]
    combos = generate_valid_combinations(config)


    print(f"\n{'═' * 60}")
    print(f" 3-Layer 조합 벤치마크 — 프리셋: {preset_name}")
    print(f"{'═' * 60}")
    print(f"  Dense Models: {config['dense_models']}")
    print(f"  Sparse Models: {config['sparse_models']}")
    print(f"  Rerankers: {config['rerankers']}")
    print(f"  LLM Support: {config['llm_support']}")
    print(f"  총 조합: {len(combos)}개")

    # --dry-run: 조합 목록만 출력
    if args.dry_run:
        print(f"\n{'─' * 80}")
        print(f" {'#':>3}  {'Label':<40} {'Retrieval Mode':<50}")
        print(f"{'─' * 80}")
        for i, spec in enumerate(combos, 1):
            print(f" {i:>3}  {spec.label:<40} {spec.retrieval_mode:<50}")
        print(f"{'─' * 80}")
        print(f" 합계: {len(combos)}개 유효 조합")

        # 인덱스 키 요약
        unique_keys = set(spec.index_key for spec in combos)
        print(f" 고유 인덱스: {len(unique_keys)}개 (실제 인덱싱 횟수)")
        return

    # --layers 분석 (dry-run 모드에서만)
    if args.layers and args.dry_run:
        _print_layer_analysis_preview(combos, config)
        return

    # ── RunTracker 초기화 ──
    tracker = RunTracker(output_dir=BENCH_DATA_DIR)

    # ── Step 1: QA 로드 ──
    print(f"\n{'=' * 60}")
    print("Step 1: QA 데이터셋 로드")
    print(f"{'=' * 60}")
    with tracker.phase("qa_dataset_load"):
        dataset = _load_qa_dataset()
        qa_pairs = dataset["qa_pairs"]
        queries = [qa["question"] for qa in qa_pairs]
        ground_truths = [qa["ground_truth"] for qa in qa_pairs]

    # ── Step 2: 문서 청킹 ──
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

    # 트래커에 설정 기록
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

    # ── Step 3: 전략 생성 (인덱스 캐싱) ──
    print(f"\n{'=' * 60}")
    print("Step 3: 전략 생성 및 인덱싱")
    print(f"{'=' * 60}")

    index_cache = IndexCacheManager()
    strategies: List[Tuple[ComboSpec, Any]] = []  # (spec, strategy)
    build_results: List[Tuple[str, object, Optional[str]]] = []

    reindex = args.reindex

    with tracker.phase("strategy_build_and_indexing"):
        for i, spec in enumerate(combos, 1):
            progress = f"[{i}/{len(combos)}]"
            label = spec.label

            strategy, err = _safe_build(
                label,
                lambda s=spec: (
                    build_strategy_from_spec(s, index_cache, child_chunks, parent_pairs, reindex),
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

    # ── Step 4: Pass 1 — 레이턴시 측정 ──
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

    # 레이턴시 결과 저장
    latency_df = runner.to_dataframe()
    summary_df = None
    if latency_df is not None:
        latency_path = BENCH_DATA_DIR / "all_combos_latency.csv"
        latency_df.to_csv(latency_path, index=False, encoding="utf-8-sig")
        print(f"  레이턴시 결과: {latency_path}")
        # 전략별 요약 DataFrame (avg_latency 등)
        summary_df = _build_latency_summary(latency_df)

        # 트래커에 쿼리 레이턴시 통계 기록
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

    # 레이어별 기여도 분석 (레이턴시 기반)
    if args.layers and summary_df is not None:
        _print_layer_contribution(strategies, summary_df)

    # --pass1-only: 여기서 종료
    if args.pass1_only:
        print(f"\n{'═' * 60}")
        print(f" Pass 1 완료 — {len(active_strategies)}개 전략 레이턴시 측정")
        print(f"{'═' * 60}")
        tracker.finalize()
        _cleanup_strategies(active_strategies)
        return

    # ── Step 5: Pass 2 — RAGAS 평가 (상위 N 또는 전체) ──
    top_n = args.top_n or len(strategies)
    if top_n < len(strategies):
        # 레이턴시 기준 상위 N 선별
        print(f"\n{'=' * 60}")
        print(f"Step 5: Pass 2 — 상위 {top_n}개 RAGAS 평가")
        print(f"{'=' * 60}")

        # 평균 레이턴시로 정렬
        if summary_df is not None and "avg_latency" in summary_df.columns:
            strategy_latencies = []
            for spec, strat in strategies:
                mask = summary_df["strategy"] == strat.name
                if mask.any():
                    avg_lat = summary_df.loc[mask, "avg_latency"].values[0]
                else:
                    avg_lat = float("inf")
                strategy_latencies.append((spec, strat, avg_lat))
            strategy_latencies.sort(key=lambda x: x[2])
            eval_strategies = [(sp, st) for sp, st, _ in strategy_latencies[:top_n]]
        else:
            eval_strategies = strategies[:top_n]
    else:
        print(f"\n{'=' * 60}")
        print(f"Step 5: Pass 2 — 전체 {len(strategies)}개 RAGAS 평가")
        print(f"{'=' * 60}")
        eval_strategies = strategies

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

    if evaluator is not None:
        eval_runner = BenchmarkRunner(
            strategies=[s for _, s in eval_strategies],
            queries=queries,
            k=args.k,
            evaluator=evaluator,
        )
        # Pass 1 결과 재사용 (재검색 방지)
        eval_runner.inject_results(runner._results)
        with tracker.phase("pass2_ragas"):
            with track_openai_tokens() as ragas_tokens:
                scores_df = eval_runner.evaluate(ground_truths=ground_truths)
        if ragas_tokens.total_tokens > 0:
            tracker.record_ragas_tokens(ragas_tokens)
        _print_ragas_table(scores_df, scoring_profile=args.scoring_profile)

        # per-sample CSV 저장 (ExtendedRAGEvaluator 사용 시)
        if eval_runner.reports:
            per_sample_dir = BENCH_DATA_DIR / "per_sample"
            per_sample_dir.mkdir(parents=True, exist_ok=True)
            for strat_name, report in eval_runner.reports.items():
                if not report.per_sample_df.empty:
                    safe_name = strat_name.replace("/", "_").replace(" ", "_")
                    csv_path = per_sample_dir / f"{safe_name}.csv"
                    report.per_sample_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"  per-sample 결과: {per_sample_dir}/")

        if scores_df is not None:
            scores_path = BENCH_DATA_DIR / "all_combos_ragas.csv"
            scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")
            print(f"  RAGAS 점수: {scores_path}")

            # 트래커에 RAGAS 점수 기록
            for _, row in scores_df.iterrows():
                strat_name = row["strategy"]
                # eval_strategies에서 매칭되는 spec 찾기
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

        # 레이어별 기여도 (RAGAS 기반)
        if args.layers and scores_df is not None:
            _print_layer_contribution_ragas(eval_strategies, scores_df)

    # ── Step 6: 리포트 생성 ──
    if summary_df is not None:
        _generate_report(summary_df, scores_df if evaluator else None, combos, BENCH_DATA_DIR, tracker=tracker)

    # ── 수행 이력 저장 ──
    tracker.finalize()

    # ── Cleanup ──
    _cleanup_strategies(active_strategies)

    print(f"\n{'═' * 60}")
    print(f" 벤치마크 완료 — {len(active_strategies)}개 전략 비교")
    print(f"{'═' * 60}")


# ===========================================================================
# 레이어 기여도 분석
# ===========================================================================


def _print_layer_analysis_preview(combos: List[ComboSpec], config: dict):
    """dry-run 시 레이어 분석 미리보기."""
    print(f"\n{'═' * 60}")
    print(" 레이어별 조합 분포")
    print(f"{'═' * 60}")

    for layer_name, values in [
        ("Dense Model", config["dense_models"]),
        ("Sparse Model", config["sparse_models"]),
        ("Reranker", config["rerankers"]),
        ("LLM Support", config["llm_support"]),
    ]:
        print(f"\n  {layer_name}:")
        for val in values:
            count = 0
            if layer_name == "Dense Model":
                count = sum(1 for c in combos if c.dense == val)
            elif layer_name == "Sparse Model":
                count = sum(1 for c in combos if c.sparse == val)
            elif layer_name == "Reranker":
                count = sum(1 for c in combos if c.reranker == val)
            elif layer_name == "LLM Support":
                count = sum(1 for c in combos if c.llm_support == val)
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


def _print_layer_contribution(strategies: List[Tuple[ComboSpec, Any]], summary_df):
    """레이턴시 기반 레이어 기여도 출력."""
    print(f"\n{'═' * 60}")
    print(" 레이어별 평균 레이턴시 기여도")
    print(f"{'═' * 60}")

    if summary_df is None or summary_df.empty:
        print("  (레이턴시 데이터 없음)")
        return

    # 전략명 → 평균 레이턴시(s) 매핑
    lat_map = {}
    for _, row in summary_df.iterrows():
        lat_map[row["strategy"]] = row["avg_latency"]

    # 레이어별 분석
    for layer_name, get_val in [
        ("Dense Model", lambda s: s.dense),
        ("Sparse Model", lambda s: s.sparse),
        ("Reranker", lambda s: s.reranker or "none"),
        ("LLM Support", lambda s: s.llm_support or "none"),
    ]:
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

    for layer_name, get_val in [
        ("Dense Model", lambda s: s.dense),
        ("Sparse Model", lambda s: s.sparse),
        ("Reranker", lambda s: s.reranker or "none"),
        ("LLM Support", lambda s: s.llm_support or "none"),
    ]:
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


def _generate_report(latency_summary_df, ragas_df, combo_specs, output_dir, tracker=None):
    """Markdown 리포트 생성. latency_summary_df는 전략별 요약 DataFrame."""
    report_path = output_dir / "e2e_report.md"

    lines = [
        "# E2E 3-Layer 조합 벤치마크 리포트",
        "",
        f"**조합 수**: {len(combo_specs)}개",
        f"**생성 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # 수행 이력 요약 (플랫폼, 시간, 토큰)
    if tracker and hasattr(tracker, '_record'):
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

    lines.extend([
        "---",
        "",
        "## 레이턴시 결과 (Top 10)",
        "",
    ])

    if latency_summary_df is not None and "strategy" in latency_summary_df.columns:
        if "avg_latency" in latency_summary_df.columns:
            sorted_df = latency_summary_df.sort_values("avg_latency")
            lines.append("| # | 전략 | 평균 레이턴시 |")
            lines.append("|---|------|:----------:|")
            for i, (_, row) in enumerate(sorted_df.head(10).iterrows(), 1):
                lines.append(f"| {i} | {row['strategy']} | {row['avg_latency']:.3f}s |")
        lines.append("")

    if ragas_df is not None and not ragas_df.empty:
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

        # 가중 점수 테이블 (모든 프로파일)
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
                ws = 0.0
                for metric, weight in weights.items():
                    val = row.get(metric, 0.0)
                    if isinstance(val, (int, float)):
                        ws += val * weight
                vals.append(f"{ws:.4f}")
            lines.append(f"| {row['strategy']} | " + " | ".join(vals) + " |")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  리포트: {report_path}")


# ===========================================================================
# 공통 유틸리티
# ===========================================================================


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
# 레거시 모드 (기존 --combos / --skip_* 방식)
# ===========================================================================


def _run_legacy_mode(args):
    """기존 방식 실행."""
    setup_ssl_bypass()

    if args.combos:
        combo_ids = [int(x.strip()) for x in args.combos.split(",")]
    else:
        combo_ids = list(ALL_COMBO_IDS)

    reindex = args.reindex

    print(f"대상 DenseSparse 조합: {combo_ids}")
    print(f"ColBERT: {'OFF' if args.skip_colbert else 'ON'}")
    print(f"ColBERTRerank: {'OFF' if args.skip_rerank else 'ON'}")
    print(f"Contextual Retrieval: {'OFF' if args.skip_contextual else f'ON (base=combo{args.contextual_base})'}")
    print(f"FlashRank Rerank: {'OFF' if args.skip_flashrank else 'ON'}")
    print(f"RAGAS 평가: {'OFF' if args.no_ragas else 'ON'}")
    print(f"인덱스 모드: {'재인덱싱' if reindex else '기존 재사용'}")

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

    build_results: List[Tuple[str, object, Optional[str]]] = []

    total_strategies = (
        len(combo_ids)
        + (0 if args.skip_colbert else 1)
        + (0 if args.skip_rerank else len(combo_ids))
        + (0 if args.skip_contextual else 1)
        + (0 if args.skip_flashrank else len(combo_ids))
    )
    current = 0

    # 3-a. DenseSparse
    ds_strategies = {}
    for cid in combo_ids:
        current += 1
        label = f"DenseSparse combo={cid}"
        strategy, err = _safe_build(
            label, _try_build_dense_sparse, cid, child_chunks, f"combo{cid}", reindex,
            progress=f"[{current}/{total_strategies}]",
        )
        build_results.append((label, strategy, err))
        if strategy is not None:
            ds_strategies[cid] = strategy

    # 3-b. ColBERT
    colbert_strategy = None
    if not args.skip_colbert:
        current += 1
        label = "ColBERT (jina-colbert-v2, brute-force)"
        strategy, err = _safe_build(
            label, _try_build_colbert, child_chunks,
            progress=f"[{current}/{total_strategies}]",
        )
        build_results.append((label, strategy, err))
        colbert_strategy = strategy

    # 3-c. ColBERTRerank
    rerank_strategies = {}
    if not args.skip_rerank:
        for cid, ds in ds_strategies.items():
            current += 1
            label = f"ColBERTRerank (base=combo{cid})"
            strategy, err = _safe_build(
                label, _try_build_rerank, ds, child_chunks,
                progress=f"[{current}/{total_strategies}]",
            )
            build_results.append((label, strategy, err))
            if strategy is not None:
                rerank_strategies[cid] = strategy

    # 3-d. Contextual Retrieval
    contextual_strategy = None
    if not args.skip_contextual:
        current += 1
        ctx_base_id = args.contextual_base
        label = f"Contextual Retrieval (base=combo{ctx_base_id})"
        strategy, err = _safe_build(
            label, _try_build_contextual, ctx_base_id, child_chunks, parent_pairs,
            f"contextual_combo{ctx_base_id}", reindex,
            progress=f"[{current}/{total_strategies}]",
        )
        build_results.append((label, strategy, err))
        contextual_strategy = strategy

    # 3-e. FlashRank Rerank
    flashrank_strategies = {}
    if not args.skip_flashrank:
        for cid, ds in ds_strategies.items():
            current += 1
            label = f"FlashRank Rerank (base=combo{cid})"
            strategy, err = _safe_build(
                label, _try_build_flashrank_rerank, ds, child_chunks,
                progress=f"[{current}/{total_strategies}]",
            )
            build_results.append((label, strategy, err))
            if strategy is not None:
                flashrank_strategies[cid] = strategy

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
    if contextual_strategy is not None:
        active_strategies.append(contextual_strategy)
    for cid in combo_ids:
        if cid in flashrank_strategies:
            active_strategies.append(flashrank_strategies[cid])

    if not active_strategies:
        print("\n성공한 전략이 없습니다. 종료합니다.")
        sys.exit(1)

    print(f"\n벤치마크 대상 전략: {len(active_strategies)}개")

    # ── Step 4: 벤치마크 실행 ──
    print(f"\n{'=' * 60}")
    print("Step 4: 벤치마크 실행")
    print(f"{'=' * 60}")
    print(f"  총 검색: {len(active_strategies)}개 전략 x {len(queries)}개 쿼리 = {len(active_strategies) * len(queries)}회")

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
        _print_ragas_table(scores_df, scoring_profile=args.scoring_profile)

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
    _cleanup_strategies(active_strategies)

    print(f"\n{'═' * 60}")
    print(f" 벤치마크 완료 — 전략 {len(active_strategies)}개 비교")
    print(f"{'═' * 60}")


# ===========================================================================
# main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="전체 조합 벤치마크 — 3-Layer 교차 조합 + 2-Pass 실행"
    )

    # 공통 옵션
    parser.add_argument("--k", type=int, default=3, help="검색 결과 수 (기본: 3)")
    parser.add_argument("--no_ragas", action="store_true", help="RAGAS 평가 건너뛰기")
    parser.add_argument("--reindex", action="store_true", help="기존 인덱스 삭제 후 재인덱싱")

    # 새 모드 옵션
    parser.add_argument("--preset", type=str, default=None,
                        help="프리셋 선택: quick|standard|full")
    parser.add_argument("--pass1-only", action="store_true",
                        help="레이턴시만 측정 (RAGAS 없음)")
    parser.add_argument("--pass1-workers", type=int, default=0,
                        help="Pass 1 전략 병렬 워커 수 (기본: 0=순차). 예: --pass1-workers 4")
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
    # 레거시 모드 옵션
    parser.add_argument("--combos", type=str, default=None,
                        help="DenseSparse 조합 ID (쉼표 구분, 예: 1,3,4)")
    parser.add_argument("--skip_colbert", action="store_true",
                        help="ColBERT 단독 전략 건너뛰기")
    parser.add_argument("--skip_rerank", action="store_true",
                        help="ColBERTRerank 전략 건너뛰기")
    parser.add_argument("--skip_contextual", action="store_true",
                        help="Contextual Retrieval 전략 건너뛰기")
    parser.add_argument("--skip_flashrank", action="store_true",
                        help="FlashRank Rerank 전략 건너뛰기")
    parser.add_argument("--contextual_base", type=int, default=3,
                        help="Contextual Retrieval 기반 DenseSparse 조합 ID (기본: 3)")

    args = parser.parse_args()

    # --preset이 지정되면 새 모드, 아니면 레거시 모드
    if args.preset is not None:
        _run_preset_mode(args)
    else:
        _run_legacy_mode(args)


if __name__ == "__main__":
    main()
