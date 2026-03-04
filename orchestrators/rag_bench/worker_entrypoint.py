"""
K8s 벤치마크 워커 — 2-Phase 실행.

Phase 1 (prep): 단일 카테고리의 데이터 준비 + Contextual enrichment
Phase 2 (bench): 단일 (카테고리 × 전략 조합) 벤치마크 실행

환경변수:
  # 공통
  WORKER_PHASE      : "prep" | "bench"
  BENCH_CATEGORY    : 카테고리 (general, legal, business, medical, technical)
  RESULTS_DIR       : 결과 저장 경로 (기본: /results)
  WORKSPACE_DIR     : 임시 작업 공간 (기본: /workspace)
  LLM_PROVIDER      : "openai" 고정 (K8s에서는 Ollama 미사용)
  OPENAI_API_KEY    : OpenAI API 키

  # Phase 1 전용
  BENCH_MAX_CORPUS  : HF 코퍼스 샘플링 크기 (기본: 10000)
  BENCH_MAX_QUERIES : HF 쿼리 샘플링 크기 (기본: 100)
  CONTEXTUAL_LLM    : Contextual Retrieval LLM (기본: gpt-4o-mini)

  # Phase 2 전용
  COMBO_DENSE       : Dense 모델 키 (예: bge-m3)
  COMBO_SPARSE      : Sparse 모델 키 (예: korean_bm25)
  COMBO_RERANKER    : 리랭커 (예: colbert, flashrank)
  COMBO_LLM_SUPPORT : LLM 지원 (예: contextual)
  BENCH_K           : 검색 결과 수 (기본: 3)
  BENCH_PASS1_ONLY  : "true"이면 레이턴시만 측정
  BENCH_NO_RAGAS    : "true"이면 RAGAS 평가 건너뜀
  COLBERT_MODEL     : ColBERT 모델명 (기본: jinaai/jina-colbert-v2)

Qdrant는 파일 모드(path=)를 사용하므로 별도 서버 불필요.
"""

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple


def env_bool(key: str) -> bool:
    return os.environ.get(key, "").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# 직렬화 유틸
# ---------------------------------------------------------------------------


def serialize_documents(docs: list) -> list:
    """langchain Document → JSON-serializable dict."""
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]


def deserialize_documents(data: list) -> list:
    """JSON dict → langchain Document."""
    from langchain_core.documents import Document

    return [
        Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data
    ]


def serialize_parent_pairs(pairs: list) -> list:
    """(parent_id, Document) 튜플 리스트 → JSON-serializable."""
    return [
        {
            "parent_id": pid,
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for pid, doc in pairs
    ]


def deserialize_parent_pairs(data: list) -> list:
    """JSON → (parent_id, Document) 튜플 리스트."""
    from langchain_core.documents import Document

    return [
        (
            d["parent_id"],
            Document(page_content=d["page_content"], metadata=d["metadata"]),
        )
        for d in data
    ]


PREPARED_DIR_NAME = "prepared"


# ===========================================================================
# Phase 1: Prep
# ===========================================================================


def phase_prep():
    """카테고리 데이터 준비 + Contextual enrichment → PVC에 직렬화."""
    category = os.environ["BENCH_CATEGORY"]
    max_corpus = int(os.environ.get("BENCH_MAX_CORPUS", "10000"))
    max_queries = int(os.environ.get("BENCH_MAX_QUERIES", "100"))
    contextual_llm = os.environ.get("CONTEXTUAL_LLM", "gpt-4o-mini")
    results_dir = Path(os.environ.get("RESULTS_DIR", "/results"))

    print(f"\n{'=' * 60}")
    print(f" Phase 1: Prep — {category}")
    print(f"{'=' * 60}")
    print(f"  Max Corpus   : {max_corpus:,}")
    print(f"  Max Queries  : {max_queries:,}")
    print(f"  Context LLM  : {contextual_llm}")
    print()

    from autorag_retrieval.config import setup_ssl_bypass

    setup_ssl_bypass()

    from autorag_retrieval.document_types.types import DocType

    doc_type = DocType(category)

    # ── 1. HF 데이터 로드 + 청킹 ──────────────────────────────
    t0 = time.time()
    parent_pairs, child_chunks, qa_pairs = _load_hf_data(
        doc_type,
        results_dir / category,
        max_corpus,
        max_queries,
    )
    t_data = time.time() - t0
    print(
        f"\n  데이터 준비: {len(child_chunks):,} children / {len(qa_pairs):,} QA ({t_data:.0f}s)"
    )

    # ── 2. Contextual enrichment ───────────────────────────────
    t0 = time.time()
    from autorag_retrieval.strategies.contextual_retrieval import (
        ContextualRetrievalStrategy,
    )

    cache_dir = results_dir / category / "ctx_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ctx = ContextualRetrievalStrategy(
        base_strategy=None,
        parent_pairs=parent_pairs,
        llm_model=contextual_llm,
        cache_dir=str(cache_dir),
    )
    enriched_chunks = ctx.enrich_only(child_chunks)
    t_enrich = time.time() - t0
    print(f"  Enriched: {len(enriched_chunks):,}개 ({t_enrich:.0f}s)")

    # ── 3. PVC에 직렬화 ────────────────────────────────────────
    out_dir = results_dir / category / PREPARED_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(out_dir / "child_chunks.json", serialize_documents(child_chunks))
    _write_json(out_dir / "parent_pairs.json", serialize_parent_pairs(parent_pairs))
    _write_json(out_dir / "qa_pairs.json", qa_pairs)
    _write_json(out_dir / "enriched_chunks.json", serialize_documents(enriched_chunks))

    # 완료 시그널
    _write_json(
        out_dir / "DONE",
        {
            "category": category,
            "n_children": len(child_chunks),
            "n_parents": len(parent_pairs),
            "n_qa": len(qa_pairs),
            "n_enriched": len(enriched_chunks),
            "data_time_s": round(t_data, 1),
            "enrich_time_s": round(t_enrich, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    total = t_data + t_enrich
    print(f"\n  Phase 1 완료: {total:.0f}s ({total / 60:.1f}min)")
    print(f"  저장: {out_dir}")


# ===========================================================================
# Phase 2: Bench
# ===========================================================================


def phase_bench():
    """단일 (카테고리 × 전략 조합) 벤치마크 실행."""
    category = os.environ["BENCH_CATEGORY"]
    dense = os.environ["COMBO_DENSE"]
    sparse = os.environ["COMBO_SPARSE"]
    reranker = os.environ["COMBO_RERANKER"]
    llm_support = os.environ["COMBO_LLM_SUPPORT"]
    k = int(os.environ.get("BENCH_K", "3"))
    pass1_only = env_bool("BENCH_PASS1_ONLY")
    no_ragas = env_bool("BENCH_NO_RAGAS")
    colbert_model = os.environ.get("COLBERT_MODEL", "jinaai/jina-colbert-v2")
    contextual_llm = os.environ.get("CONTEXTUAL_LLM", "gpt-4o-mini")
    results_dir = Path(os.environ.get("RESULTS_DIR", "/results"))
    workspace_dir = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))

    # COMBO_LABEL은 오케스트레이터가 주입 — 결과 디렉토리명의 단일 진실 공급원
    combo_label = os.environ.get("COMBO_LABEL")
    if not combo_label:
        print("ERROR: COMBO_LABEL 환경변수 필수 (오케스트레이터가 주입해야 함)")
        sys.exit(1)
    combo_display = f"{dense}+{sparse}+{reranker}+{llm_support}"

    print(f"\n{'=' * 60}")
    print(f" Phase 2: Bench — {category} / {combo_display}")
    print(f"{'=' * 60}")
    print(f"  k={k}  pass1_only={pass1_only}  no_ragas={no_ragas}")
    print()

    from autorag_retrieval.config import setup_ssl_bypass

    setup_ssl_bypass()

    # ── 1. 준비된 데이터 로드 ──────────────────────────────────
    prep_dir = results_dir / category / PREPARED_DIR_NAME
    if not (prep_dir / "DONE").exists():
        print(f"ERROR: Phase 1 미완료 — {prep_dir / 'DONE'} 없음")
        sys.exit(1)

    t0 = time.time()
    child_chunks = deserialize_documents(_read_json(prep_dir / "child_chunks.json"))
    parent_pairs = deserialize_parent_pairs(_read_json(prep_dir / "parent_pairs.json"))
    qa_pairs = _read_json(prep_dir / "qa_pairs.json")
    enriched_chunks = deserialize_documents(
        _read_json(prep_dir / "enriched_chunks.json")
    )
    t_load = time.time() - t0
    print(
        f"  데이터 로드: {len(child_chunks):,} children, {len(enriched_chunks):,} enriched ({t_load:.1f}s)"
    )

    # ── 2. 전략 빌드 ──────────────────────────────────────────
    from autorag_retrieval.combo import CacheConfig, ComboSpec, IndexCacheManager
    from autorag_retrieval.combo.builder import build_strategy_from_spec

    spec = ComboSpec(
        dense=dense, sparse=sparse, reranker=reranker, llm_support=llm_support
    )
    cfg = CacheConfig(
        colbert_model=colbert_model,
        colbert_device="cpu",
        contextual_llm=contextual_llm,
    )
    index_cache = IndexCacheManager(cfg)

    qdrant_dir = workspace_dir / category / combo_label / "qdrant"
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"  전략 빌드 중: {spec.label}")
    strategy = build_strategy_from_spec(
        spec=spec,
        index_cache=index_cache,
        child_chunks=child_chunks,
        parent_pairs=parent_pairs,
        reindex=False,
        pre_enriched=enriched_chunks,
        qdrant_base_dir=qdrant_dir,
    )
    t_build = time.time() - t0
    print(f"  빌드 완료: {t_build:.1f}s")
    _release_memory()

    # ── 3. Pass 1: 레이턴시 ────────────────────────────────────
    from autorag_retrieval.runner import BenchmarkRunner

    queries = [qa["question"] for qa in qa_pairs]
    ground_truths = [qa.get("ground_truth", "") for qa in qa_pairs]

    print(f"\n  Pass 1 — 레이턴시 (1 전략 x {len(queries)} 쿼리)")
    runner = BenchmarkRunner(
        strategies=[strategy], queries=queries, k=k, evaluator=None
    )
    runner.run()
    runner.compare()
    latency_df = runner.to_dataframe()

    # ── 4. Pass 2: RAGAS 평가 ─────────────────────────────────
    ragas_df = None
    if not pass1_only and not no_ragas:
        print("\n  Pass 2 — RAGAS 평가")
        try:
            from autorag_rag_eval import ExtendedRAGEvaluator
            from autorag_rag_eval.metrics import MetricPreset

            evaluator = ExtendedRAGEvaluator(preset=MetricPreset("core_only"))
            eval_runner = BenchmarkRunner(
                strategies=[strategy],
                queries=queries,
                k=k,
                evaluator=evaluator,
            )
            eval_runner.inject_results(runner._results)
            ragas_df = eval_runner.evaluate(ground_truths=ground_truths)
        except Exception as e:
            print(f"  RAGAS 평가 실패: {e}")
            traceback.print_exc()

    # ── 5. 결과 저장 ──────────────────────────────────────────
    # combo_label은 오케스트레이터와 동일한 값 — 경로 불일치 방지
    out_dir = results_dir / category / combo_label
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_combo_result(out_dir, category, combo_display, latency_df, ragas_df, qa_pairs)

    total = time.time() - (t0 - t_build - t_load)  # rough
    print(f"\n  Phase 2 완료: {category}/{combo_label}")
    print(f"  결과: {out_dir}")


# ===========================================================================
# 공통 유틸
# ===========================================================================


def _load_hf_data(doc_type, cat_dir: Path, max_corpus: int, max_queries: int) -> Tuple:
    from autorag_retrieval.datasets.hf_loader import (
        HFDatasetLoader,
        beir_to_parent_child_chunks,
    )
    from autorag_retrieval.document_types.types import DocType

    loader = HFDatasetLoader(max_corpus=max_corpus, max_queries=max_queries)
    cache_dir = cat_dir / "hf_cache"

    source_names = {
        DocType.GENERAL: "miracl-ko",
        DocType.LEGAL: "markers_bm-law",
        DocType.BUSINESS: "markers_bm-finance+public+commerce",
        DocType.MEDICAL: "publichealth-qa-ko",
        DocType.TECHNICAL: "nanobeir-ko-NanoSCIDOCS",
    }
    source_name = source_names.get(doc_type, "unknown")

    dataset = HFDatasetLoader.load_cache(cache_dir, source_name)
    if dataset:
        print(
            f"  [캐시] {source_name} ({dataset.n_docs:,}docs / {dataset.n_queries:,}queries)"
        )
    else:
        print(f"  HuggingFace 데이터셋 로드: {doc_type.value}")
        dataset = loader.load(doc_type)
        loader.save_cache(dataset, cache_dir)

    if dataset.n_docs == 0:
        raise ValueError(f"{doc_type.value} 코퍼스 비어 있음")

    print(f"  코퍼스: {dataset.n_docs:,}개 | 쿼리: {dataset.n_queries:,}개")
    parent_pairs, child_chunks = beir_to_parent_child_chunks(dataset)
    print(f"  청크: {len(parent_pairs):,} parents / {len(child_chunks):,} children")

    qa_pairs = dataset.get_qa_pairs()
    if not qa_pairs:
        raise ValueError(f"{doc_type.value} QA 쌍 없음")

    return parent_pairs, child_chunks, qa_pairs


def _save_combo_result(
    out_dir: Path,
    category: str,
    combo_label: str,
    latency_df,
    ragas_df,
    qa_pairs: list,
) -> None:
    if latency_df is not None:
        tmp = out_dir / "latency.csv.tmp"
        latency_df.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.rename(out_dir / "latency.csv")

    result: Dict[str, Any] = {
        "category": category,
        "combo": combo_label,
        "n_qa": len(qa_pairs),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if ragas_df is not None:
        result["ragas"] = ragas_df.to_dict(orient="records")
        tmp = out_dir / "ragas.csv.tmp"
        ragas_df.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.rename(out_dir / "ragas.csv")

    _write_json(out_dir / "result.json", result)
    # DONE은 반드시 마지막에 기록 — 수집기가 완료 판단에 사용
    _write_json(
        out_dir / "DONE",
        {
            "category": category,
            "combo": combo_label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"  결과 저장: {out_dir}")


def _safe_label(label: str) -> str:
    """K8s / 파일 시스템 안전한 이름으로 변환."""
    return label.replace("+", "_").replace("/", "-").replace(":", "-").lower()


def _write_json(path: Path, data) -> None:
    """원자적 JSON 쓰기 — .tmp에 쓴 후 rename으로 교체."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _release_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ===========================================================================
# Main
# ===========================================================================


def main():
    os.environ.setdefault("LLM_PROVIDER", "openai")

    phase = os.environ.get("WORKER_PHASE", "").lower()
    if phase == "prep":
        phase_prep()
    elif phase == "bench":
        phase_bench()
    else:
        print(
            f"ERROR: WORKER_PHASE 환경변수 필수 ('prep' 또는 'bench'). 현재: '{phase}'"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
