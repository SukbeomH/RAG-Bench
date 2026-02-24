"""
QA 데이터셋 자동 생성 스크립트.

docs/*.pdf → (선택: 페이지 샘플링) → rag_bench/docs/*.md
           → Parent-Child 청킹
           → RAGAS KnowledgeGraph QA 생성
           → rag_bench/_benchdata/qa_dataset.json

Usage:
    # 기본 (기존 .md 파일 사용, 청크 수 × max_qa_per_page 만큼 생성)
    python -m rag_bench.scripts.generate_qa

    # PDF 페이지 샘플링 적용 (대용량 문서 비용 절감)
    python -m rag_bench.scripts.generate_qa --sample_pages --max_qa_per_page 2

    # KG만 사전 구축 (QA 생성 없이)
    python -m rag_bench.scripts.generate_qa --build-kg-only

    # 기존 KG 재사용하여 QA 생성
    python -m rag_bench.scripts.generate_qa --reuse-kg --max_qa_per_page 3
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional

from rag_bench.config import (
    BENCH_DATA_DIR,
    BENCH_DOCS_DIR,
    DEFAULT_RAGAS_QA_LLM,
    DOCS_DIR,
    make_async_http_client,
    make_http_client,
    setup_ssl_bypass,
)
from rag_bench.indexing.chunker import create_parent_child_chunks
from rag_bench.indexing.pdf_converter import pdfs_to_markdowns
from rag_bench.run_tracker import RunTracker, track_openai_tokens


def _compute_docs_hash(docs_dir: Path) -> str:
    """docs 디렉토리 내 *.md 파일 내용의 해시를 계산한다."""
    h = hashlib.sha256()
    for md_file in sorted(docs_dir.glob("*.md")):
        h.update(md_file.read_bytes())
    return h.hexdigest()[:16]


def compute_effective_num_qa(args, parent_pairs: list) -> int:
    """청크 수 × max_qa_per_page 로 QA 생성 수를 결정한다."""
    sampled_page_count = len(parent_pairs)
    effective = sampled_page_count * args.max_qa_per_page
    print(
        f"  [QA 수] 청크 {sampled_page_count}개 × {args.max_qa_per_page}/청크 = {effective}개"
    )
    return effective


# ---------------------------------------------------------------------------
# RAGAS KG 기반 QA 생성
# ---------------------------------------------------------------------------

KG_SAVE_PATH = BENCH_DATA_DIR / "ragas_knowledge_graph.json"


def generate_qa_ragas(
    parent_pairs: list,
    num_qa: int,
    reuse_kg: bool = False,
    build_kg_only: bool = False,
    num_personas: int = 3,
    query_dist: str = "balanced",
) -> Optional[List[dict]]:
    """RAGAS KnowledgeGraph + TestsetGenerator로 QA 생성."""
    import os

    from langchain_openai import OpenAIEmbeddings
    from openai import AsyncOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import llm_factory
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph
    from ragas.testset.transforms import default_transforms, apply_transforms
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.testset.synthesizers.multi_hop import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
    )

    # 1. LLM/Embedding 초기화 (OpenAI 고정 — QA 생성 품질)
    print("  [RAGAS] LLM/Embedding 초기화...")
    openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        http_client=make_async_http_client(),
    )
    ragas_llm = llm_factory(model=DEFAULT_RAGAS_QA_LLM, client=openai_client)

    embeddings = OpenAIEmbeddings(http_client=make_http_client())
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # 2. LangChain 문서 준비
    docs = [doc for _, doc in parent_pairs]
    print(f"  [RAGAS] 문서 수: {len(docs)}개")

    # 3. KG 로드 또는 구축
    kg = None
    if reuse_kg and KG_SAVE_PATH.exists():
        print(f"  [RAGAS] 기존 KG 로드: {KG_SAVE_PATH}")
        kg = KnowledgeGraph.load(str(KG_SAVE_PATH))
        print(f"  [RAGAS] KG 로드 완료 (nodes: {len(kg.nodes)})")
    else:
        print("  [RAGAS] KG 구축 시작...")
        from ragas.testset.graph import Node, NodeType

        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata,
                    },
                )
            )

        transforms = default_transforms(
            documents=docs,
            llm=ragas_llm,
            embedding_model=ragas_embeddings,
        )
        print(f"  [RAGAS] Transforms 적용 중 ({len(transforms)} transforms)...")
        apply_transforms(kg, transforms)
        print(f"  [RAGAS] KG 구축 완료 (nodes: {len(kg.nodes)})")

        KG_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        kg.save(str(KG_SAVE_PATH))
        print(f"  [RAGAS] KG 저장: {KG_SAVE_PATH}")

    if build_kg_only:
        print("  [RAGAS] --build-kg-only: KG 구축만 완료. QA 생성 건너뜀.")
        return None

    # 4. TestsetGenerator 생성
    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
        knowledge_graph=kg,
    )

    # 5. query_distribution 설정
    if query_dist == "single_hop":
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 1.0),
        ]
    elif query_dist == "multi_hop":
        query_distribution = [
            (MultiHopAbstractQuerySynthesizer(llm=ragas_llm), 0.5),
            (MultiHopSpecificQuerySynthesizer(llm=ragas_llm), 0.5),
        ]
    else:  # balanced
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 0.6),
            (MultiHopAbstractQuerySynthesizer(llm=ragas_llm), 0.2),
            (MultiHopSpecificQuerySynthesizer(llm=ragas_llm), 0.2),
        ]

    print(f"  [RAGAS] QA 생성 시작 (n={num_qa}, dist={query_dist}, personas={num_personas})...")
    testset = generator.generate(
        testset_size=num_qa,
        query_distribution=query_distribution,
        num_personas=num_personas,
    )

    # 6. testset → qa_pairs 변환
    df = testset.to_pandas()
    print(f"  [RAGAS] 생성된 샘플: {len(df)}개")

    qa_pairs = []
    for i, row in df.iterrows():
        qa = {
            "question": row.get("user_input", ""),
            "ground_truth": row.get("reference", ""),
            "parent_id": f"ragas_{i}",
            "source": "ragas_testset",
        }
        synth_name = row.get("synthesizer_name", "")
        if synth_name:
            qa["synthesizer_name"] = synth_name

        if "SingleHop" in synth_name:
            qa["query_type"] = "single_hop"
        elif "MultiHop" in synth_name:
            qa["query_type"] = "multi_hop"
        else:
            qa["query_type"] = "unknown"

        ref_ctx = row.get("reference_contexts", None)
        if ref_ctx is not None:
            qa["reference_contexts"] = ref_ctx if isinstance(ref_ctx, list) else [str(ref_ctx)]

        qa_pairs.append(qa)
        print(f"  [{i + 1}/{len(df)}] Q: {qa['question'][:60]}...")

    return qa_pairs


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="QA 데이터셋 자동 생성 (RAGAS KG 방식)")
    parser.add_argument(
        "--sample_pages", action="store_true",
        help="docs/*.pdf를 페이지 샘플링하여 rag_bench/docs/*.md 재생성",
    )
    parser.add_argument(
        "--page_sample_ratio", type=float, default=0.1,
        help="페이지 샘플링 비율 (기본: 0.1 = 10%%, --sample_pages와 함께 사용)",
    )
    parser.add_argument(
        "--max_sample_pages", type=int, default=5,
        help="최대 샘플 페이지 수 (기본: 5, --sample_pages와 함께 사용)",
    )
    parser.add_argument(
        "--max_qa_per_page", type=int, default=2,
        help="청크당 최대 QA 수 (기본: 2, --sample_pages와 함께 QA 상한 계산에 사용)",
    )
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 재생성")
    parser.add_argument("--build-kg-only", action="store_true",
                        help="KG만 구축, QA 생성 안 함")
    parser.add_argument("--reuse-kg", action="store_true",
                        help="기존 KG 파일 재사용")
    parser.add_argument("--num-personas", type=int, default=3,
                        help="자동 페르소나 수 (기본: 3)")
    parser.add_argument(
        "--query-dist", type=str, default="balanced",
        choices=["single_hop", "multi_hop", "balanced"],
        help="쿼리 분포 (기본: balanced)",
    )
    args = parser.parse_args()

    setup_ssl_bypass()

    # Step 0: PDF 페이지 샘플링 → Markdown 변환 (--sample_pages 시)
    if args.sample_pages:
        pdf_files = list(DOCS_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"Warning: {DOCS_DIR}에 PDF 파일이 없습니다. 기존 .md 파일을 사용합니다.")
        else:
            print(f"\n=== Step 0: PDF 페이지 샘플링 ({len(pdf_files)}개) ===")
            BENCH_DOCS_DIR.mkdir(parents=True, exist_ok=True)
            pdfs_to_markdowns(
                docs_dir=str(DOCS_DIR),
                output_dir=str(BENCH_DOCS_DIR),
                sample_pages=True,
                page_sample_ratio=args.page_sample_ratio,
                max_sample_pages=args.max_sample_pages,
            )

    # docs 디렉토리 확인
    md_files = list(BENCH_DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"Error: {BENCH_DOCS_DIR}에 .md 파일이 없습니다.")
        print("  docs/*.pdf 파일이 있다면 --sample_pages 옵션을 사용하세요.")
        sys.exit(1)

    print(f"\n문서 디렉토리: {BENCH_DOCS_DIR}")
    print(f"발견된 문서: {len(md_files)}개")

    # 캐시 확인
    BENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    qa_path = BENCH_DATA_DIR / "qa_dataset.json"
    docs_hash = _compute_docs_hash(BENCH_DOCS_DIR)

    if qa_path.exists() and not args.force and not args.build_kg_only:
        existing = json.loads(qa_path.read_text(encoding="utf-8"))
        if existing.get("docs_hash") == docs_hash:
            print(f"캐시된 QA 데이터셋 사용: {qa_path}")
            print(f"  QA 수: {len(existing['qa_pairs'])}개")
            print("  재생성하려면 --force 옵션을 사용하세요.")
            return

    tracker = RunTracker(output_dir=BENCH_DATA_DIR)

    # Step 1: Parent-Child 청킹
    print("\n=== Step 1: Parent-Child 청킹 ===")
    with tracker.phase("qa_chunking"):
        parent_store_path = BENCH_DATA_DIR / "parent_store"
        parent_pairs, child_chunks = create_parent_child_chunks(
            markdown_dir=str(BENCH_DOCS_DIR),
            parent_store_path=str(parent_store_path),
        )

    if not parent_pairs:
        print("Error: Parent 청크가 생성되지 않았습니다.")
        sys.exit(1)

    # Step 2: 유효 QA 수 계산
    effective_num_qa = compute_effective_num_qa(args, parent_pairs)

    # Step 3: RAGAS KG 기반 QA 생성
    print(f"\n=== Step 2: RAGAS KG 기반 QA 생성 (n={effective_num_qa}) ===")
    with tracker.phase("ragas_kg_qa_generation"):
        with track_openai_tokens() as qa_tokens:
            qa_pairs = generate_qa_ragas(
                parent_pairs=parent_pairs,
                num_qa=effective_num_qa,
                reuse_kg=args.reuse_kg,
                build_kg_only=args.build_kg_only,
                num_personas=args.num_personas,
                query_dist=args.query_dist,
            )
    if qa_tokens.total_tokens > 0:
        qa_tokens.llm_model = "gpt-4o-mini"
        tracker.add_tokens(qa_tokens, phase="ragas_kg_qa_generation")
        print(
            f"  토큰 사용: {qa_tokens.total_tokens:,} "
            f"(prompt: {qa_tokens.prompt_tokens:,}, "
            f"completion: {qa_tokens.completion_tokens:,}, "
            f"cost: ${qa_tokens.total_cost_usd:.4f})"
        )

    if qa_pairs is None:
        # --build-kg-only
        return

    # Step 4: 저장
    dataset = {
        "docs_hash": docs_hash,
        "num_qa": len(qa_pairs),
        "method": "ragas",
        "query_distribution": args.query_dist,
        "sampled_pages": args.sample_pages,
        "qa_pairs": qa_pairs,
    }
    qa_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 완료: {len(qa_pairs)}개 QA 저장 → {qa_path} ===")

    tracker.set_config(
        preset="qa_generation_ragas",
        k=0,
        top_n=None,
        pass1_only=False,
        layers=False,
        num_combos=0,
        num_queries=effective_num_qa,
        num_docs=len(child_chunks),
    )
    tracker.finalize()


if __name__ == "__main__":
    main()
