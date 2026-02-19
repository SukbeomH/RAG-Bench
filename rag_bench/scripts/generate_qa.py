"""
QA 데이터셋 자동 생성 스크립트.

rag_bench/docs/*.md → Parent-Child 청킹 → GPT-4o-mini QA 생성
→ rag_bench/_benchdata/qa_dataset.json

방법:
  legacy: GPT-4o-mini 기반 단순 QA 생성 (기본)
  ragas:  RAGAS KnowledgeGraph 기반 다양한 QA 생성

Usage:
    # 레거시 방식
    python -m rag_bench.scripts.generate_qa --method legacy --num_qa 20

    # RAGAS KG 방식
    python -m rag_bench.scripts.generate_qa --method ragas --num_qa 50
    python -m rag_bench.scripts.generate_qa --method ragas --build-kg-only
    python -m rag_bench.scripts.generate_qa --method ragas --num_qa 50 --reuse-kg
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

from rag_bench.config import (
    BENCH_DATA_DIR,
    BENCH_DOCS_DIR,
    setup_ssl_bypass,
)
from rag_bench.indexing.chunker import create_parent_child_chunks
from rag_bench.run_tracker import RunTracker, track_openai_tokens


def _compute_docs_hash(docs_dir: Path) -> str:
    """docs 디렉토리 내 *.md 파일 내용의 해시를 계산한다."""
    h = hashlib.sha256()
    for md_file in sorted(docs_dir.glob("*.md")):
        h.update(md_file.read_bytes())
    return h.hexdigest()[:16]


def _sample_parents(
    parent_pairs: list,
    num_qa: int,
    min_size: int = 200,
) -> list:
    """Parent 청크에서 문서별 균등 샘플링한다."""
    # 최소 크기 필터링 (max_size 제한 없음 — QA 생성 시 context[:3000]으로 잘림)
    valid = [
        (pid, doc)
        for pid, doc in parent_pairs
        if len(doc.page_content) >= min_size
    ]
    if not valid:
        valid = list(parent_pairs)

    # 문서별 그룹핑
    by_source: dict = {}
    for pid, doc in valid:
        source = doc.metadata.get("source", "unknown")
        by_source.setdefault(source, []).append((pid, doc))

    # 문서별 균등 배분
    sources = list(by_source.keys())
    per_source = max(1, num_qa // len(sources))
    remainder = num_qa - per_source * len(sources)

    sampled = []
    for i, source in enumerate(sources):
        pool = by_source[source]
        n = per_source + (1 if i < remainder else 0)
        n = min(n, len(pool))
        sampled.extend(random.sample(pool, n))

    # 부족하면 전체에서 추가 샘플링
    if len(sampled) < num_qa:
        remaining = [p for p in valid if p not in sampled]
        extra = min(num_qa - len(sampled), len(remaining))
        sampled.extend(random.sample(remaining, extra))

    return sampled[:num_qa]


def _generate_qa_pairs(
    sampled_parents: list,
    num_qa: int,
) -> List[dict]:
    """GPT-4o-mini를 사용하여 QA 쌍을 생성한다."""
    import httpx
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        http_client=httpx.Client(verify=False),
    )

    qa_pairs = []
    for i, (pid, doc) in enumerate(sampled_parents):
        context = doc.page_content
        source = doc.metadata.get("source", "unknown")

        prompt = (
            "아래 컨텍스트를 읽고, 이 내용에 기반한 질문(question)과 "
            "정답(ground_truth)을 JSON 형식으로 생성하세요.\n\n"
            "규칙:\n"
            "- 질문은 컨텍스트의 핵심 내용을 묻는 한국어 질문이어야 합니다.\n"
            "- 정답은 컨텍스트에서 직접 찾을 수 있는 사실 기반 답변이어야 합니다.\n"
            "- 질문은 구체적이고 명확해야 합니다.\n"
            "- 정답은 2~3문장 이내로 간결하게 작성하세요.\n\n"
            f"컨텍스트:\n{context[:3000]}\n\n"
            '출력 형식 (JSON만 출력):\n{"question": "...", "ground_truth": "..."}'
        )

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()
            # JSON 추출 (코드블록 제거)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            qa = json.loads(content)
            qa["parent_id"] = pid
            qa["source"] = source
            qa_pairs.append(qa)
            print(f"  [{i + 1}/{len(sampled_parents)}] Q: {qa['question'][:60]}...")
        except Exception as e:
            print(f"  [{i + 1}/{len(sampled_parents)}] 생성 실패: {e}")

    return qa_pairs


# ---------------------------------------------------------------------------
# RAGAS KG 기반 QA 생성
# ---------------------------------------------------------------------------

KG_SAVE_PATH = BENCH_DATA_DIR / "ragas_knowledge_graph.json"


def _generate_qa_ragas(
    parent_pairs: list,
    num_qa: int,
    reuse_kg: bool = False,
    build_kg_only: bool = False,
    num_personas: int = 3,
    query_dist: str = "balanced",
) -> Optional[List[dict]]:
    """RAGAS KnowledgeGraph + TestsetGenerator로 QA 생성."""
    import os

    import httpx
    from langchain_openai import OpenAIEmbeddings
    from openai import AsyncOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import llm_factory
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph
    from ragas.testset.transforms import default_transforms, apply_transforms
    from ragas.testset.synthesizers.single_hop import SingleHopQuerySynthesizer
    from ragas.testset.synthesizers.multi_hop import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
    )

    # 1. LLM/Embedding 초기화 (SSL bypass, llm_factory 네이티브)
    print("  [RAGAS] LLM/Embedding 초기화...")
    async_client = httpx.AsyncClient(verify=False)

    openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        http_client=async_client,
    )
    ragas_llm = llm_factory(model="gpt-4o-mini", client=openai_client)

    sync_http_client = httpx.Client(verify=False)
    embeddings = OpenAIEmbeddings(http_client=sync_http_client)
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
                        "document": doc,
                    },
                )
            )

        transforms = default_transforms(
            llm=ragas_llm,
            embedding_model=ragas_embeddings,
        )
        print(f"  [RAGAS] Transforms 적용 중 ({len(transforms)} transforms)...")
        apply_transforms(kg, transforms)
        print(f"  [RAGAS] KG 구축 완료 (nodes: {len(kg.nodes)})")

        # KG 저장 (재사용용)
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
            (SingleHopQuerySynthesizer(llm=ragas_llm), 1.0),
        ]
    elif query_dist == "multi_hop":
        query_distribution = [
            (MultiHopAbstractQuerySynthesizer(llm=ragas_llm), 0.5),
            (MultiHopSpecificQuerySynthesizer(llm=ragas_llm), 0.5),
        ]
    else:  # balanced
        query_distribution = [
            (SingleHopQuerySynthesizer(llm=ragas_llm), 0.6),
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
        # 신규 필드 (하위 호환 — 추가 전용)
        synth_name = row.get("synthesizer_name", "")
        if synth_name:
            qa["synthesizer_name"] = synth_name

        # query_type 추론
        if "SingleHop" in synth_name:
            qa["query_type"] = "single_hop"
        elif "MultiHop" in synth_name:
            qa["query_type"] = "multi_hop"
        else:
            qa["query_type"] = "unknown"

        # reference_contexts
        ref_ctx = row.get("reference_contexts", None)
        if ref_ctx is not None:
            if isinstance(ref_ctx, list):
                qa["reference_contexts"] = ref_ctx
            else:
                qa["reference_contexts"] = [str(ref_ctx)]

        qa_pairs.append(qa)
        print(f"  [{i + 1}/{len(df)}] Q: {qa['question'][:60]}...")

    return qa_pairs


# ---------------------------------------------------------------------------
# Legacy / RAGAS 분기 실행
# ---------------------------------------------------------------------------


def _run_legacy_method(args, parent_pairs, child_chunks, docs_hash, qa_path, tracker):
    """레거시 GPT-4o-mini 기반 QA 생성."""
    # 2. Parent 샘플링
    print(f"\n=== Step 2: Parent 샘플링 ({args.num_qa}개) ===")
    sampled = _sample_parents(parent_pairs, args.num_qa)
    print(f"  샘플링된 Parent 청크: {len(sampled)}개")

    # 3. QA 생성 (토큰 추적)
    print(f"\n=== Step 3: GPT-4o-mini QA 생성 ===")
    with tracker.phase("qa_generation"):
        with track_openai_tokens() as qa_tokens:
            qa_pairs = _generate_qa_pairs(sampled, args.num_qa)
    if qa_tokens.total_tokens > 0:
        qa_tokens.llm_model = "gpt-4o-mini"
        tracker.add_tokens(qa_tokens, phase="qa_generation")
        print(f"  토큰 사용: {qa_tokens.total_tokens:,} "
              f"(prompt: {qa_tokens.prompt_tokens:,}, "
              f"completion: {qa_tokens.completion_tokens:,}, "
              f"cost: ${qa_tokens.total_cost_usd:.4f})")

    # 4. 저장
    dataset = {
        "docs_hash": docs_hash,
        "num_qa": len(qa_pairs),
        "method": "legacy",
        "qa_pairs": qa_pairs,
    }
    qa_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 완료: {len(qa_pairs)}개 QA 저장 → {qa_path} ===")

    return child_chunks


def _run_ragas_method(args, parent_pairs, child_chunks, docs_hash, qa_path, tracker):
    """RAGAS KG 기반 QA 생성."""
    print(f"\n=== Step 2: RAGAS KG 기반 QA 생성 ===")
    with tracker.phase("ragas_kg_qa_generation"):
        with track_openai_tokens() as qa_tokens:
            qa_pairs = _generate_qa_ragas(
                parent_pairs=parent_pairs,
                num_qa=args.num_qa,
                reuse_kg=args.reuse_kg,
                build_kg_only=args.build_kg_only,
                num_personas=args.num_personas,
                query_dist=args.query_dist,
            )
    if qa_tokens.total_tokens > 0:
        qa_tokens.llm_model = "gpt-4o-mini"
        tracker.add_tokens(qa_tokens, phase="ragas_kg_qa_generation")
        print(f"  토큰 사용: {qa_tokens.total_tokens:,} "
              f"(prompt: {qa_tokens.prompt_tokens:,}, "
              f"completion: {qa_tokens.completion_tokens:,}, "
              f"cost: ${qa_tokens.total_cost_usd:.4f})")

    if qa_pairs is None:
        # --build-kg-only
        return child_chunks

    # 저장
    dataset = {
        "docs_hash": docs_hash,
        "num_qa": len(qa_pairs),
        "method": "ragas",
        "query_distribution": args.query_dist,
        "qa_pairs": qa_pairs,
    }
    qa_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 완료: {len(qa_pairs)}개 QA 저장 → {qa_path} ===")

    return child_chunks


def main():
    parser = argparse.ArgumentParser(description="QA 데이터셋 자동 생성")
    parser.add_argument("--num_qa", type=int, default=20, help="생성할 QA 수 (기본: 20)")
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 재생성")
    parser.add_argument("--method", type=str, default="legacy",
                        choices=["legacy", "ragas"],
                        help="QA 생성 방법 (기본: legacy)")
    parser.add_argument("--build-kg-only", action="store_true",
                        help="[ragas] KG만 구축, QA 생성 안 함")
    parser.add_argument("--reuse-kg", action="store_true",
                        help="[ragas] 기존 KG 파일 재사용")
    parser.add_argument("--num-personas", type=int, default=3,
                        help="[ragas] 자동 페르소나 수 (기본: 3)")
    parser.add_argument("--query-dist", type=str, default="balanced",
                        choices=["single_hop", "multi_hop", "balanced"],
                        help="[ragas] 쿼리 분포 (기본: balanced)")
    args = parser.parse_args()

    setup_ssl_bypass()

    # docs 디렉토리 확인
    md_files = list(BENCH_DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"Error: {BENCH_DOCS_DIR}에 .md 파일이 없습니다.")
        sys.exit(1)

    print(f"문서 디렉토리: {BENCH_DOCS_DIR}")
    print(f"발견된 문서: {len(md_files)}개")
    print(f"QA 생성 방법: {args.method}")

    # 캐시 확인 (--build-kg-only일 때는 캐시 건너뜀)
    BENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    qa_path = BENCH_DATA_DIR / "qa_dataset.json"
    docs_hash = _compute_docs_hash(BENCH_DOCS_DIR)

    if qa_path.exists() and not args.force and not args.build_kg_only:
        existing = json.loads(qa_path.read_text(encoding="utf-8"))
        if existing.get("docs_hash") == docs_hash:
            print(f"캐시된 QA 데이터셋 사용: {qa_path}")
            print(f"  QA 수: {len(existing['qa_pairs'])}개")
            print(f"  방법: {existing.get('method', 'legacy')}")
            print("  재생성하려면 --force 옵션을 사용하세요.")
            return

    # RunTracker 초기화
    tracker = RunTracker(output_dir=BENCH_DATA_DIR)

    # 1. Parent-Child 청킹
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

    # 방법별 분기
    if args.method == "ragas":
        child_chunks = _run_ragas_method(args, parent_pairs, child_chunks, docs_hash, qa_path, tracker)
    else:
        child_chunks = _run_legacy_method(args, parent_pairs, child_chunks, docs_hash, qa_path, tracker)

    # 수행 이력 저장
    tracker.set_config(
        preset=f"qa_generation_{args.method}",
        k=0,
        top_n=None,
        pass1_only=False,
        layers=False,
        num_combos=0,
        num_queries=args.num_qa,
        num_docs=len(child_chunks),
    )
    tracker.finalize()


if __name__ == "__main__":
    main()
