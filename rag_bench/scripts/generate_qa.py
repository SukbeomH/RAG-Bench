"""
QA 데이터셋 자동 생성 스크립트.

rag_bench/docs/*.md → Parent-Child 청킹 → GPT-4o-mini QA 생성
→ rag_bench/_benchdata/qa_dataset.json

Usage:
    python -m rag_bench.scripts.generate_qa [--num_qa 20] [--force]
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import List

from rag_bench.config import (
    BENCH_DATA_DIR,
    BENCH_DOCS_DIR,
    setup_ssl_bypass,
)
from rag_bench.indexing.chunker import create_parent_child_chunks


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


def main():
    parser = argparse.ArgumentParser(description="QA 데이터셋 자동 생성")
    parser.add_argument("--num_qa", type=int, default=20, help="생성할 QA 수 (기본: 20)")
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 재생성")
    args = parser.parse_args()

    setup_ssl_bypass()

    # docs 디렉토리 확인
    md_files = list(BENCH_DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"Error: {BENCH_DOCS_DIR}에 .md 파일이 없습니다.")
        sys.exit(1)

    print(f"문서 디렉토리: {BENCH_DOCS_DIR}")
    print(f"발견된 문서: {len(md_files)}개")

    # 캐시 확인
    BENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    qa_path = BENCH_DATA_DIR / "qa_dataset.json"
    docs_hash = _compute_docs_hash(BENCH_DOCS_DIR)

    if qa_path.exists() and not args.force:
        existing = json.loads(qa_path.read_text(encoding="utf-8"))
        if existing.get("docs_hash") == docs_hash:
            print(f"캐시된 QA 데이터셋 사용: {qa_path}")
            print(f"  QA 수: {len(existing['qa_pairs'])}개")
            print("  재생성하려면 --force 옵션을 사용하세요.")
            return

    # 1. Parent-Child 청킹
    print("\n=== Step 1: Parent-Child 청킹 ===")
    parent_store_path = BENCH_DATA_DIR / "parent_store"
    parent_pairs, child_chunks = create_parent_child_chunks(
        markdown_dir=str(BENCH_DOCS_DIR),
        parent_store_path=str(parent_store_path),
    )

    if not parent_pairs:
        print("Error: Parent 청크가 생성되지 않았습니다.")
        sys.exit(1)

    # 2. Parent 샘플링
    print(f"\n=== Step 2: Parent 샘플링 ({args.num_qa}개) ===")
    sampled = _sample_parents(parent_pairs, args.num_qa)
    print(f"  샘플링된 Parent 청크: {len(sampled)}개")

    # 3. QA 생성
    print(f"\n=== Step 3: GPT-4o-mini QA 생성 ===")
    qa_pairs = _generate_qa_pairs(sampled, args.num_qa)

    # 4. 저장
    dataset = {
        "docs_hash": docs_hash,
        "num_qa": len(qa_pairs),
        "qa_pairs": qa_pairs,
    }
    qa_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 완료: {len(qa_pairs)}개 QA 저장 → {qa_path} ===")


if __name__ == "__main__":
    main()
