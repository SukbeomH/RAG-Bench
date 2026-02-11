"""
Parent-Child 청킹 모듈.

Markdown 문서를 상위(Parent) 청크와 하위(Child) 청크로 분할한다.
Parent 청크는 전체 컨텍스트 제공, Child 청크는 검색에 사용.
"""

import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def merge_small_parents(chunks: List[Document], min_size: int = 2000) -> List[Document]:
    """작은 청크들을 병합하여 최소 크기 이상으로 만든다."""
    if not chunks:
        return []
    merged, current = [], None
    for chunk in chunks:
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            for k, v in chunk.metadata.items():
                if k in current.metadata:
                    current.metadata[k] = f"{current.metadata[k]} -> {v}"
                else:
                    current.metadata[k] = v
        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None
    if current:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
        else:
            merged.append(current)
    return merged


def split_large_parents(
    chunks: List[Document],
    max_size: int = 10000,
    chunk_overlap: int = 100,
) -> List[Document]:
    """과도하게 큰 청크를 분할한다."""
    result = []
    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            result.append(chunk)
        else:
            large_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size, chunk_overlap=chunk_overlap
            )
            result.extend(large_splitter.split_documents([chunk]))
    return result


def clean_small_chunks(chunks: List[Document], min_size: int = 2000) -> List[Document]:
    """최소 크기 미만의 청크를 인접 청크에 병합한다."""
    cleaned: List[Document] = []
    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) < min_size:
            if cleaned:
                cleaned[-1].page_content += "\n\n" + chunk.page_content
            elif i < len(chunks) - 1:
                chunks[i + 1].page_content = (
                    chunk.page_content + "\n\n" + chunks[i + 1].page_content
                )
            else:
                cleaned.append(chunk)
        else:
            cleaned.append(chunk)
    return cleaned


def create_parent_child_chunks(
    markdown_dir: str,
    parent_store_path: str,
    min_parent_size: int = 2000,
    max_parent_size: int = 10000,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 100,
) -> Tuple[List[Tuple[str, Document]], List[Document]]:
    """
    Markdown 파일들에서 Parent-Child 청크를 생성한다.

    Args:
        markdown_dir: Markdown 파일 디렉토리.
        parent_store_path: Parent 청크 JSON 저장 디렉토리.
        min_parent_size: Parent 청크 최소 크기.
        max_parent_size: Parent 청크 최대 크기.
        child_chunk_size: Child 청크 크기.
        child_chunk_overlap: Child 청크 오버랩 크기.

    Returns:
        (parent_pairs, child_chunks): Parent (id, doc) 쌍 목록, Child 문서 목록.
    """
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    parent_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
    )

    all_parent_pairs: List[Tuple[str, Document]] = []
    all_child_chunks: List[Document] = []

    md_files = sorted(glob.glob(os.path.join(markdown_dir, "*.md")))
    if not md_files:
        print(f"  No .md files found in {markdown_dir}/")
        return all_parent_pairs, all_child_chunks

    print(f"\nProcessing {len(md_files)} Markdown files...")
    for doc_path_str in md_files:
        doc_path = Path(doc_path_str)
        print(f"  Processing: {doc_path.name}")
        md_text = doc_path.read_text(encoding="utf-8")

        parent_chunks = parent_splitter.split_text(md_text)
        merged_parents = merge_small_parents(parent_chunks, min_parent_size)
        split_parents = split_large_parents(
            merged_parents, max_parent_size, child_chunk_overlap
        )
        cleaned_parents = clean_small_chunks(split_parents, min_parent_size)

        for i, p_chunk in enumerate(cleaned_parents):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update(
                {"source": doc_path.stem + ".pdf", "parent_id": parent_id}
            )
            all_parent_pairs.append((parent_id, p_chunk))
            children = child_splitter.split_documents([p_chunk])
            all_child_chunks.extend(children)

    # Parent 청크를 JSON으로 저장
    parent_store = Path(parent_store_path)
    parent_store.mkdir(parents=True, exist_ok=True)

    # 기존 파일 정리
    for item in os.listdir(parent_store):
        os.remove(os.path.join(parent_store, item))

    for parent_id, doc in all_parent_pairs:
        doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
        filepath = os.path.join(parent_store, f"{parent_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)

    print(
        f"Chunking complete: {len(all_parent_pairs)} parents, "
        f"{len(all_child_chunks)} children"
    )
    return all_parent_pairs, all_child_chunks
