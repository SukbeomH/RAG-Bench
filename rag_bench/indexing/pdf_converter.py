"""
PDF → Markdown 변환 모듈.

pymupdf4llm을 사용하여 PDF 파일을 Markdown으로 변환한다.
"""

import glob
import random
from pathlib import Path
from typing import List, Optional


def pdf_to_markdown(
    pdf_path: str,
    output_dir: str,
    sample_pages: bool = False,
    page_sample_ratio: float = 0.1,
    max_sample_pages: int = 5,
) -> str:
    """
    단일 PDF 파일을 Markdown으로 변환.

    Args:
        pdf_path: 입력 PDF 경로.
        output_dir: 출력 Markdown 디렉토리.
        sample_pages: True이면 페이지 샘플링 적용.
        page_sample_ratio: 샘플링 비율 (기본: 10%).
        max_sample_pages: 최대 샘플 페이지 수 (기본: 5).

    Returns:
        출력 Markdown 파일 경로.
    """
    import fitz
    import pymupdf4llm

    _pdf_path = Path(pdf_path)
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)

    if sample_pages:
        doc = fitz.open(str(_pdf_path))
        total_pages = doc.page_count
        doc.close()

        sample_count = min(max(1, int(total_pages * page_sample_ratio)), max_sample_pages)
        sampled_pages = sorted(random.sample(range(total_pages), sample_count))

        md_text = pymupdf4llm.to_markdown(str(_pdf_path), pages=sampled_pages)
        print(f"  {_pdf_path.name} → {_output_dir.name}/ "
              f"(샘플링: {sample_count}/{total_pages}페이지, {len(md_text):,} chars)")
    else:
        md_text = pymupdf4llm.to_markdown(str(_pdf_path))
        print(f"  {_pdf_path.name} → {_output_dir.name}/ ({len(md_text):,} chars)")

    output_path = _output_dir / f"{_pdf_path.stem}.md"
    output_path.write_text(md_text, encoding="utf-8")

    return str(output_path)


def pdfs_to_markdowns(
    docs_dir: str,
    output_dir: str,
    pattern: str = "*.pdf",
    sample_pages: bool = False,
    page_sample_ratio: float = 0.1,
    max_sample_pages: int = 5,
) -> List[str]:
    """
    디렉토리 내 모든 PDF를 Markdown으로 일괄 변환.

    Args:
        docs_dir: PDF 파일이 있는 디렉토리.
        output_dir: 출력 디렉토리.
        pattern: glob 패턴 (기본: *.pdf).
        sample_pages: True이면 페이지 샘플링 적용.
        page_sample_ratio: 샘플링 비율 (기본: 10%).
        max_sample_pages: 최대 샘플 페이지 수 (기본: 5).

    Returns:
        생성된 Markdown 파일 경로 목록.
    """
    _docs_dir = Path(docs_dir)
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(glob.glob(str(_docs_dir / pattern)))
    if not pdf_files:
        print(f"  No PDF files found in {docs_dir}/")
        return []

    print(f"\nConverting {len(pdf_files)} PDFs to Markdown...")
    results = []
    for pdf_path in pdf_files:
        md_path = pdf_to_markdown(
            pdf_path,
            str(output_dir),
            sample_pages=sample_pages,
            page_sample_ratio=page_sample_ratio,
            max_sample_pages=max_sample_pages,
        )
        results.append(md_path)

    print(f"Conversion complete: {len(results)} files")
    return results
