"""
PDF → Markdown 변환 모듈.

pymupdf4llm을 사용하여 PDF 파일을 Markdown으로 변환한다.
"""

import glob
from pathlib import Path
from typing import List, Optional


def pdf_to_markdown(pdf_path: str, output_dir: str) -> str:
    """
    단일 PDF 파일을 Markdown으로 변환.

    Args:
        pdf_path: 입력 PDF 경로.
        output_dir: 출력 Markdown 디렉토리.

    Returns:
        출력 Markdown 파일 경로.
    """
    import pymupdf4llm

    _pdf_path = Path(pdf_path)
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)

    md_text = pymupdf4llm.to_markdown(str(_pdf_path))
    output_path = _output_dir / f"{_pdf_path.stem}.md"
    output_path.write_text(md_text, encoding="utf-8")

    print(f"  {_pdf_path.name} → {output_path.name} ({len(md_text):,} chars)")
    return str(output_path)


def pdfs_to_markdowns(
    docs_dir: str,
    output_dir: str,
    pattern: str = "*.pdf",
) -> List[str]:
    """
    디렉토리 내 모든 PDF를 Markdown으로 일괄 변환.

    Args:
        docs_dir: PDF 파일이 있는 디렉토리.
        output_dir: 출력 디렉토리.
        pattern: glob 패턴 (기본: *.pdf).

    Returns:
        생성된 Markdown 파일 경로 목록.
    """
    import pymupdf4llm

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
        md_path = pdf_to_markdown(pdf_path, str(output_dir))
        results.append(md_path)

    print(f"Conversion complete: {len(results)} files")
    return results
