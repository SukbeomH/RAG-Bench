"""
Category 1: Simple PDFs - Fast Text Extraction

대상: 디지털 PDF (스캔 아님), 텍스트 위주, 단순 레이아웃
도구: PyMuPDF4LLM
"""

import pymupdf4llm
import pathlib


def convert_pdf(pdf_path: str, output_path: str) -> str:
    """
    단일 PDF 파일을 Markdown으로 변환.

    Args:
        pdf_path: PDF 파일 경로
        output_path: 출력 Markdown 파일 경로

    Returns:
        변환된 Markdown 텍스트
    """
    md_text = pymupdf4llm.to_markdown(pdf_path)

    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md_text, encoding="utf-8")

    print(f"✓ 변환 완료: {pdf_path} → {output_path}")
    return md_text


def convert_folder(pdf_folder: str, output_folder: str) -> None:
    """
    폴더 내 모든 PDF를 Markdown으로 일괄 변환.

    Args:
        pdf_folder: PDF 파일이 있는 폴더 경로
        output_folder: Markdown 출력 폴더 경로
    """
    pdf_path = pathlib.Path(pdf_folder)
    output_path = pathlib.Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠ PDF 파일 없음: {pdf_folder}")
        return

    for pdf_file in pdf_files:
        try:
            md_text = pymupdf4llm.to_markdown(str(pdf_file))
            out_file = output_path / f"{pdf_file.stem}.md"
            out_file.write_text(md_text, encoding="utf-8")
            print(f"✓ 변환 완료: {pdf_file.name}")
        except Exception as e:
            print(f"✗ 오류 ({pdf_file.name}): {e}")

    print(f"\n변환 완료. 출력 위치: '{output_folder}'")


if __name__ == "__main__":
    # 사용 예시
    convert_folder("./simple_pdfs", "./md_output/simple")
