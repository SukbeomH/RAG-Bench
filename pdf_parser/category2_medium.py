"""
Category 2: Medium Complexity PDFs - OCR + Structure Recognition

대상: 스캔 문서, 표가 있는 문서, 멀티컬럼 레이아웃
도구: Docling
"""

from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def build_converter(
    do_ocr: bool = True,
    do_table_structure: bool = True,
    images_scale: float = 2.0,
) -> DocumentConverter:
    """
    Docling DocumentConverter 생성.

    Args:
        do_ocr: OCR 활성화 여부 (스캔 문서에 필요)
        do_table_structure: 표 구조 인식 여부
        images_scale: 이미지 해상도 배율
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure
    pipeline_options.images_scale = images_scale
    pipeline_options.generate_picture_images = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def convert_pdf(pdf_path: str, output_path: str, converter: DocumentConverter = None) -> str:
    """
    단일 PDF 파일을 Markdown으로 변환.

    Args:
        pdf_path: PDF 파일 경로
        output_path: 출력 Markdown 파일 경로
        converter: 재사용할 DocumentConverter (없으면 새로 생성)

    Returns:
        변환된 Markdown 텍스트
    """
    if converter is None:
        converter = build_converter()

    result = converter.convert(pdf_path)
    markdown_content = result.document.export_to_markdown()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown_content, encoding="utf-8")

    print(f"✓ 변환 완료: {pdf_path} → {output_path}")
    return markdown_content


def convert_folder(pdf_folder: str, output_folder: str) -> None:
    """
    폴더 내 모든 PDF를 Markdown으로 일괄 변환.

    Args:
        pdf_folder: PDF 파일이 있는 폴더 경로
        output_folder: Markdown 출력 폴더 경로
    """
    converter = build_converter()

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠ PDF 파일 없음: {pdf_folder}")
        return

    for pdf_file in pdf_files:
        try:
            result = converter.convert(str(pdf_file))
            markdown_content = result.document.export_to_markdown()

            out_file = output_path / f"{pdf_file.stem}.md"
            out_file.write_text(markdown_content, encoding="utf-8")
            print(f"✓ 변환 완료: {pdf_file.name}")
        except Exception as e:
            print(f"✗ 오류 ({pdf_file.name}): {e}")

    print(f"\n변환 완료. 출력 위치: '{output_folder}'")


if __name__ == "__main__":
    # 사용 예시
    convert_folder("./medium_pdfs", "./md_output/medium")
