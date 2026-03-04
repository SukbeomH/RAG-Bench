"""
Docling subprocess worker.
PDF → per-page Markdown 변환 후 JSON을 stdout으로 출력.

실행: python worker.py <pdf_path>

출력 형식:
    ---OUTPUT_START---
    {"1": "page1 markdown", "2": "page2 markdown", ...}
    ---OUTPUT_END---
"""

from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    if len(sys.argv) < 2:
        print("---OUTPUT_START---")
        print(json.dumps({"error": "pdf_path 인수가 없습니다."}))
        print("---OUTPUT_END---")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as e:
        print("---OUTPUT_START---")
        print(json.dumps({"error": f"docling 미설치: {e}"}))
        print("---OUTPUT_END---")
        sys.exit(1)

    try:
        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.do_table_structure = True
        opts.images_scale = 2.0
        opts.generate_picture_images = True

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

        print(f"[docling-worker] 변환 시작: {pdf_path}", file=sys.stderr)
        result = converter.convert(pdf_path)
        doc = result.document

        page_numbers = sorted(doc.pages.keys()) if doc.pages else []
        pages_output: dict[str, str] = {}

        if page_numbers:
            for page_no in page_numbers:
                md = doc.export_to_markdown(page_no=page_no)
                pages_output[str(page_no)] = md
                print(f"  ✓ 페이지 {page_no}/{len(page_numbers)}", file=sys.stderr)
        else:
            md = doc.export_to_markdown()
            pages_output["1"] = md
            print("  ✓ 단일 페이지 출력", file=sys.stderr)

        print("---OUTPUT_START---")
        print(json.dumps(pages_output, ensure_ascii=False))
        print("---OUTPUT_END---")

    except Exception as e:
        print("---OUTPUT_START---")
        print(json.dumps({"error": f"변환 실패: {e}"}))
        print("---OUTPUT_END---")
        sys.exit(1)


if __name__ == "__main__":
    main()
