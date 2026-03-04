"""Docling backend — OCR + layout structure recognition."""

from __future__ import annotations

import time
from typing import Any

from autorag_parsers._protocol import ConversionResult, PageResult
from autorag_parsers.registry import register


@register("docling")
class DoclingParser:
    """Docling-based PDF parser with OCR and table structure recognition."""

    def __init__(
        self,
        do_ocr: bool = True,
        do_table_structure: bool = True,
        images_scale: float = 2.0,
    ):
        self._do_ocr = do_ocr
        self._do_table_structure = do_table_structure
        self._images_scale = images_scale
        self._converter: Any = None

    @property
    def name(self) -> str:
        return "docling"

    def _get_converter(self) -> Any:
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            opts = PdfPipelineOptions()
            opts.do_ocr = self._do_ocr
            opts.do_table_structure = self._do_table_structure
            opts.images_scale = self._images_scale
            opts.generate_picture_images = True

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        return self._converter

    def convert(self, pdf_path: str, **kwargs: object) -> ConversionResult:
        t0 = time.perf_counter()
        converter = self._get_converter()
        result = converter.convert(pdf_path)
        doc = result.document

        # Per-page markdown export via docling-core API
        page_numbers = sorted(doc.pages.keys()) if doc.pages else []

        pages: list[PageResult] = []
        if page_numbers:
            for page_no in page_numbers:
                md = doc.export_to_markdown(page_no=page_no)
                pages.append(
                    PageResult(
                        page_num=page_no,
                        markdown=md,
                        backend="docling",
                        metadata={"source": "docling"},
                    )
                )
        else:
            # Fallback: single-page output if page info unavailable
            md = doc.export_to_markdown()
            pages.append(
                PageResult(
                    page_num=1,
                    markdown=md,
                    backend="docling",
                    metadata={"source": "docling"},
                )
            )

        return ConversionResult(
            pdf_path=pdf_path,
            pages=pages,
            total_time_s=time.perf_counter() - t0,
        )
