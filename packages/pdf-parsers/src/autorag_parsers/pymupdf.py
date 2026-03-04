"""PyMuPDF4LLM backend — fast text extraction for digital PDFs."""

from __future__ import annotations

import time

import pymupdf4llm

from autorag_parsers._protocol import ConversionResult, PageResult
from autorag_parsers.registry import register


@register("pymupdf")
class PyMuPDFParser:
    """PyMuPDF4LLM-based PDF parser. Fast, text-only."""

    @property
    def name(self) -> str:
        return "pymupdf"

    def convert(self, pdf_path: str, **kwargs: object) -> ConversionResult:
        import fitz

        t0 = time.perf_counter()
        pages_raw = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

        # Extract word-level bboxes directly from PyMuPDF
        doc = fitz.open(pdf_path)

        pages: list[PageResult] = []
        for i, chunk in enumerate(pages_raw):
            md = chunk["text"] if isinstance(chunk, dict) else str(chunk)

            # Get word bboxes from the PDF page
            bbox_data = None
            if i < len(doc):
                words = doc[i].get_text("words")  # (x0, y0, x1, y1, "word", ...)
                if words:
                    bbox_data = [
                        {"x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3], "text": w[4]}
                        for w in words
                    ]

            pages.append(
                PageResult(
                    page_num=i + 1,
                    markdown=md,
                    backend="pymupdf",
                    bbox_data=bbox_data,
                    metadata={"source": "pymupdf4llm"},
                )
            )

        doc.close()
        return ConversionResult(
            pdf_path=pdf_path,
            pages=pages,
            total_time_s=time.perf_counter() - t0,
        )
