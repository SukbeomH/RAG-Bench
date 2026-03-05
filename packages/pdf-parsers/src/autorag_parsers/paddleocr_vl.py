"""PaddleOCR-VL backend — native pipeline via subprocess bridge."""

from __future__ import annotations

import time

from autorag_parsers._protocol import ConversionResult, PageResult
from autorag_parsers.registry import register


@register("paddleocr-vl")
class PaddleOCRVLParser:
    """PaddleOCR-VL native pipeline (layout analysis + VLM + markdownify)."""

    @property
    def name(self) -> str:
        return "paddleocr-vl"

    def convert(self, pdf_path: str, **kwargs: object) -> ConversionResult:
        from isolated_backends.paddleocr.bridge import convert_pdf

        t0 = time.perf_counter()
        combined = convert_pdf(pdf_path, output_path="/dev/null")

        # bridge returns "# Page N\n\n...\n\n---\n\n# Page N+1\n\n..." format
        import re

        parts = re.split(r"^# Page \d+\s*", combined, flags=re.MULTILINE)
        # Extract page numbers from headers
        page_nums = [
            int(m) for m in re.findall(r"^# Page (\d+)", combined, re.MULTILINE)
        ]

        pages: list[PageResult] = []
        for num, content in zip(page_nums, parts[1:]):  # parts[0] is empty
            # Remove trailing --- separator
            content = re.sub(r"\n*---\s*$", "", content).strip()
            pages.append(
                PageResult(
                    page_num=num,
                    markdown=content,
                    backend="paddleocr-vl",
                    metadata={"source": "paddleocr-vl-pipeline"},
                )
            )

        return ConversionResult(
            pdf_path=pdf_path,
            pages=pages,
            total_time_s=time.perf_counter() - t0,
        )
