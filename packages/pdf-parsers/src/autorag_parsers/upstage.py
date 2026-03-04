"""Upstage Document Parse API backend — Korean document specialist."""

import os
import time
from pathlib import Path

import fitz  # PyMuPDF (for page splitting)
import requests

from autorag_parsers._protocol import ConversionResult, PageResult
from autorag_parsers.registry import register

UPSTAGE_API_URL = "https://api.upstage.ai/v1/document-digitization"
MAX_FILE_SIZE_MB = 30
PAGE_DELAY_S = 5.0


def _call_api(
    pdf_bytes: bytes,
    filename: str,
    api_key: str,
    model: str,
    output_format: str,
    ocr: str,
    mode: str,
) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    data: dict = {
        "model": model,
        "output_formats": f'["{output_format}"]',
        "ocr": ocr,
        "coordinates": "false",
    }
    if mode and mode != "auto":
        data["mode"] = mode

    response = requests.post(
        UPSTAGE_API_URL,
        headers=headers,
        files={"document": (filename, pdf_bytes, "application/pdf")},
        data=data,
    )
    response.raise_for_status()
    return response.json()


def _parse_response(
    result: dict, output_format: str, page_offset: int = 0
) -> dict[int, str]:
    content = result.get("content", {})
    markdown_text = content.get(output_format, "") or content.get("text", "")

    pages_data = result.get("pages", [])
    if pages_data:
        return {
            page_info.get("page", idx + 1) + page_offset: (
                page_info.get("content", {}).get(output_format, "")
                or page_info.get("content", {}).get("text", "")
            )
            for idx, page_info in enumerate(pages_data)
        }
    return {1 + page_offset: markdown_text}


def _make_upstage_parser(parser_name: str, default_mode: str) -> type:
    @register(parser_name)
    class _UpstageParser:
        def __init__(
            self,
            api_key: str | None = None,
            model: str = "document-parse",
            output_format: str = "markdown",
            ocr: str = "auto",
            mode: str | None = None,
        ):
            self._api_key = api_key or os.environ.get("UPSTAGE_API_KEY", "")
            self._model = model
            self._output_format = output_format
            self._ocr = ocr
            self._mode = mode or default_mode
            self._name = parser_name

        @property
        def name(self) -> str:
            return self._name

        def convert(self, pdf_path: str, **kwargs: object) -> ConversionResult:
            t0 = time.perf_counter()
            file_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)

            if file_size_mb > MAX_FILE_SIZE_MB:
                raw = self._convert_by_pages(pdf_path)
            else:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                raw = _call_api(
                    pdf_bytes,
                    Path(pdf_path).name,
                    self._api_key,
                    self._model,
                    self._output_format,
                    self._ocr,
                    self._mode,
                )
                raw = _parse_response(raw, self._output_format)

            pages = [
                PageResult(
                    page_num=pn,
                    markdown=md,
                    backend=self._name,
                )
                for pn, md in sorted(raw.items())
            ]

            return ConversionResult(
                pdf_path=pdf_path,
                pages=pages,
                total_time_s=time.perf_counter() - t0,
            )

        def _convert_by_pages(self, pdf_path: str) -> dict[int, str]:
            doc = fitz.open(pdf_path)
            all_pages: dict[int, str] = {}

            for page_num in range(doc.page_count):
                if page_num > 0 and PAGE_DELAY_S > 0:
                    time.sleep(PAGE_DELAY_S)

                tmp_doc = fitz.open()
                tmp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                pdf_bytes = tmp_doc.tobytes()
                tmp_doc.close()

                try:
                    result = _call_api(
                        pdf_bytes,
                        f"page_{page_num + 1}.pdf",
                        self._api_key,
                        self._model,
                        self._output_format,
                        self._ocr,
                        self._mode,
                    )
                    pages = _parse_response(result, self._output_format)
                    content = "\n\n".join(pages.values())
                    all_pages[page_num + 1] = content
                except Exception as e:
                    all_pages[page_num + 1] = f"<!-- page {page_num + 1} error: {e} -->"

            doc.close()
            return all_pages

    _UpstageParser.__name__ = f"Upstage_{parser_name.replace('-', '_')}"
    _UpstageParser.__qualname__ = _UpstageParser.__name__
    return _UpstageParser


UpstageParser = _make_upstage_parser("upstage", "auto")
UpstageEnhancedParser = _make_upstage_parser("upstage-enhanced", "enhanced")
