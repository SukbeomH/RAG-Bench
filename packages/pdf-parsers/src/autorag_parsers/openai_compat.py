"""OpenAI-compatible OCR VLM backend — K8s local serving."""

import base64
import os
import time

import fitz  # PyMuPDF

from autorag_parsers._protocol import ConversionResult, PageResult
from autorag_parsers.openai_vision import SYSTEM_PROMPT
from autorag_parsers.registry import register

MODEL_PROFILES: dict[str, tuple[str, str]] = {
    "paddleocr-vl": (
        "paddleocr-vl-1.5",
        "http://paddleocr-vl-server:8000/v1",
    ),
    "deepseek-ocr2": (
        "deepseek-ocr2",
        "http://deepseek-ocr2-server:8000/v1",
    ),
}

RENDER_DPI = 300


def _resolve_endpoint_and_model(
    backend_key: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, str, str]:
    env_endpoint = os.environ.get("OPENSOURCE_VLM_ENDPOINT")
    env_model = os.environ.get("OPENSOURCE_VLM_MODEL")

    default_model, default_url = MODEL_PROFILES.get(
        backend_key, (backend_key, "http://localhost:8000/v1")
    )

    resolved_url = base_url or env_endpoint or default_url
    resolved_model = model or env_model or default_model
    resolved_key = api_key or os.environ.get("OPENSOURCE_VLM_API_KEY", "ollama")

    return resolved_url, resolved_model, resolved_key


def _make_compat_parser(parser_key: str) -> type:
    @register(parser_key)
    class _OpenAICompatParser:
        def __init__(
            self,
            api_key: str | None = None,
            model: str | None = None,
            base_url: str | None = None,
            dpi: int = RENDER_DPI,
        ):
            self._api_key = api_key
            self._model_override = model
            self._base_url_override = base_url
            self._dpi = dpi
            self._backend_key = parser_key

        @property
        def name(self) -> str:
            return self._backend_key

        def convert(self, pdf_path: str, **kwargs: object) -> ConversionResult:
            from openai import OpenAI

            t0 = time.perf_counter()
            url, model, key = _resolve_endpoint_and_model(
                self._backend_key,
                self._model_override,
                self._base_url_override,
                self._api_key,
            )

            client = OpenAI(api_key=key, base_url=url)
            doc = fitz.open(pdf_path)
            scale = self._dpi / 72
            pages: list[PageResult] = []

            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                    img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

                    response = client.chat.completions.create(
                        model=model,
                        temperature=0.1,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "이 PDF 페이지의 모든 내용을 마크다운으로 변환하십시오.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_b64}",
                                            "detail": "high",
                                        },
                                    },
                                ],
                            },
                        ],
                    )

                    md = response.choices[0].message.content or ""
                    pages.append(
                        PageResult(
                            page_num=page_num + 1,
                            markdown=md,
                            backend=self._backend_key,
                        )
                    )
                except Exception as e:
                    pages.append(
                        PageResult(
                            page_num=page_num + 1,
                            markdown=f"<!-- page error: {e} -->",
                            backend=self._backend_key,
                        )
                    )

            doc.close()
            return ConversionResult(
                pdf_path=pdf_path,
                pages=pages,
                total_time_s=time.perf_counter() - t0,
            )

    _OpenAICompatParser.__name__ = f"Compat_{parser_key.replace('-', '_')}"
    _OpenAICompatParser.__qualname__ = _OpenAICompatParser.__name__
    return _OpenAICompatParser


PaddleOCRVLParser = _make_compat_parser("paddleocr-vl")
DeepSeekOCR2Parser = _make_compat_parser("deepseek-ocr2")
