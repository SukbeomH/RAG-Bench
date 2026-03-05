"""OpenAI-compatible OCR VLM backend — K8s local serving."""

import base64
import os
import time

import fitz  # PyMuPDF

from autorag_parsers._protocol import ConversionResult, PageResult
from autorag_parsers.registry import register

_OPENSOURCE_SYSTEM_PROMPT = """\
You are an expert document parser specializing in converting PDF pages to markdown format.

**Your task:**
Extract ALL content from the provided page image and return it as clean, well-structured markdown.

**Text Extraction Rules:**
1. Preserve the EXACT text as written (including typos, formatting, special characters)
2. Maintain the logical reading order (top-to-bottom, left-to-right)
3. Preserve hierarchical structure using appropriate markdown headers (#, ##, ###)
4. Keep paragraph breaks and line spacing as they appear
5. Use markdown lists (-, *, 1.) for bullet points and numbered lists
6. Preserve text emphasis: **bold**, *italic*, `code`
7. For multi-column layouts, extract left column first, then right column

**Tables:**
- Convert all tables to markdown table format
- Preserve column alignment and structure
- Use | for columns and - for headers

**Images, Diagrams, Charts:**
- Insert markdown image placeholder: `![Description](image)`
- Provide a detailed description of the visual content

**Quality Guidelines:**
- DO NOT add explanations, comments, or meta-information
- DO NOT skip or summarize content
- DO NOT invent or hallucinate text not present in the image
- Output ONLY the markdown content, nothing else

**Output Format:**
Return raw markdown with no wrapper, no code blocks, no explanations.
Start immediately with the page content."""

_OPENSOURCE_USER_PROMPT = (
    "Convert this PDF page to clean, structured markdown. "
    "Extract all text, describe images, and preserve the layout."
)

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
                        max_tokens=4096,
                        messages=[
                            {"role": "system", "content": _OPENSOURCE_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": _OPENSOURCE_USER_PROMPT,
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


DeepSeekOCR2Parser = _make_compat_parser("deepseek-ocr2")
