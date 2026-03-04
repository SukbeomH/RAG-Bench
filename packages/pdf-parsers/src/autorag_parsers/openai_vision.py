"""OpenAI GPT-4o Vision backend — page-by-page VLM extraction."""

from __future__ import annotations

import base64
import os
import time

import fitz  # PyMuPDF

from autorag_parsers._protocol import ConversionResult, PageResult
from autorag_parsers.registry import register

SYSTEM_PROMPT = """**[Role & Objective]**
당신은 복잡한 레이아웃을 가진 PDF 문서를 완벽한 구조의 마크다운(Markdown)으로 변환하는 최고 수준의 문서 분석 및 추출 AI입니다.
당신의 유일한 목표는 제공된 문서 이미지에서 텍스트, 위계 구조(Hierarchy), 시각적 요소, 표(Table), 콜아웃(Callout) 등 모든 문서 요소를 단 하나의 누락이나 왜곡 없이 100% 동일하게 마크다운 문법으로 변환하는 것입니다.

**[Core Principles: 절대 규칙]**
1. **Zero Omission**: 원문의 텍스트, 띄어쓰기, 기호, 오탈자까지 임의로 수정하거나 생략하지 말고 그대로 추출하십시오.
2. **No Yapping**: 인사말, 설명, 요약 등 변환 결과물 외의 어떠한 텍스트(메타 코멘트)도 출력하지 마십시오. 오직 마크다운 본문만 출력합니다.

**[Structural Guidelines: 위계 및 서식 지침]**
1. **문서 위계구조(Hierarchy) 및 헤더 완벽 보존**:
   - 폰트의 크기, 굵기(Bold), 색상, 들여쓰기, 번호 매기기(예: I, A, 1, a, • 등) 등 시각적 단서를 철저히 분석하여 문서의 위계 구조를 파악하십시오.
   - 파악된 위계에 따라 대제목은 `#`, 중제목은 `##`, 소제목은 `###`부터 `######`까지 마크다운 헤더를 정확하게 매핑하십시오.
   - 제목에 포함된 번호나 기호(예: "II. 주류 연구방향", "1)")도 생략하지 말고 포함하십시오.

2. **콜아웃(Callout) 및 인용구/박스 텍스트**:
   - 본문과 분리된 배경색이 있는 박스, 요약 하이라이트, 콜아웃, 또는 인용구 형태의 텍스트는 마크다운 인용문 문법(`> `)을 사용하여 시각적으로 철저히 분리하십시오.
   - 다중 단락으로 이루어진 콜아웃의 경우 모든 줄에 `> `를 적용하십시오.

3. **표 (Table)**:
   - 모든 표는 마크다운 테이블 문법(`|---|---|`)으로 변환하십시오.
   - 병합된 셀(Merged Cells)이 있는 경우, 데이터의 의미가 훼손되지 않도록 병합된 모든 칸에 내용을 반복해서 채워 넣거나, 문맥에 맞게 풀어서 기입하십시오.
   - 셀 내부의 줄바꿈은 반드시 `<br>` 태그를 사용하십시오.
   - 표 내부의 숫자, 소수점, 단위는 반올림하거나 축약하지 마십시오.

4. **이미지, 차트, 다이어그램 (Visual Elements)**:
   - 문서 내의 이미지, 차트, 그래프는 `![설명](image)` 형태의 플레이스홀더로 대체하십시오.
   - `[설명]` 영역에는 다음 요소가 빠짐없이 포함되어야 합니다:
     (1) 차트/이미지의 종류 (막대형, 선형, 모식도 등)
     (2) 제목 및 축 이름 (X축, Y축)
     (3) 범례(Legend) 항목
     (4) 차트에 명시된 핵심 데이터 값과 텍스트
     (5) 보여주고자 하는 데이터의 추세나 시각적 핵심 포인트

5. **목록 (Lists)**:
   - 글머리 기호(Bullet points)와 번호 매기기(Numbered lists)는 원문의 들여쓰기 깊이(Depth) 수준을 정확히 반영하여 `-` 또는 `1.` 형식으로 변환하십시오. 하위 목록은 들여쓰기(Space 2번 또는 4번)를 통해 계층을 명확히 하십시오.

6. **메타데이터 (머리말, 꼬리말, 출처, 페이지 구분)**:
   - 문서 상단/하단에 위치한 출처, 주석, 페이지 번호, 보고서명 등도 빠짐없이 텍스트로 추출하여 제 위치에 배치하십시오.
   - 페이지가 넘어갈 때는 반드시 `---` (수평선)을 추가하고, 그 바로 위에 `` 주석을 달아 페이지 경계를 명확히 표시하십시오."""


def _make_parser(model: str, name: str) -> type:
    """Factory to create OpenAI Vision parser classes for different models."""

    @register(name)
    class _OpenAIVisionParser:
        _model = model
        _name = name

        def __init__(self, api_key: str | None = None, dpi: int = 300):
            self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self._dpi = dpi

        @property
        def name(self) -> str:
            return self._name

        def convert(self, pdf_path: str, **kwargs: object) -> ConversionResult:
            from openai import OpenAI

            t0 = time.perf_counter()
            client = OpenAI(api_key=self._api_key)
            doc = fitz.open(pdf_path)
            scale = self._dpi / 72
            pages: list[PageResult] = []

            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                    img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

                    response = client.chat.completions.create(
                        model=self._model,
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
                            backend=self._name,
                        )
                    )
                except Exception as e:
                    pages.append(
                        PageResult(
                            page_num=page_num + 1,
                            markdown=f"<!-- page error: {e} -->",
                            backend=self._name,
                        )
                    )

            doc.close()
            return ConversionResult(
                pdf_path=pdf_path,
                pages=pages,
                total_time_s=time.perf_counter() - t0,
            )

    _OpenAIVisionParser.__name__ = f"OpenAI_{name.replace('-', '_')}"
    _OpenAIVisionParser.__qualname__ = _OpenAIVisionParser.__name__
    return _OpenAIVisionParser


OpenAIParser = _make_parser("gpt-4o", "openai")
OpenAI41Parser = _make_parser("gpt-4.1", "openai-4.1")
