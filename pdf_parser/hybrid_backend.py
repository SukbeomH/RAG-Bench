"""
Hybrid Backend - 페이지 단위 자동 라우팅

MinerU 2.0+ 방식에서 착안:
  - 텍스트 직접 추출 가능 페이지 → Rule-based (PyMuPDF, 빠름)
  - 스캔 / 이미지 중심 페이지   → VLM (OpenAI / Upstage / Gemini, 정확)

기존 smart_router.py의 문서 단위 분류보다 세밀하게 동작.
혼합 문서(일부 페이지는 텍스트, 일부는 스캔)에서 속도와 정확도를 동시에 확보.

VLM 백엔드:
  - "openai"   : GPT-4o Vision (base64 인코딩 방식)
  - "upstage"  : Upstage Document Parse API (페이지별 PDF 전송)
  - "gemini"   : Google Gemini (레거시, category3_complex.py 필요)
"""

from __future__ import annotations

import io
import os
import base64
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

# ── 판별 임계값 ────────────────────────────────────────────────────────────────
TEXT_THRESHOLD  = 50    # 페이지 문자 수가 이 미만이면 스캔/이미지 페이지로 판단
IMAGE_THRESHOLD = 3     # 이미지 수가 이 이상이면 이미지 중심 페이지로 판단
RENDER_DPI      = 300   # VLM 전송용 이미지 렌더링 해상도


# ── 데이터 구조 ────────────────────────────────────────────────────────────────

@dataclass
class PageResult:
    page_num: int
    backend: str          # "text" | "vlm"
    content: str
    char_count: int = 0
    image_count: int = 0


@dataclass
class ConversionReport:
    pdf_path: str
    total_pages: int
    text_pages: list[int] = field(default_factory=list)
    vlm_pages: list[int] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)

    @property
    def text_ratio(self) -> float:
        return len(self.text_pages) / max(self.total_pages, 1)

    def summary(self) -> str:
        return (
            f"총 {self.total_pages}p | "
            f"Rule-based {len(self.text_pages)}p | "
            f"VLM {len(self.vlm_pages)}p | "
            f"오류 {len(self.errors)}p"
        )


# ── 페이지 분류 ────────────────────────────────────────────────────────────────

def classify_page(page: fitz.Page) -> tuple[str, int, int]:
    """
    단일 페이지를 분석해 처리 방법 결정.

    Returns:
        (backend, char_count, image_count)
        backend: "text" | "vlm"
    """
    text = page.get_text()
    char_count = len(text.strip())
    image_count = len(page.get_images())

    if char_count < TEXT_THRESHOLD or image_count >= IMAGE_THRESHOLD:
        return "vlm", char_count, image_count
    return "text", char_count, image_count


# ── Rule-based 텍스트 추출 ─────────────────────────────────────────────────────

def extract_text_page(page: fitz.Page) -> str:
    """PyMuPDF로 페이지 텍스트를 Markdown 형태로 추출."""
    blocks = page.get_text("blocks")
    lines: list[str] = []
    for block in sorted(blocks, key=lambda b: (b[1], b[0])):
        text = block[4].strip()
        if text:
            lines.append(text)
    return "\n\n".join(lines)


# ── VLM 추출 — OpenAI ──────────────────────────────────────────────────────────

def _extract_vlm_openai(page: fitz.Page, api_key: str, model: str) -> str:
    """페이지를 이미지로 렌더링 후 GPT-4o Vision으로 Markdown 변환."""
    from openai import OpenAI
    from category3_openai import SYSTEM_PROMPT

    scale = RENDER_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Convert this PDF page to clean, structured markdown. "
                        "Extract all text, describe images, and preserve the layout."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "high",
                    }},
                ],
            },
        ],
    )
    return response.choices[0].message.content


# ── VLM 추출 — Upstage (페이지 단위 PDF 전송) ─────────────────────────────────

def _extract_vlm_upstage(page: fitz.Page, api_key: str, mode: str = "auto") -> str:
    """
    단일 페이지를 1-page PDF로 변환 후 Upstage Document Parse API 전송.

    Upstage는 이미지가 아닌 PDF 포맷을 받으므로 페이지를 단일 PDF로 재구성.
    """
    import requests

    # 단일 페이지 PDF 생성 (메모리 버퍼)
    tmp_doc = fitz.open()
    tmp_doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
    pdf_bytes = tmp_doc.tobytes()
    tmp_doc.close()

    headers = {"Authorization": f"Bearer {api_key}"}
    data: dict = {
        "model": "document-parse",
        "output_formats": '["markdown"]',
        "ocr": "auto",
        "coordinates": "false",
    }
    if mode and mode != "auto":
        data["mode"] = mode

    response = requests.post(
        "https://api.upstage.ai/v1/document-digitization",
        headers=headers,
        files={"document": (f"page_{page.number + 1}.pdf", pdf_bytes, "application/pdf")},
        data=data,
    )
    response.raise_for_status()
    result = response.json()

    content = result.get("content", {})
    return content.get("markdown", "") or content.get("text", "")


# ── VLM 추출 — Gemini (레거시) ────────────────────────────────────────────────

def _extract_vlm_gemini(page: fitz.Page, api_key: str, model: str) -> str:
    """페이지를 이미지로 렌더링 후 Gemini로 Markdown 변환 (레거시)."""
    from google import genai
    from google.genai import types
    from category3_complex import SYSTEM_PROMPT

    scale = RENDER_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img_data = pix.tobytes("png")

    client = genai.Client(api_key=api_key)
    image = types.Part.from_bytes(data=img_data, mime_type="image/png")
    response = client.models.generate_content(
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, temperature=0.1,
        ),
        model=model,
        contents=[
            "Convert this PDF page to clean, structured markdown. "
            "Extract all text, describe images, and preserve the layout.",
            image,
        ],
    )
    return response.text


# ── 통합 VLM 디스패처 ─────────────────────────────────────────────────────────

def extract_vlm_page(
    page: fitz.Page,
    vlm_backend: str,
    api_key: str,
    model: str = "",
    upstage_mode: str = "auto",
) -> str:
    """
    VLM 백엔드 디스패처.

    Args:
        vlm_backend: "openai" | "upstage" | "upstage-enhanced" | "gemini"
        api_key:     해당 서비스 API 키
        model:       모델명 (openai: gpt-4o, gemini: gemini-2.5-flash)
        upstage_mode: Upstage 처리 모드 (auto|standard|enhanced)
    """
    if vlm_backend == "openai":
        return _extract_vlm_openai(page, api_key, model or "gpt-4o")
    elif vlm_backend in ("upstage", "upstage-enhanced"):
        mode = "enhanced" if vlm_backend == "upstage-enhanced" else upstage_mode
        return _extract_vlm_upstage(page, api_key, mode=mode)
    elif vlm_backend == "gemini":
        return _extract_vlm_gemini(page, api_key, model or "gemini-2.5-flash")
    else:
        raise ValueError(f"지원하지 않는 VLM 백엔드: {vlm_backend}")


# ── 메인 변환 ─────────────────────────────────────────────────────────────────

def convert_pdf(
    pdf_path: str,
    output_path: str,
    vlm_backend: str = "openai",
    api_key: str | None = None,
    model: str = "",
    upstage_mode: str = "auto",
    verbose: bool = True,
) -> ConversionReport:
    """
    PDF를 페이지별로 분류해 최적 백엔드로 변환 후 단일 Markdown 파일 저장.

    Args:
        pdf_path:     PDF 파일 경로
        output_path:  출력 Markdown 파일 경로
        vlm_backend:  VLM 백엔드 ("openai" | "upstage" | "upstage-enhanced" | "gemini")
        api_key:      API 키 (VLM 페이지 존재 시 필요)
        model:        모델명 (openai: gpt-4o/gpt-4.1, gemini: gemini-2.5-flash)
        upstage_mode: Upstage 처리 모드 (auto|standard|enhanced)
        verbose:      페이지별 진행 출력 여부

    Returns:
        ConversionReport
    """
    doc = fitz.open(pdf_path)
    report = ConversionReport(pdf_path=pdf_path, total_pages=doc.page_count)
    results: list[PageResult] = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        backend, char_count, image_count = classify_page(page)
        display_num = page_num + 1

        try:
            if backend == "text":
                content = extract_text_page(page)
                report.text_pages.append(display_num)
            else:
                if not api_key:
                    raise ValueError(
                        f"페이지 {display_num}는 VLM 처리가 필요하지만 "
                        "api_key가 없습니다."
                    )
                content = extract_vlm_page(
                    page, vlm_backend, api_key,
                    model=model, upstage_mode=upstage_mode,
                )
                report.vlm_pages.append(display_num)

            results.append(PageResult(
                page_num=display_num,
                backend=backend,
                content=content,
                char_count=char_count,
                image_count=image_count,
            ))

            if verbose:
                label = "[text]" if backend == "text" else f"[{vlm_backend}]"
                print(
                    f"  {label} p{display_num:>3}/{doc.page_count} | "
                    f"chars={char_count:>5} imgs={image_count}"
                )

        except Exception as e:
            report.errors[display_num] = str(e)
            results.append(PageResult(
                page_num=display_num,
                backend=backend,
                content=f"<!-- 페이지 {display_num} 처리 오류: {e} -->",
            ))
            if verbose:
                print(f"  ✗ p{display_num} 오류: {e}")

    doc.close()

    combined = "\n\n---\n\n".join(
        f"<!-- page={r.page_num} backend={r.backend} -->\n\n{r.content}"
        for r in results
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(combined, encoding="utf-8")

    return report


def convert_folder(
    pdf_folder: str,
    output_folder: str,
    vlm_backend: str = "openai",
    api_key: str | None = None,
    model: str = "",
) -> None:
    """폴더 내 모든 PDF를 Hybrid backend로 일괄 변환."""
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠ PDF 파일 없음: {pdf_folder}")
        return

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    total_reports: list[ConversionReport] = []

    for pdf_file in pdf_files:
        print(f"\n{pdf_file.name}")
        output_path = os.path.join(output_folder, f"{pdf_file.stem}.md")
        report = convert_pdf(
            str(pdf_file), output_path,
            vlm_backend=vlm_backend, api_key=api_key, model=model,
        )
        total_reports.append(report)
        print(f"  → {report.summary()}")

    print("\n=== 일괄 변환 완료 ===")
    print(f"  Rule-based 페이지: {sum(len(r.text_pages) for r in total_reports)}")
    print(f"  VLM 페이지:        {sum(len(r.vlm_pages) for r in total_reports)}")
    print(f"  오류 페이지:       {sum(len(r.errors) for r in total_reports)}")
    print(f"  출력 위치:         '{output_folder}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid PDF → Markdown 변환")
    parser.add_argument("--input",   default="./pdfs")
    parser.add_argument("--output",  default="./md_output/hybrid")
    parser.add_argument("--vlm",     default="openai",
                        choices=["openai", "upstage", "upstage-enhanced", "gemini"])
    parser.add_argument("--model",   default="")
    args = parser.parse_args()

    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UPSTAGE_API_KEY")
    convert_folder(args.input, args.output, vlm_backend=args.vlm, api_key=key, model=args.model)
