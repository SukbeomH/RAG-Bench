"""
Hybrid Backend - 페이지 단위 자동 라우팅

MinerU 2.0+ 방식에서 착안:
  - 텍스트 직접 추출 가능 페이지 → Rule-based (PyMuPDF, 빠름)
  - 스캔 / 이미지 중심 페이지   → VLM (Gemini, 정확)

기존 smart_router.py의 문서 단위 분류보다 세밀하게 동작.
혼합 문서(일부 페이지는 텍스트, 일부는 스캔)에서 속도와 정확도를 동시에 확보.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from google import genai
from google.genai import types

from category3_complex import SYSTEM_PROMPT

# ── 판별 임계값 ────────────────────────────────────────────────────────────────
TEXT_THRESHOLD = 50     # 페이지에서 추출된 문자 수가 이 미만이면 스캔/이미지 페이지로 판단
IMAGE_THRESHOLD = 3     # 이미지 수가 이 이상이면 이미지 중심 페이지로 판단
RENDER_DPI = 300        # VLM 전송용 이미지 렌더링 해상도


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
    """
    PyMuPDF로 페이지 텍스트를 Markdown 형태로 추출.
    블록 단위로 읽어 단락 구조 보존.
    """
    blocks = page.get_text("blocks")
    lines: list[str] = []

    for block in sorted(blocks, key=lambda b: (b[1], b[0])):  # y → x 순 정렬
        text = block[4].strip()
        if not text:
            continue
        lines.append(text)

    return "\n\n".join(lines)


# ── VLM 추출 ──────────────────────────────────────────────────────────────────

def extract_vlm_page(
    page: fitz.Page,
    client: genai.Client,
    model: str,
) -> str:
    """
    페이지를 이미지로 렌더링 후 VLM으로 Markdown 변환.
    """
    scale = RENDER_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img_data = pix.tobytes("png")

    image = types.Part.from_bytes(data=img_data, mime_type="image/png")

    response = client.models.generate_content(
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
        ),
        model=model,
        contents=[
            "Convert this PDF page to clean, structured markdown. "
            "Extract all text, describe images, and preserve the layout.",
            image,
        ],
    )
    return response.text


# ── 메인 변환 ─────────────────────────────────────────────────────────────────

def convert_pdf(
    pdf_path: str,
    output_path: str,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    verbose: bool = True,
) -> ConversionReport:
    """
    PDF를 페이지별로 분류해 최적 백엔드로 변환 후 단일 Markdown 파일 저장.

    Args:
        pdf_path:    PDF 파일 경로
        output_path: 출력 Markdown 파일 경로
        api_key:     Gemini API 키 (VLM 페이지 존재 시 필요)
        model:       Gemini 모델명
        verbose:     페이지별 진행 출력 여부

    Returns:
        ConversionReport (백엔드 선택 이력, 오류 등)
    """
    doc = fitz.open(pdf_path)
    report = ConversionReport(pdf_path=pdf_path, total_pages=doc.page_count)

    # VLM 클라이언트는 실제로 필요할 때만 초기화
    client: genai.Client | None = None

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
                if client is None:
                    if not api_key:
                        raise ValueError(
                            f"페이지 {display_num}는 VLM 처리가 필요하지만 "
                            "api_key가 없습니다. GEMINI_API_KEY를 설정하세요."
                        )
                    client = genai.Client(api_key=api_key)
                content = extract_vlm_page(page, client, model)
                report.vlm_pages.append(display_num)

            results.append(PageResult(
                page_num=display_num,
                backend=backend,
                content=content,
                char_count=char_count,
                image_count=image_count,
            ))

            if verbose:
                backend_label = "[text]" if backend == "text" else "[vlm ]"
                print(
                    f"  {backend_label} p{display_num:>3}/{doc.page_count} | "
                    f"chars={char_count:>5} imgs={image_count}"
                )

        except Exception as e:
            report.errors[display_num] = str(e)
            results.append(PageResult(
                page_num=display_num,
                backend=backend,
                content=f"<!-- 페이지 {display_num} 처리 오류: {e} -->",
            ))
            print(f"  ✗ p{display_num} 오류: {e}")

    doc.close()

    # Markdown 파일 저장
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
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
) -> None:
    """
    폴더 내 모든 PDF를 Hybrid backend로 일괄 변환.

    Args:
        pdf_folder:    PDF 파일이 있는 폴더 경로
        output_folder: Markdown 출력 폴더 경로
        api_key:       Gemini API 키
        model:         Gemini 모델명
    """
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠ PDF 파일 없음: {pdf_folder}")
        return

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    total_reports: list[ConversionReport] = []

    for pdf_file in pdf_files:
        print(f"\n📄 {pdf_file.name}")
        output_path = os.path.join(output_folder, f"{pdf_file.stem}.md")

        report = convert_pdf(
            str(pdf_file),
            output_path,
            api_key=api_key,
            model=model,
        )
        total_reports.append(report)
        print(f"  → {report.summary()}")

    # 전체 요약
    print("\n=== 일괄 변환 완료 ===")
    all_text = sum(len(r.text_pages) for r in total_reports)
    all_vlm = sum(len(r.vlm_pages) for r in total_reports)
    all_errors = sum(len(r.errors) for r in total_reports)
    print(f"  Rule-based 페이지: {all_text}")
    print(f"  VLM 페이지:        {all_vlm}")
    print(f"  오류 페이지:       {all_errors}")
    print(f"  출력 위치:         '{output_folder}'")


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    convert_folder("./pdfs", "./md_output/hybrid", api_key=api_key)
