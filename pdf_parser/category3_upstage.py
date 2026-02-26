"""
Category 3: Complex PDFs - Upstage Document Parse API

대상: 차트/다이어그램/이미지가 핵심인 문서, 복잡한 레이아웃, 한국어 문서
도구: Upstage Document Parse API (OCR + 레이아웃 인식 특화)

Upstage Document Parse는 이미지 기반 VLM이 아닌 전용 문서 파싱 API로,
한국어 문서와 표/레이아웃 인식에 특히 강점이 있음.

파일 크기 초과(413) 대응:
  MAX_FILE_SIZE_MB 초과 시 페이지 단위로 분할해 각각 전송 후 합산.
  hybrid_backend.py의 _extract_vlm_upstage()와 동일 방식.
"""

import os
from pathlib import Path

import fitz  # PyMuPDF (페이지 분할용)
import requests

# Upstage Document Parse API 엔드포인트
UPSTAGE_API_URL = "https://api.upstage.ai/v1/document-digitization"

# 파일 크기 임계값: 이 이상이면 페이지 단위 분할 처리
MAX_FILE_SIZE_MB = 30


def _call_api(
    pdf_bytes: bytes,
    filename: str,
    api_key: str,
    model: str,
    output_format: str,
    ocr: str,
    mode: str,
) -> dict:
    """단일 PDF bytes → Upstage API 호출 → 응답 dict 반환."""
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


def _parse_response(result: dict, output_format: str, page_offset: int = 0) -> dict[int, str]:
    """
    Upstage 응답에서 {페이지번호: Markdown} 추출.

    page_offset: 분할 처리 시 실제 페이지 번호 보정용 (0-based)
    """
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
    else:
        return {1 + page_offset: markdown_text}


def convert_pdf(
    pdf_path: str,
    api_key: str,
    model: str = "document-parse",
    output_format: str = "markdown",
    ocr: str = "auto",
    mode: str = "auto",
) -> dict[int, str]:
    """
    Upstage Document Parse API로 PDF → Markdown 변환.

    파일이 MAX_FILE_SIZE_MB 초과 시 페이지 단위로 자동 분할 처리.

    Args:
        pdf_path: PDF 파일 경로
        api_key: Upstage API 키
        model: 사용할 모델
               - "document-parse"         : 기본값 (자동으로 최신 안정 버전)
               - "document-parse-250404"  : 2025.04 버전 (폼·회전 문서·장문 이미지 개선)
        output_format: 출력 포맷 ("markdown" | "html" | "text")
        ocr: OCR 모드 ("auto" | "force" | "off")
        mode: 처리 모드
              - "auto"     : 페이지별 자동 선택 (권장)
              - "standard" : 기본 OCR 텍스트 추출 (빠름)
              - "enhanced" : VLM 기반 정밀 파싱 (복잡한 표·차트·다이어그램 특화)

    Returns:
        {페이지번호: Markdown 텍스트} 딕셔너리
    """
    file_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        print(f"  파일 크기 {file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB → 페이지 단위 분할 처리")
        return _convert_pdf_by_pages(pdf_path, api_key, model, output_format, ocr, mode)

    # 크기 정상 → 전체 파일 한 번에 전송
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = _call_api(pdf_bytes, Path(pdf_path).name, api_key, model, output_format, ocr, mode)
    pages = _parse_response(result, output_format)
    print(f"✓ {len(pages)}페이지 처리 완료")
    return pages


def _convert_pdf_by_pages(
    pdf_path: str,
    api_key: str,
    model: str,
    output_format: str,
    ocr: str,
    mode: str,
) -> dict[int, str]:
    """
    PDF를 한 페이지씩 분할해 각각 Upstage API 전송 후 합산.

    413 / 500 오류 방지용. smart_router hybrid 모드와 동일한 분할 방식 사용.
    """
    doc = fitz.open(pdf_path)
    all_pages: dict[int, str] = {}
    errors: list[str] = []

    for page_num in range(doc.page_count):
        # 단일 페이지 PDF 생성 (메모리 버퍼)
        tmp_doc = fitz.open()
        tmp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        pdf_bytes = tmp_doc.tobytes()
        tmp_doc.close()

        try:
            result = _call_api(
                pdf_bytes, f"page_{page_num + 1}.pdf",
                api_key, model, output_format, ocr, mode,
            )
            pages = _parse_response(result, output_format)
            # 응답의 페이지 번호와 무관하게 실제 페이지 번호로 저장
            content = "\n\n".join(pages.values())
            all_pages[page_num + 1] = content
            print(f"  ✓ 페이지 {page_num + 1}/{doc.page_count}")
        except Exception as e:
            print(f"  ✗ 페이지 {page_num + 1} 오류: {e}")
            errors.append(f"p{page_num + 1}: {e}")
            all_pages[page_num + 1] = f"<!-- 페이지 {page_num + 1} 처리 오류: {e} -->"

    doc.close()
    print(f"✓ 총 {len(all_pages)}페이지 처리 완료 (오류 {len(errors)}건)")
    return all_pages


def save_markdown(markdown_pages: dict[int, str], output_path: str) -> None:
    """
    페이지별 Markdown 딕셔너리를 하나의 파일로 저장.

    Args:
        markdown_pages: {페이지번호: Markdown} 딕셔너리
        output_path: 저장할 파일 경로
    """
    if len(markdown_pages) == 1:
        combined = next(iter(markdown_pages.values()))
    else:
        combined = "\n\n---\n\n".join(
            f"# Page {page_num}\n\n{content}"
            for page_num, content in sorted(markdown_pages.items())
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"✓ 저장 완료: {output_path}")


def convert_folder(pdf_folder: str, output_folder: str, api_key: str) -> None:
    """폴더 내 모든 PDF를 Upstage Document Parse로 일괄 변환."""
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠ PDF 파일 없음: {pdf_folder}")
        return

    for filename in pdf_files:
        print(f"\n처리 중: {filename}")
        pdf_path = os.path.join(pdf_folder, filename)
        pdf_name = os.path.splitext(filename)[0]

        markdown_pages = convert_pdf(pdf_path, api_key)

        output_path = os.path.join(output_folder, f"{pdf_name}.md")
        save_markdown(markdown_pages, output_path)

    print(f"\n일괄 변환 완료. 출력 위치: '{output_folder}'")


if __name__ == "__main__":
    api_key = os.environ.get("UPSTAGE_API_KEY", "")
    convert_folder("./complex_pdfs", "./md_output/upstage", api_key)
