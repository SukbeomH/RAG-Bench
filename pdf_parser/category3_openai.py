"""
Category 3: Complex PDFs - OpenAI GPT-4o Vision

대상: 차트/다이어그램/이미지가 핵심인 문서, 복잡한 레이아웃, 과학 논문
도구: OpenAI GPT-4o (Vision)
"""

import base64
import os

import fitz  # PyMuPDF
from openai import OpenAI

SYSTEM_PROMPT = """You are an expert document parser specializing in converting PDF pages to markdown format.

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

**Mathematical Formulas:**
- Convert to LaTeX format: inline `$formula$`, display `$$formula$$`
- If LaTeX conversion is uncertain, describe the formula clearly

**Images, Diagrams, Charts:**
- Insert markdown image placeholder: `![Description](image)`
- Provide a detailed, informative description including:
  * Type of visual (photo, diagram, chart, graph, illustration)
  * Main subject or purpose
  * Key elements, labels, or data points
  * Colors, patterns, or notable visual features
  * Context or relationship to surrounding text
- For charts/graphs: mention axes, data trends, and key values
- For diagrams: describe components and their relationships

**Special Elements:**
- Footnotes: Use markdown footnote syntax `[^1]`
- Citations: Preserve as written
- Code blocks: Use triple backticks with language specification
- Quotes: Use `>` for blockquotes
- Links: Preserve as `[text](url)` if visible

**Quality Guidelines:**
- DO NOT add explanations, comments, or meta-information
- DO NOT skip or summarize content
- DO NOT invent or hallucinate text not present in the image
- DO NOT include "Here is the markdown..." or similar preambles
- Output ONLY the markdown content, nothing else

**Output Format:**
Return raw markdown with no wrapper, no code blocks, no explanations.
Start immediately with the page content."""


def convert_pdf(
    pdf_path: str,
    api_key: str,
    model: str = "gpt-4o",
    dpi: int = 300,
) -> dict[int, str]:
    """
    단일 PDF를 페이지별로 OpenAI Vision을 통해 Markdown 변환.

    Args:
        pdf_path: PDF 파일 경로
        api_key: OpenAI API 키
        model: 사용할 모델명
                - "gpt-4o"       : 균형 (현재 기본값, 검증됨)
                - "gpt-4.1"      : 최고 성능 (2025.04, 1M 컨텍스트, +6.7%)
                - "gpt-4.1-mini" : 저비용 고품질 (2025.04)
                - "gpt-4o-mini"  : 최저비용 (단순 문서용)
        dpi: 페이지 렌더링 해상도 (300 권장, 이미지 PDF는 150도 충분)

    Returns:
        {페이지번호: Markdown 텍스트} 딕셔너리
    """
    client = OpenAI(api_key=api_key)
    pdf_document = fitz.open(pdf_path)
    markdown_pages: dict[int, str] = {}
    scale = dpi / 72

    for page_num in range(pdf_document.page_count):
        try:
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

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
                                "text": (
                                    "Convert this PDF page to clean, structured markdown. "
                                    "Extract all text, describe images, and preserve the layout."
                                ),
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

            markdown_pages[page_num + 1] = response.choices[0].message.content
            print(f"✓ 페이지 {page_num + 1}/{pdf_document.page_count} 처리 완료")

        except Exception as e:
            print(f"✗ 페이지 {page_num + 1} 오류: {e}")
            markdown_pages[page_num + 1] = f"<!-- 페이지 처리 오류: {e} -->"

    pdf_document.close()
    return markdown_pages


def save_markdown(markdown_pages: dict[int, str], output_path: str) -> None:
    """
    페이지별 Markdown 딕셔너리를 하나의 파일로 저장.

    Args:
        markdown_pages: {페이지번호: Markdown} 딕셔너리
        output_path: 저장할 파일 경로
    """
    combined = "\n\n---\n\n".join(
        f"# Page {page_num}\n\n{content}"
        for page_num, content in sorted(markdown_pages.items())
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"✓ 저장 완료: {output_path}")


def convert_folder(pdf_folder: str, output_folder: str, api_key: str) -> None:
    """폴더 내 모든 PDF를 GPT-4o Vision으로 일괄 변환."""
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
    api_key = os.environ.get("OPENAI_API_KEY", "")
    convert_folder("./complex_pdfs", "./md_output/openai", api_key)
