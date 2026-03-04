"""벤치마크용 PDF 생성 스크립트.

카테고리별 5페이지 PDF를 생성:
  A. text_only.pdf        — 순수 텍스트 페이지 (이미지 없음, 표 없음)
  B. table_native.pdf     — 표/그래프가 문서 형식(텍스트)으로 포함된 페이지
  C. table_image*.pdf     — B와 동일 페이지를 DPI별 이미지로 래스터화 (비교용)
  E. graph_rich.pdf       — 차트/다이어그램/이미지 위주 페이지 (원본)
     graph_rich_image*.pdf — E를 DPI별 래스터화 (비교용)
"""

import fitz  # PyMuPDF
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"
OUTPUT_DIR = Path(__file__).resolve().parent

AI_REPORT = DOCS_DIR / "20250910_AI 현황 보고서.pdf"
SPRI_BRIEF = DOCS_DIR / "SPRi AI Brief_1월호_산업동향_0102_F.pdf"

# 0-indexed 페이지 번호 (PDF 내부 인덱스)

# A. 텍스트 전용: chars>1200, imgs=0, table=False
#   p32(1927), p114(1840), p111(1739), p66(1709), p12(1649)
TEXT_ONLY_PAGES = {
    AI_REPORT: [31, 113, 110, 65, 11],  # 0-indexed
}

# B. 표/그래프 문서형식: table=True, imgs<=1 (텍스트 기반 표)
#   AI: p188(2050, table), p62(1416, table), p43(1387, table), p61(1349, table)
#   SPRi: p7(1987, table, 정책법제 표 포함)
TABLE_NATIVE_PAGES = {
    AI_REPORT: [187, 61, 42, 60],  # 0-indexed
    SPRI_BRIEF: [6],               # 0-indexed
}

# C. 표/그래프 이미지: B(table_native)와 동일 페이지를 래스터화하여 비교 가능하게
#   TABLE_NATIVE_PAGES와 동일 소스 사용 → create_scanned_pdf()로 생성

# E. 그래프/이미지 위주: imgs>=3, 차트·다이어그램 포함 (다양한 주제)
#   p8(헬스케어 AI 차트, imgs=3), p74(Colossus 인프라 다이어그램, imgs=3),
#   p94(태양광/ESS 에너지 도해, imgs=3), p122(화웨이 광학칩 아키텍처, imgs=4),
#   p178(로보택시 자율주행 이미지, imgs=4)
GRAPH_RICH_PAGES = {
    AI_REPORT: [7, 73, 93, 121, 177],  # 0-indexed
}

DPI = 300


def extract_pages(page_map: dict, output_path: Path) -> None:
    """여러 PDF에서 지정 페이지를 추출하여 새 PDF로 합침."""
    out_doc = fitz.open()
    for pdf_path, pages in page_map.items():
        src = fitz.open(str(pdf_path))
        for pg in pages:
            out_doc.insert_pdf(src, from_page=pg, to_page=pg)
        src.close()
    out_doc.save(str(output_path))
    out_doc.close()
    print(f"  -> {output_path.name} ({sum(len(v) for v in page_map.values())} pages)")


def create_scanned_pdf(page_map: dict, output_path: Path, dpi: int = DPI) -> None:
    """지정 페이지를 이미지로 래스터화하여 스캔 시뮬레이션 PDF 생성."""
    out_doc = fitz.open()
    zoom = dpi / 72  # 72 DPI 기본 → 300 DPI
    mat = fitz.Matrix(zoom, zoom)

    for pdf_path, pages in page_map.items():
        src = fitz.open(str(pdf_path))
        for pg in pages:
            page = src[pg]
            pix = page.get_pixmap(matrix=mat)

            # 이미지 → 새 PDF 페이지
            img_pdf = fitz.open()
            rect = fitz.Rect(0, 0, pix.width * 72 / dpi, pix.height * 72 / dpi)
            img_page = img_pdf.new_page(width=rect.width, height=rect.height)
            img_page.insert_image(rect, pixmap=pix)
            out_doc.insert_pdf(img_pdf)
            img_pdf.close()
        src.close()

    out_doc.save(str(output_path))
    out_doc.close()
    print(f"  -> {output_path.name} ({sum(len(v) for v in page_map.values())} pages, {dpi} DPI scanned)")


def main():
    print("=== 벤치마크 PDF 생성 ===\n")

    # 소스 파일 확인
    for p in [AI_REPORT, SPRI_BRIEF]:
        if not p.exists():
            raise FileNotFoundError(f"소스 PDF 없음: {p}")
        print(f"  소스: {p.name}")
    print()

    # A. 텍스트 전용
    print("[A] 텍스트 전용 (text_only.pdf)")
    extract_pages(TEXT_ONLY_PAGES, OUTPUT_DIR / "text_only.pdf")

    # B. 표/그래프 문서형식
    print("[B] 표/그래프 문서형식 (table_native.pdf)")
    extract_pages(TABLE_NATIVE_PAGES, OUTPUT_DIR / "table_native.pdf")

    # C. 표/그래프 이미지 (B와 동일 페이지를 DPI별 래스터화)
    for dpi in [300, 200, 150, 72]:
        suffix = f"_{dpi}dpi" if dpi != 300 else ""
        fname = f"table_image{suffix}.pdf"
        print(f"[C] 표/그래프 이미지 — B를 {dpi}DPI 래스터화 ({fname})")
        create_scanned_pdf(TABLE_NATIVE_PAGES, OUTPUT_DIR / fname, dpi=dpi)

    # E. 그래프/이미지 위주 (원본 + DPI별 래스터화)
    print("[E] 그래프/이미지 위주 원본 (graph_rich.pdf)")
    extract_pages(GRAPH_RICH_PAGES, OUTPUT_DIR / "graph_rich.pdf")

    for dpi in [300, 200, 150, 72]:
        suffix = f"_{dpi}dpi" if dpi != 300 else ""
        fname = f"graph_rich_image{suffix}.pdf"
        print(f"[E] 그래프/이미지 — {dpi}DPI 래스터화 ({fname})")
        create_scanned_pdf(GRAPH_RICH_PAGES, OUTPUT_DIR / fname, dpi=dpi)

    print("\n=== 완료 ===")
    print(f"출력 디렉토리: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
