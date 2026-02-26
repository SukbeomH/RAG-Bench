"""
Smart PDF Router - PDF 특성에 따라 최적 변환 도구 자동 선택

mode="document" (기본):
  문서 전체를 분석해 카테고리를 결정 후 단일 도구 적용.
  판별 기준:
    1. 스캔 문서 여부 (텍스트 선택 불가 → Category 2 / Docling)
    2. 이미지 밀도 (이미지 多 → Category 3 / VLM)
    3. 그 외 디지털 텍스트 PDF → Category 1 / PyMuPDF4LLM

mode="hybrid":
  페이지별로 백엔드를 선택 (MinerU 2.0+ 방식).
  텍스트 추출 가능 페이지 → Rule-based,  스캔/이미지 페이지 → VLM.
  혼합 문서에서 속도와 정확도를 동시에 확보.

VLM 백엔드:
  "openai"          : GPT-4o Vision (기본값)
  "openai-4.1"      : GPT-4.1 Vision (최고 성능)
  "upstage"         : Upstage Document Parse (페이지 단위 분할 전송)
  "upstage-enhanced": Upstage enhanced 모드 (차트·표 특화)
"""

import os
import fitz  # PyMuPDF
from pathlib import Path

import hybrid_backend
from quality_checker import check_quality


# 판별 임계값
SCANNED_TEXT_THRESHOLD = 50   # 페이지당 추출 문자 수가 이 미만이면 스캔 문서로 판단
IMAGE_COUNT_THRESHOLD = 2     # 첫 페이지 이미지 수가 이 초과면 이미지 중심 문서로 판단


def classify_pdf(pdf_path: str) -> str:
    """
    PDF를 분석해 카테고리 반환.

    Returns:
        "simple" | "medium" | "complex"
    """
    doc = fitz.open(pdf_path)
    sample_page = doc[0]

    text = sample_page.get_text()
    image_count = len(sample_page.get_images())
    doc.close()

    is_scanned = len(text.strip()) < SCANNED_TEXT_THRESHOLD
    has_many_images = image_count > IMAGE_COUNT_THRESHOLD

    if is_scanned:
        return "medium"
    if has_many_images:
        return "complex"
    return "simple"


def _get_vlm_api_key(vlm_backend: str) -> str | None:
    """VLM 백엔드에 맞는 API 키 환경변수에서 로드."""
    if vlm_backend in ("openai", "openai-4.1"):
        return os.environ.get("OPENAI_API_KEY")
    elif vlm_backend in ("upstage", "upstage-enhanced"):
        return os.environ.get("UPSTAGE_API_KEY")
    return None


def _get_vlm_model(vlm_backend: str) -> str:
    """VLM 백엔드 기본 모델명 반환."""
    if vlm_backend == "openai-4.1":
        return "gpt-4.1"
    elif vlm_backend == "openai":
        return "gpt-4o"
    return ""


def route_pdf(
    pdf_path: str,
    output_folder: str,
    vlm_backend: str = "openai",
    api_key: str | None = None,
    mode: str = "document",
) -> str:
    """
    단일 PDF를 분석 후 적절한 변환 도구로 처리.

    Args:
        pdf_path:     PDF 파일 경로
        output_folder:Markdown 출력 폴더 경로
        vlm_backend:  VLM 백엔드 ("openai" | "openai-4.1" | "upstage" | "upstage-enhanced")
        api_key:      API 키 (없으면 환경변수에서 자동 로드)
        mode:         "document" | "hybrid"
                      document - 문서 단위 분류 (기존 방식)
                      hybrid   - 페이지 단위 분류 (MinerU 방식)

    Returns:
        저장된 Markdown 파일 경로
    """
    pdf_name = Path(pdf_path).stem
    output_path = os.path.join(output_folder, f"{pdf_name}.md")
    os.makedirs(output_folder, exist_ok=True)

    key = api_key or _get_vlm_api_key(vlm_backend)
    model = _get_vlm_model(vlm_backend)

    if mode == "hybrid":
        print(f"{Path(pdf_path).name} → HYBRID (페이지별 라우팅, vlm={vlm_backend})")
        report = hybrid_backend.convert_pdf(
            pdf_path, output_path,
            vlm_backend=vlm_backend,
            api_key=key,
            model=model,
        )
        print(f"  → {report.summary()}")
        return output_path

    # mode == "document"
    category = classify_pdf(pdf_path)
    print(f"{Path(pdf_path).name} → {category.upper()}")

    if category == "simple":
        import category1_simple as cat1
        cat1.convert_pdf(pdf_path, output_path)

    elif category == "medium":
        import category2_medium as cat2
        converter = cat2.build_converter()
        cat2.convert_pdf(pdf_path, output_path, converter=converter)

    else:  # complex → VLM
        if not key:
            raise ValueError(
                f"complex PDF 변환에는 {vlm_backend.upper()}_API_KEY가 필요합니다."
            )
        if vlm_backend in ("openai", "openai-4.1"):
            import category3_openai as cat3
            pages = cat3.convert_pdf(pdf_path, key, model=model)
            cat3.save_markdown(pages, output_path)
        elif vlm_backend in ("upstage", "upstage-enhanced"):
            import category3_upstage as cat3
            upstage_mode = "enhanced" if vlm_backend == "upstage-enhanced" else "auto"
            pages = cat3.convert_pdf(pdf_path, key, mode=upstage_mode)
            cat3.save_markdown(pages, output_path)
        else:
            raise ValueError(f"알 수 없는 VLM 백엔드: {vlm_backend}")

    return output_path


def route_folder(
    pdf_folder: str,
    output_folder: str,
    vlm_backend: str = "openai",
    api_key: str | None = None,
    mode: str = "document",
    run_quality_check: bool = True,
) -> None:
    """
    폴더 내 모든 PDF를 자동 분류 후 변환.

    Args:
        pdf_folder:       PDF 파일이 있는 폴더 경로
        output_folder:    Markdown 출력 폴더 경로
        vlm_backend:      VLM 백엔드
        api_key:          API 키 (없으면 환경변수 자동 로드)
        mode:             "document" | "hybrid"
        run_quality_check:변환 후 품질 검사 실행 여부
    """
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠ PDF 파일 없음: {pdf_folder}")
        return

    category_tally: dict[str, list[str]] = {
        "simple": [], "medium": [], "complex": [], "hybrid": []
    }

    for pdf_file in pdf_files:
        try:
            output_path = route_pdf(
                str(pdf_file), output_folder,
                vlm_backend=vlm_backend, api_key=api_key, mode=mode,
            )

            if mode == "hybrid":
                category_tally["hybrid"].append(pdf_file.name)
            else:
                cat = classify_pdf(str(pdf_file))
                category_tally[cat].append(pdf_file.name)

            if run_quality_check:
                metrics = check_quality(output_path)
                print(
                    f"  품질: 단어 {metrics['word_count']}개 | "
                    f"헤더 {'O' if metrics['has_headers'] else 'X'} | "
                    f"표 {'O' if metrics['has_tables'] else 'X'}"
                )
        except Exception as e:
            print(f"✗ 오류 ({pdf_file.name}): {e}")

    print("\n=== 변환 요약 ===")
    for cat, files in category_tally.items():
        if files:
            print(f"  {cat.upper()} ({len(files)}개): {', '.join(files)}")
    print(f"출력 위치: '{output_folder}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF → Markdown 변환")
    parser.add_argument("--input",   default="./pdfs",      help="PDF 폴더 경로")
    parser.add_argument("--output",  default="./md_output",  help="출력 폴더 경로")
    parser.add_argument("--mode",    choices=["document", "hybrid"], default="hybrid",
                        help="document: 문서 단위 분류 / hybrid: 페이지 단위 분류 (기본값)")
    parser.add_argument("--vlm",     default="openai",
                        choices=["openai", "openai-4.1", "upstage", "upstage-enhanced"],
                        help="VLM 백엔드 (기본값: openai)")
    args = parser.parse_args()

    route_folder(args.input, args.output, vlm_backend=args.vlm, mode=args.mode)
