"""
Category 3: Complex PDFs - OCR 특화 VLM (K8s 로컬 서빙)

GPU 없는 CPU K8s 환경에서 실행. 각 모델은 K8s 내부 서비스로 사전 배포됨.

지원 모델 (8B 미만):
  백엔드 키        모델                    파라미터  K8s 서비스
  ─────────────────────────────────────────────────────────────────
  granite-vision   granite3.3-vision:2b     2B       Ollama  → ollama-server:11434
  got-ocr2         GOT-OCR-2.0-hf           580M     FastAPI → got-ocr2-server:8000
  paddleocr-vl     PaddleOCR-VL-1.5         0.9B     FastAPI → paddleocr-vl-server:8000

K8s 배포 사전 조건:
  kubectl apply -f k8s/manifests/ollama-deployment.yaml
  kubectl apply -f k8s/manifests/got-ocr2-deployment.yaml
  kubectl apply -f k8s/manifests/paddleocr-vl-deployment.yaml

로컬 테스트 (포트 포워딩):
  kubectl -n rag-bench-test port-forward svc/ollama-server 11434:11434
  kubectl -n rag-bench-test port-forward svc/got-ocr2-server 8000:8000
  kubectl -n rag-bench-test port-forward svc/paddleocr-vl-server 8001:8000
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import fitz  # PyMuPDF

from category3_openai import SYSTEM_PROMPT

# ── 모델 프로파일 ──────────────────────────────────────────────────────────────
# { backend_key: (model_id, base_url) }
# base_url은 환경변수 OPENSOURCE_VLM_ENDPOINT로 오버라이드 가능

MODEL_PROFILES: dict[str, tuple[str, str]] = {
    # Ollama 서빙 (K8s 내부)
    "granite-vision": (
        "ibm/granite3.3-vision:2b",
        "http://ollama-server:11434/v1",
    ),
    # FastAPI 서빙 (K8s 내부, CPU)
    "got-ocr2": (
        "got-ocr2",
        "http://got-ocr2-server:8000/v1",
    ),
    "paddleocr-vl": (
        "paddleocr-vl-1.5",
        "http://paddleocr-vl-server:8000/v1",
    ),
    # FastAPI 서빙 (K8s 내부, GPU 필수)
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
    """
    엔드포인트 URL, 모델명, API 키 결정.

    우선순위:
      1. 명시적 인수 (model, base_url, api_key)
      2. 환경변수 (OPENSOURCE_VLM_ENDPOINT, OPENSOURCE_VLM_MODEL)
      3. MODEL_PROFILES 기본값
    """
    env_endpoint = os.environ.get("OPENSOURCE_VLM_ENDPOINT")
    env_model    = os.environ.get("OPENSOURCE_VLM_MODEL")

    default_model, default_url = MODEL_PROFILES.get(
        backend_key, (backend_key, "http://ollama-server:11434/v1")
    )

    resolved_url   = base_url  or env_endpoint or default_url
    resolved_model = model     or env_model    or default_model
    resolved_key   = api_key   or os.environ.get("OPENSOURCE_VLM_API_KEY", "ollama")

    return resolved_url, resolved_model, resolved_key


def convert_pdf(
    pdf_path: str,
    api_key: str = "ollama",
    backend_key: str = "granite-vision",
    model: str | None = None,
    base_url: str | None = None,
    dpi: int = RENDER_DPI,
) -> dict[int, str]:
    """
    단일 PDF를 페이지별로 OCR VLM을 통해 Markdown 변환.

    Args:
        pdf_path:    PDF 파일 경로
        api_key:     API 키 (로컬 Ollama/FastAPI는 아무 값)
        backend_key: 모델 프로파일 키 (MODEL_PROFILES 참고)
        model:       모델명 명시 (None이면 backend_key 기본값)
        base_url:    엔드포인트 URL (None이면 환경변수 또는 기본값)
        dpi:         페이지 렌더링 해상도

    Returns:
        {페이지번호: Markdown 텍스트} 딕셔너리
    """
    from openai import OpenAI

    resolved_url, resolved_model, resolved_key = _resolve_endpoint_and_model(
        backend_key, model, base_url, api_key or None
    )

    client = OpenAI(api_key=resolved_key, base_url=resolved_url)
    pdf_document = fitz.open(pdf_path)
    markdown_pages: dict[int, str] = {}
    scale = dpi / 72

    print(f"  엔드포인트: {resolved_url}")
    print(f"  모델:       {resolved_model}")

    for page_num in range(pdf_document.page_count):
        try:
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

            response = client.chat.completions.create(
                model=resolved_model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                "이 PDF 페이지의 모든 내용을 마크다운으로 변환하십시오."
                            )},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "high",
                            }},
                        ],
                    },
                ],
            )

            markdown_pages[page_num + 1] = response.choices[0].message.content
            print(f"  ✓ 페이지 {page_num + 1}/{pdf_document.page_count}")

        except Exception as e:
            print(f"  ✗ 페이지 {page_num + 1} 오류: {e}")
            markdown_pages[page_num + 1] = f"<!-- 페이지 처리 오류: {e} -->"

    pdf_document.close()
    return markdown_pages


def save_markdown(markdown_pages: dict[int, str], output_path: str) -> None:
    """페이지별 Markdown 딕셔너리를 하나의 파일로 저장."""
    combined = "\n\n---\n\n".join(
        f"# Page {page_num}\n\n{content}"
        for page_num, content in sorted(markdown_pages.items())
    )
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"✓ 저장 완료: {output_path}")
