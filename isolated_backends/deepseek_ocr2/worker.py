"""
DeepSeek-OCR-2 subprocess worker.
페이지별 PNG → Markdown 변환 후 JSON을 stdout으로 출력.

실행: python deepseek_ocr2_worker.py <pdf_path>

출력 형식:
    ---OUTPUT_START---
    {"1": "page1 markdown", "2": "page2 markdown", ...}
    ---OUTPUT_END---

환경:
  - torch MPS (Apple Silicon) 또는 CUDA, fallback CPU
  - transformers >= 4.51.1 (flash_attention_2 불필요, sdpa 사용)
  - MODEL_NAME 환경변수로 모델 경로 지정 (기본: deepseek-ai/DeepSeek-OCR-2)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

SSL_CERT = "/Users/sukbeom/Documents/cert/combined-ca-bundle.pem"
if os.path.exists(SSL_CERT):
    os.environ.setdefault("SSL_CERT_FILE", SSL_CERT)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", SSL_CERT)

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")
DOC_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def _get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = _get_device()
    attn_impl = "flash_attention_2" if device == "cuda" else "sdpa"

    print(f"[DeepSeek-OCR-2] 모델 로딩: {MODEL_NAME} | device={device} | attn={attn_impl}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation=attn_impl,
        trust_remote_code=True,
        use_safetensors=True,
    )

    dtype = torch.bfloat16
    model = model.eval().to(dtype).to(device)
    print(f"[DeepSeek-OCR-2] 준비 완료 ({device}, {dtype})", file=sys.stderr)
    return tokenizer, model, device


def _infer_page(tokenizer, model, img_path: str, tmpdir: str, page_num: int) -> str:
    out_base = os.path.join(tmpdir, f"out_p{page_num}")
    res = model.infer(
        tokenizer,
        prompt=DOC_PROMPT,
        image_file=img_path,
        output_path=out_base,
        base_size=1024,
        image_size=768,
        crop_mode=True,
        save_results=True,
    )

    # 반환값이 문자열이면 그대로 사용
    if isinstance(res, str) and res.strip():
        return res

    # save_results=True 로 저장된 파일 읽기
    for candidate in [out_base + ".md", out_base]:
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                return f.read()

    return str(res) if res else ""


def main():
    if len(sys.argv) < 2:
        print("---OUTPUT_START---")
        print(json.dumps({"error": "pdf_path 인수가 없습니다."}))
        print("---OUTPUT_END---")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("---OUTPUT_START---")
        print(json.dumps({"error": "PyMuPDF(fitz) 미설치. pip install pymupdf"}))
        print("---OUTPUT_END---")
        sys.exit(1)

    try:
        tokenizer, model, device = _load_model()
    except Exception as e:
        print("---OUTPUT_START---")
        print(json.dumps({"error": f"모델 로드 실패: {e}"}))
        print("---OUTPUT_END---")
        sys.exit(1)

    pages_output: dict[str, str] = {}
    pdf_doc = fitz.open(pdf_path)
    DPI = 300
    scale = DPI / 72

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(pdf_doc.page_count):
            page_num = i + 1
            try:
                page = pdf_doc[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                img_path = os.path.join(tmpdir, f"page_{page_num}.png")
                pix.save(img_path)

                md = _infer_page(tokenizer, model, img_path, tmpdir, page_num)
                pages_output[str(page_num)] = md
                print(f"  ✓ 페이지 {page_num}/{pdf_doc.page_count}", file=sys.stderr)

            except Exception as e:
                pages_output[str(page_num)] = f"<!-- 페이지 처리 오류: {e} -->"
                print(f"  ✗ 페이지 {page_num} 오류: {e}", file=sys.stderr)

    pdf_doc.close()

    print("---OUTPUT_START---")
    print(json.dumps(pages_output, ensure_ascii=False))
    print("---OUTPUT_END---")


if __name__ == "__main__":
    main()
