"""
Subprocess worker — PaddleOCR 블록 단위 구조화 JSON 출력.
기존 worker.py와 동일한 격리 venv에서 실행되며,
블록별 label·content·bbox + 페이지 메타(width·height)를 보존한다.

출력 형식:
[
  {
    "page_index": 0,
    "page_count": N,
    "width": 595,
    "height": 842,
    "parsing_res_list": [
      {"block_label": "text", "block_content": "...", "block_bbox": [x1,y1,x2,y2],
       "block_id": 0, "block_order": 0},
      ...
    ]
  },
  ...
]
"""

import sys
import os
import json
import warnings

warnings.filterwarnings("ignore")

# base64 추출 대상 label
_BASE64_LABELS = frozenset(
    {
        "table",
        "image",
        "figure",
        "chart",
        "flowchart",
        "seal",
        "formula",
        "header_image",
        "footer_image",
    }
)

_cert = os.environ.get("SSL_CERT_BUNDLE", "")
if _cert and os.path.exists(_cert):
    os.environ.setdefault("SSL_CERT_FILE", _cert)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _cert)

try:
    from paddleocr._pipelines.paddleocr_vl import PaddleOCRVL
except ImportError as e:
    print(json.dumps({"error": f"Import error: {str(e)}"}))
    sys.exit(1)


def _pil_to_base64(pil_img, quality: int = 85) -> str:
    """PIL.Image → JPEG base64 문자열."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_block(b) -> dict | None:
    """LayoutBlock (object or dict) → serializable dict.
    table/image/chart 등은 block.image에서 base64 이미지를 추출한다.
    """
    if isinstance(b, dict):
        label = b.get("block_label") or b.get("label")
        content = b.get("block_content") or b.get("content")
        bbox = b.get("block_bbox") or b.get("bbox")
        block_id = b.get("block_id") or b.get("index")
        block_order = b.get("block_order") or b.get("order_index")
        img_data = b.get("image")
    elif hasattr(b, "__dict__"):
        d = b.__dict__
        label = d.get("label") or d.get("block_label")
        content = d.get("content") or d.get("block_content")
        bbox = d.get("bbox") or d.get("block_bbox")
        block_id = d.get("index") or d.get("block_id")
        block_order = d.get("order_index") or d.get("block_order")
        img_data = d.get("image")
    else:
        return None

    if not label:
        return None

    result = {
        "block_label": str(label),
        "block_content": str(content) if content else "",
        "block_bbox": list(bbox) if bbox else None,
    }
    if block_id is not None:
        result["block_id"] = int(block_id)
    if block_order is not None:
        result["block_order"] = int(block_order)

    # base64 이미지 추출 — 메모리에 이미지가 있으면 무조건 저장
    if img_data is not None:
        pil_img = None
        if isinstance(img_data, dict):
            pil_img = img_data.get("img")
        elif hasattr(img_data, "save"):  # PIL.Image duck-typing
            pil_img = img_data
        if pil_img is not None:
            try:
                result["base64_encoding"] = _pil_to_base64(pil_img)
            except Exception:
                pass  # 이미지 변환 실패 시 무시

    return result


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing pdf path argument."}))
        sys.exit(1)

    pdf_path = sys.argv[1]

    # 선택적 페이지 범위: worker_structured.py <pdf> [start_page] [end_page]
    start_page = int(sys.argv[2]) if len(sys.argv) > 2 else None
    end_page = int(sys.argv[3]) if len(sys.argv) > 3 else None

    try:
        pipeline = PaddleOCRVL(
            vl_rec_backend="mlx-vlm-server",
            vl_rec_server_url=os.environ.get(
                "OPENSOURCE_VLM_ENDPOINT", "http://localhost:8111/"
            ),
            vl_rec_api_model_name=os.environ.get(
                "OPENSOURCE_VLM_MODEL", "PaddlePaddle/PaddleOCR-VL-1.5"
            ),
            device="cpu",
        )

        results = pipeline.predict(pdf_path, max_new_tokens=4096)

        structured_output = []
        for res in results:
            page_index = (
                res.get("page_index", 0)
                if isinstance(res, dict)
                else getattr(res, "page_index", 0)
            )

            # 페이지 범위 필터
            if start_page is not None and page_index < start_page:
                continue
            if end_page is not None and page_index > end_page:
                continue

            page_width = (
                res.get("width", 0)
                if isinstance(res, dict)
                else getattr(res, "width", 0)
            )
            page_height = (
                res.get("height", 0)
                if isinstance(res, dict)
                else getattr(res, "height", 0)
            )
            page_count = (
                res.get("page_count", 0)
                if isinstance(res, dict)
                else getattr(res, "page_count", 0)
            )
            blocks_raw = (
                res.get("parsing_res_list", [])
                if isinstance(res, dict)
                else getattr(res, "parsing_res_list", [])
            )

            blocks = []
            for b in blocks_raw:
                extracted = _extract_block(b)
                if extracted:
                    blocks.append(extracted)

            structured_output.append(
                {
                    "page_index": page_index,
                    "page_count": page_count,
                    "width": page_width,
                    "height": page_height,
                    "parsing_res_list": blocks,
                }
            )

        print("---OUTPUT_START---")
        print(json.dumps(structured_output, ensure_ascii=False))
        print("---OUTPUT_END---")

    except Exception as e:
        import traceback

        print("---OUTPUT_START---")
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
        print("---OUTPUT_END---")
        sys.exit(1)


if __name__ == "__main__":
    main()
