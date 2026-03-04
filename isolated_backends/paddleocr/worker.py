"""
Subprocess worker to run PaddleOCR doc_parser in an isolated .venv (Python 3.13).
Writes JSON list to stdout so the parent process can capture it.
"""

import sys
import os
import json
import warnings

# Suppress warnings for cleaner JSON output
warnings.filterwarnings("ignore")

_cert = os.environ.get("SSL_CERT_BUNDLE", "")
if _cert and os.path.exists(_cert):
    os.environ.setdefault("SSL_CERT_FILE", _cert)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _cert)

try:
    from paddleocr._pipelines.paddleocr_vl import PaddleOCRVL
except ImportError as e:
    print(json.dumps({"error": f"Import error: {str(e)}"}))
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing pdf path argument."}))
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        pipeline = PaddleOCRVL(
            vl_rec_backend="mlx-vlm-server",
            vl_rec_server_url="http://localhost:8111/",
            vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
            device="cpu",
        )

        results = pipeline.predict(pdf_path)

        pages_output = {}
        for res in results:
            page_idx = res["page_index"] + 1  # 1-based indexing for benchmark pipeline

            # Combine content blocks natively
            try:
                import markdownify
            except ImportError:
                markdownify = None

            blocks = res.get("parsing_res_list", [])
            page_md_blocks = []
            for b in blocks:
                content = None
                label = None
                if isinstance(b, dict):
                    content = b.get("block_content") or b.get("content")
                    label = b.get("block_label") or b.get("label")
                elif hasattr(b, "__dict__"):
                    content = b.__dict__.get("content") or b.__dict__.get(
                        "block_content"
                    )
                    label = b.__dict__.get("label") or b.__dict__.get("block_label")

                if content:
                    text_content = str(content)
                    if label == "table" and markdownify is not None:
                        text_content = markdownify.markdownify(text_content).strip()
                    page_md_blocks.append(text_content)

            page_md = "\n\n".join(page_md_blocks)

            pages_output[page_idx] = page_md

        # Output JSON result
        print("---OUTPUT_START---")
        print(json.dumps(pages_output, ensure_ascii=False))
        print("---OUTPUT_END---")

    except Exception as e:
        print("---OUTPUT_START---")
        print(json.dumps({"error": str(e)}))
        print("---OUTPUT_END---")
        sys.exit(1)


if __name__ == "__main__":
    main()
