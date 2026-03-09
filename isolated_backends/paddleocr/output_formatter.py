"""
PaddleOCR 네이티브 파이프라인 출력 → Upstage Document Parse API v2 호환 JSON 변환기.

HTML-first 전략: PaddleOCR 원본(table=HTML, text=plaintext)을 시맨틱 HTML로
정규화한 뒤, markdown·text를 파생한다.

Usage (worker.py 내부):
    from output_formatter import format_to_upstage
    results = pipeline.predict(pdf_path, max_new_tokens=4096)
    output = format_to_upstage(results)

Usage (CLI — 기존 worker JSON 후처리):
    python output_formatter.py <worker_output.json> [output.json]
"""

from __future__ import annotations

import base64
import html as html_mod
import io
import json
import re
import sys
from typing import Any

# ---------------------------------------------------------------------------
# label → HTML 태그 / content 생성 분류
# ---------------------------------------------------------------------------

# 표 계열 label (원본 HTML 보존)
_TABLE_LABELS = frozenset({"table"})

# 이미지/그림 계열 label (placeholder 생성)
_FIGURE_LABELS = frozenset(
    {
        "image",
        "figure",
        "flowchart",
        "seal",
        "header_image",
        "footer_image",
    }
)

# 차트 계열 (표 변환 또는 이미지)
_CHART_LABELS = frozenset({"chart"})

# 수식 계열 (LaTeX)
_EQUATION_LABELS = frozenset({"formula", "algorithm", "formula_number"})

# 제목 계열 label → HTML heading 태그
_TITLE_LABELS: dict[str, str] = {
    "doc_title": "h1",
    "paragraph_title": "h2",
    "abstract_title": "h2",
    "reference_title": "h2",
    "content_title": "h2",
}

# label → HTML 태그 (위 분류에 해당하지 않는 일반 블록)
_LABEL_HTML_TAG: dict[str, str] = {
    "text": "p",
    "aside_text": "aside",
    "abstract": "p",
    "content": "p",
    "header": "header",
    "footer": "footer",
    "number": "span",
    "footnote": "p",
    "vision_footnote": "p",
    "reference": "p",
    "table_title": "figcaption",
    "chart_title": "figcaption",
    "figure_title": "figcaption",
    "figure_table_chart_title": "figcaption",
}

# base64 이미지 추출 대상
_BASE64_LABELS = _TABLE_LABELS | _FIGURE_LABELS | _CHART_LABELS

# ---------------------------------------------------------------------------
# optional: markdownify (html→markdown 변환)
# ---------------------------------------------------------------------------
try:
    import markdownify as _markdownify

    def _html_to_markdown(html_str: str) -> str:
        return _markdownify.markdownify(html_str).strip()
except ImportError:
    _markdownify = None  # type: ignore[assignment]

    def _html_to_markdown(html_str: str) -> str:
        """markdownify 미설치 시 간이 변환."""
        text = html_str
        # headings
        for i in range(1, 7):
            text = re.sub(
                rf"<h{i}[^>]*>(.*?)</h{i}>",
                lambda m, level=i: f"{'#' * level} {m.group(1)}",
                text,
                flags=re.DOTALL,
            )
        # table 은 그대로 유지 (변환 복잡)
        # <p> → 줄바꿈
        text = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n", text, flags=re.DOTALL)
        # <li> → bullet
        text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", text, flags=re.DOTALL)
        # figure placeholder
        text = re.sub(
            r"<figure[^>]*>.*?</figure>",
            "![image](/image/placeholder)\n",
            text,
            flags=re.DOTALL,
        )
        # 나머지 태그 제거
        text = re.sub(
            r"<(?!table|/table|tr|/tr|th|/th|td|/td|thead|/thead|tbody|/tbody)[^>]+>",
            "",
            text,
        )
        return text.strip()


def _strip_html_tags(html_str: str) -> str:
    """HTML 태그를 모두 제거하여 순수 텍스트 반환."""
    text = re.sub(r"<[^>]+>", " ", html_str)
    text = html_mod.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 좌표 변환
# ---------------------------------------------------------------------------


def _bbox_to_coordinates(
    bbox: list[int | float],
    page_width: int,
    page_height: int,
) -> list[dict[str, float]]:
    """
    PaddleOCR bbox [x1, y1, x2, y2] (픽셀) →
    Upstage coordinates [{x,y}×4] (0~1 정규화, 4꼭짓점 시계방향).
    """
    x1, y1, x2, y2 = bbox
    nx1 = round(x1 / page_width, 4) if page_width else 0
    ny1 = round(y1 / page_height, 4) if page_height else 0
    nx2 = round(x2 / page_width, 4) if page_width else 0
    ny2 = round(y2 / page_height, 4) if page_height else 0
    return [
        {"x": nx1, "y": ny1},  # 좌상
        {"x": nx2, "y": ny1},  # 우상
        {"x": nx2, "y": ny2},  # 우하
        {"x": nx1, "y": ny2},  # 좌하
    ]


# ---------------------------------------------------------------------------
# 블록에서 이미지 추출 (base64)
# ---------------------------------------------------------------------------


def _extract_base64(block: Any) -> str | None:
    """LayoutBlock 객체 또는 dict에서 이미지가 있으면 JPEG base64 반환."""
    img_data = None
    if isinstance(block, dict):
        img_data = block.get("image")
    elif hasattr(block, "image"):
        img_data = block.image

    if img_data is None:
        return None

    pil_img = None
    if isinstance(img_data, dict):
        pil_img = img_data.get("img")
    elif hasattr(img_data, "save"):  # PIL.Image duck-typing
        pil_img = img_data

    if pil_img is None:
        return None

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# 블록 필드 추출 헬퍼
# ---------------------------------------------------------------------------


def _get_block_field(block: Any, *keys: str) -> Any:
    """dict 또는 object에서 여러 키를 순서대로 탐색."""
    for key in keys:
        if isinstance(block, dict):
            val = block.get(key)
        elif hasattr(block, key):
            val = getattr(block, key)
        else:
            val = None
        if val is not None:
            return val
    return None


# ---------------------------------------------------------------------------
# HTML-first: 블록 content → {html, markdown, text} 생성
# ---------------------------------------------------------------------------


def _build_content(
    raw_content: str,
    label: str,
) -> dict[str, str]:
    """
    PaddleOCR 블록의 raw content를 HTML-first로 {html, markdown, text} 생성.
    label(원본 block_label)에 따라 HTML 태그와 markdown 문법을 결정한다.
    """
    # --- 1) HTML 생성 (1차 포맷) ---
    if label in _TABLE_LABELS:
        if "<table" in raw_content.lower():
            content_html = raw_content
        else:
            content_html = (
                f"<table><tr><td>{html_mod.escape(raw_content)}</td></tr></table>"
            )
    elif label in _FIGURE_LABELS:
        content_html = '<figure><img src="/image/placeholder" /></figure>'
    elif label in _CHART_LABELS:
        if "<table" in raw_content.lower():
            content_html = raw_content
        else:
            content_html = '<figure><img src="/image/placeholder" /></figure>'
    elif label in _EQUATION_LABELS:
        content_html = f'<p class="equation">{html_mod.escape(raw_content)}</p>'
    elif label in _TITLE_LABELS:
        tag = _TITLE_LABELS[label]
        content_html = f"<{tag}>{html_mod.escape(raw_content)}</{tag}>"
    else:
        tag = _LABEL_HTML_TAG.get(label, "p")
        content_html = f"<{tag}>{html_mod.escape(raw_content)}</{tag}>"

    # --- 2) Markdown 파생 ---
    if label in _TABLE_LABELS or (
        label in _CHART_LABELS and "<table" in raw_content.lower()
    ):
        content_md = _html_to_markdown(content_html)
    elif label in _TITLE_LABELS:
        heading_level = 1 if _TITLE_LABELS[label] == "h1" else 2
        content_md = f"{'#' * heading_level} {raw_content}"
    elif label in _FIGURE_LABELS or label in _CHART_LABELS:
        content_md = "![image](/image/placeholder)"
    elif label in _EQUATION_LABELS:
        content_md = f"$${raw_content}$$" if raw_content.strip() else ""
    else:
        content_md = raw_content

    # --- 3) Text 파생 (태그/문법 제거) ---
    content_text = _strip_html_tags(content_html)

    return {
        "html": content_html,
        "markdown": content_md,
        "text": content_text,
    }


# ---------------------------------------------------------------------------
# 단일 블록 → element 변환
# ---------------------------------------------------------------------------


def _convert_block(
    block: Any,
    element_id: int,
    page_num: int,
    page_width: int,
    page_height: int,
) -> dict[str, Any] | None:
    """PaddleOCR 블록 하나를 Upstage element dict로 변환.
    category에는 PaddleOCR 원본 block_label을 그대로 사용한다.
    """
    label = _get_block_field(block, "block_label", "label")
    content_raw = _get_block_field(block, "block_content", "content")
    bbox = _get_block_field(block, "block_bbox", "bbox")

    if not label or content_raw is None:
        return None

    content_str = str(content_raw)

    element: dict[str, Any] = {
        "id": element_id,
        "category": label,  # 원본 label 그대로
        "page": page_num,
        "content": _build_content(content_str, label),
    }

    # 좌표
    if bbox and page_width > 0 and page_height > 0:
        element["coordinates"] = _bbox_to_coordinates(bbox, page_width, page_height)

    # base64 이미지 — worker에서 이미 직렬화된 경우 또는 PIL 객체
    b64 = _get_block_field(block, "base64_encoding")
    if b64:
        element["base64_encoding"] = b64
    else:
        b64 = _extract_base64(block)
        if b64:
            element["base64_encoding"] = b64

    return element


# ---------------------------------------------------------------------------
# 메인 변환 함수
# ---------------------------------------------------------------------------


def format_to_upstage(
    pipeline_results: list[Any],
    *,
    model: str = "paddleocr-vl",
) -> dict[str, Any]:
    """
    PaddleOCR VL pipeline.predict() 결과를 Upstage Document Parse API v2
    호환 JSON dict로 변환.

    Args:
        pipeline_results: pipeline.predict() 반환값 (페이지별 결과 리스트)
        model: 모델 식별자

    Returns:
        Upstage API v2 호환 dict
    """
    all_elements: list[dict[str, Any]] = []
    page_html_parts: list[str] = []
    page_md_parts: list[str] = []
    page_text_parts: list[str] = []
    element_id = 0
    total_pages = 0

    for res in pipeline_results:
        # 페이지 메타
        if isinstance(res, dict):
            page_index = res.get("page_index", 0)
            page_width = res.get("width", 0)
            page_height = res.get("height", 0)
            page_count = res.get("page_count", 0)
            blocks = res.get("parsing_res_list", [])
        else:
            page_index = getattr(res, "page_index", 0)
            page_width = getattr(res, "width", 0)
            page_height = getattr(res, "height", 0)
            page_count = getattr(res, "page_count", 0)
            blocks = getattr(res, "parsing_res_list", [])

        page_num = page_index + 1
        total_pages = max(total_pages, page_count, page_num)

        cur_html: list[str] = []
        cur_md: list[str] = []
        cur_text: list[str] = []

        for block in blocks:
            element = _convert_block(
                block, element_id, page_num, page_width, page_height
            )
            if element is None:
                continue
            all_elements.append(element)
            cur_html.append(element["content"]["html"])
            cur_md.append(element["content"]["markdown"])
            cur_text.append(element["content"]["text"])
            element_id += 1

        page_html_parts.append("\n".join(cur_html))
        page_md_parts.append("\n\n".join(cur_md))
        page_text_parts.append("\n".join(cur_text))

    return {
        "api": "2.0",
        "model": model,
        "ocr": True,
        "content": {
            "html": "\n".join(page_html_parts),
            "markdown": "\n\n".join(page_md_parts),
            "text": "\n".join(page_text_parts),
        },
        "elements": all_elements,
        "usage": {
            "pages": total_pages,
            "standard": list(range(1, total_pages + 1)),
        },
    }


# ---------------------------------------------------------------------------
# CLI: 기존 worker 출력(JSON) → Upstage 포맷 변환 (단독 실행용)
# ---------------------------------------------------------------------------


def format_from_worker_json(
    worker_json: dict[int, str],
    *,
    model: str = "paddleocr-vl",
) -> dict[str, Any]:
    """
    기존 worker.py가 출력하는 {page_num: markdown} JSON을
    Upstage 호환 형식으로 변환 (좌표/base64 없음, 구조만 매핑).

    블록 단위 분리 불가 → 페이지 전체를 단일 paragraph element로 처리.
    """
    all_elements: list[dict[str, Any]] = []
    all_html: list[str] = []
    all_md: list[str] = []
    all_text: list[str] = []
    element_id = 0
    pages = sorted(int(k) for k in worker_json.keys())

    for page_num in pages:
        md = (
            worker_json[str(page_num)]
            if str(page_num) in worker_json
            else worker_json.get(page_num, "")  # type: ignore[arg-type]
        )
        el_html = f"<p>{html_mod.escape(str(md))}</p>"
        el_text = _strip_html_tags(el_html)

        element: dict[str, Any] = {
            "id": element_id,
            "category": "paragraph",
            "page": page_num,
            "content": {
                "html": el_html,
                "markdown": str(md),
                "text": el_text,
            },
        }
        all_elements.append(element)
        all_html.append(el_html)
        all_md.append(str(md))
        all_text.append(el_text)
        element_id += 1

    return {
        "api": "2.0",
        "model": model,
        "ocr": True,
        "content": {
            "html": "\n".join(all_html),
            "markdown": "\n\n".join(all_md),
            "text": "\n".join(all_text),
        },
        "elements": all_elements,
        "usage": {
            "pages": len(pages),
            "standard": pages,
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <worker_output.json> [output.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else input_path.replace(".json", "_upstage.json")
    )

    with open(input_path, encoding="utf-8") as f:
        worker_data = json.load(f)

    result = format_from_worker_json(worker_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Converted: {input_path} → {output_path}")
    print(f"  Pages: {result['usage']['pages']}, Elements: {len(result['elements'])}")
