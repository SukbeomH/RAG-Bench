"""
BenchSpec — 벤치마크 조합 명세 및 프리셋

BenchSpec: 단일 벤치마크 실행 단위 (backend × pdf × mode)
PRESETS: 자주 쓰는 조합 모음
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ── 타입 정의 ─────────────────────────────────────────────────────────────────

Backend = Literal[
    "pymupdf",          # Category 1: 빠른 텍스트 추출 (PyMuPDF4LLM)
    "docling",          # Category 2: OCR + 레이아웃 (Docling)
    "openai",           # Category 3: VLM (GPT-4o)
    "openai-4.1",       # Category 3: VLM (GPT-4.1, 2025.04 최고성능)
    "upstage",          # Category 3: Document Parse API (auto mode)
    "upstage-enhanced", # Category 3: Document Parse API (enhanced mode, 차트·표 특화)
    "granite-vision",   # Category 3: Granite Vision 3.3 2B (K8s Ollama)
    "got-ocr2",         # Category 3: GOT-OCR2.0 580M (K8s FastAPI)
    "paddleocr-vl",     # Category 3: PaddleOCR-VL-1.5 0.9B (K8s FastAPI)
    "mineru",           # Phase 2 예정
]
ParseMode = Literal["direct", "document", "hybrid"]


# ── 개별 명세 ─────────────────────────────────────────────────────────────────

@dataclass
class BenchSpec:
    """
    단일 벤치마크 실행 명세.

    Attributes:
        backend:   사용할 파싱 백엔드
        pdf_name:  benchmark_pdfs/ 내 PDF 파일명 (확장자 포함)
        mode:      라우팅 모드
        gt_name:   gt/ 내 Ground Truth 파일명 (없으면 None)
    """
    backend: Backend
    pdf_name: str
    mode: ParseMode = "direct"
    gt_name: str | None = None

    @property
    def label(self) -> str:
        """Job 이름 / 결과 디렉토리용 레이블 (63자 이내, 특수문자 제거)."""
        pdf_stem = self.pdf_name.replace(".pdf", "").replace("_", "-")
        raw = f"{self.backend}-{pdf_stem}-{self.mode}"
        return raw[:63]


# ── 벤치마크 데이터셋 정의 ────────────────────────────────────────────────────

# Ground Truth 매핑: PDF → GT 파일
GT_MAP: dict[str, str] = {
    "text_only.pdf":        "text_only.md",
    "table_native.pdf":     "table_native.md",
    "table_image.pdf":      "table_native.md",   # 같은 내용의 래스터 버전
    "table_image_200dpi.pdf": "table_native.md",
    "table_image_150dpi.pdf": "table_native.md",
    "table_image_72dpi.pdf":  "table_native.md",
    "graph_rich.pdf":       "graph_rich.md",
    "graph_rich_image.pdf": "graph_rich.md",
    "graph_rich_image_200dpi.pdf": "graph_rich.md",
    "graph_rich_image_150dpi.pdf": "graph_rich.md",
    "graph_rich_image_72dpi.pdf":  "graph_rich.md",
}

ALL_PDFS = list(GT_MAP.keys())

EXISTING_BACKENDS: list[Backend] = ["pymupdf", "docling"]
VLM_BACKENDS: list[Backend] = ["openai", "upstage", "upstage-enhanced"]
OPENSOURCE_BACKENDS: list[Backend] = ["granite-vision", "got-ocr2", "paddleocr-vl"]
ALL_VLM_BACKENDS: list[Backend] = VLM_BACKENDS + OPENSOURCE_BACKENDS
ALL_BACKENDS: list[Backend] = [
    "pymupdf", "docling",
    "openai", "openai-4.1",
    "upstage", "upstage-enhanced",
    "granite-vision", "got-ocr2", "paddleocr-vl",
    "mineru",
]


# ── 프리셋 ────────────────────────────────────────────────────────────────────

def _make_specs(
    backends: list[Backend],
    pdfs: list[str],
    mode: ParseMode = "direct",
) -> list[BenchSpec]:
    return [
        BenchSpec(
            backend=b,
            pdf_name=p,
            mode=mode,
            gt_name=GT_MAP.get(p),
        )
        for b in backends
        for p in pdfs
    ]


PRESETS: dict[str, list[BenchSpec]] = {
    # 빠른 검증: pymupdf + docling × text_only
    "quick": _make_specs(EXISTING_BACKENDS, ["text_only.pdf"]),

    # Phase 1: pymupdf + docling × 11 PDF
    "phase1": _make_specs(EXISTING_BACKENDS, ALL_PDFS),

    # VLM 비교: openai / upstage × 11 PDF
    "vlm": _make_specs(VLM_BACKENDS, ALL_PDFS),

    # Upstage 전용: upstage / upstage-enhanced × 11 PDF
    "upstage-only": _make_specs(["upstage", "upstage-enhanced"], ALL_PDFS),

    # OCR 특화 오픈소스 모델: granite-vision / got-ocr2 / paddleocr-vl × 11 PDF
    "ocr": _make_specs(OPENSOURCE_BACKENDS, ALL_PDFS),

    # VLM 전체 비교: 상용 + 오픈소스 × 11 PDF
    "vlm-all": _make_specs(ALL_VLM_BACKENDS, ALL_PDFS),

    # Phase 2: 전체 백엔드 × 11 PDF
    "phase2": _make_specs(ALL_BACKENDS, ALL_PDFS),

    # 표 집중 테스트
    "tables": _make_specs(
        EXISTING_BACKENDS,
        ["table_native.pdf", "table_image.pdf", "table_image_200dpi.pdf",
         "table_image_150dpi.pdf", "table_image_72dpi.pdf"],
    ),

    # 그래프 집중 테스트
    "graphs": _make_specs(
        EXISTING_BACKENDS,
        ["graph_rich.pdf", "graph_rich_image.pdf", "graph_rich_image_200dpi.pdf",
         "graph_rich_image_150dpi.pdf", "graph_rich_image_72dpi.pdf"],
    ),
}


def get_preset(name: str) -> list[BenchSpec]:
    if name not in PRESETS:
        raise ValueError(f"알 수 없는 프리셋: '{name}'. 사용 가능: {list(PRESETS)}")
    return PRESETS[name]
