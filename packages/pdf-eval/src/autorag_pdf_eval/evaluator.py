"""
벤치마크 평가 모듈 — OmniDoc 메트릭 기반

평가 지표 (OmniDocScore):
  - Edit Distance (NED): 텍스트 정확도     → 0~1
  - BLEU-4: n-gram 정밀도                  → 0~100
  - METEOR: 정밀도+재현율+어순             → 0~100
  - TEDS-HTML: 표 구조 유사도              → 0~1
  - Word Count / Structure: 구조 보존 지표
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autorag_pdf_eval.omnidoc_metrics import OmniDocScore


# ── 데이터 구조 ───────────────────────────────────────────────────────────────


@dataclass
class PageScore:
    """페이지 단위 평가 결과."""

    page: int
    speed_s: float
    word_count: int
    has_headers: bool = False
    has_tables: bool = False
    has_formulas: bool = False
    omnidoc: OmniDocScore | None = None


@dataclass
class BenchResult:
    """전체 PDF 단위 평가 결과."""

    backend: str
    pdf_name: str
    mode: str
    pages: list[PageScore] = field(default_factory=list)

    # ── OmniDoc 기반 avg 프로퍼티 ────────────────────────────────────────────

    @property
    def avg_edit_dist(self) -> float | None:
        vals = [
            p.omnidoc.edit_dist
            for p in self.pages
            if p.omnidoc and p.omnidoc.edit_dist is not None
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    @property
    def avg_bleu(self) -> float | None:
        vals = [
            p.omnidoc.bleu
            for p in self.pages
            if p.omnidoc and p.omnidoc.bleu is not None
        ]
        return round(sum(vals) / len(vals), 2) if vals else None

    @property
    def avg_meteor(self) -> float | None:
        vals = [
            p.omnidoc.meteor
            for p in self.pages
            if p.omnidoc and p.omnidoc.meteor is not None
        ]
        return round(sum(vals) / len(vals), 2) if vals else None

    @property
    def avg_teds_html(self) -> float | None:
        vals = [
            p.omnidoc.teds_html
            for p in self.pages
            if p.omnidoc
            and p.omnidoc.teds_html is not None
            and p.omnidoc.teds_html >= 0
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    # ── 속도/분량 프로퍼티 ───────────────────────────────────────────────────

    @property
    def avg_speed(self) -> float:
        return sum(p.speed_s for p in self.pages) / max(len(self.pages), 1)

    @property
    def total_time_s(self) -> float:
        return sum(p.speed_s for p in self.pages)

    @property
    def total_words(self) -> int:
        return sum(p.word_count for p in self.pages)

    def summary_line(self) -> str:
        ned_str = (
            f"{self.avg_edit_dist:.3f}" if self.avg_edit_dist is not None else "  N/A"
        )
        bleu_str = f"{self.avg_bleu:.1f}" if self.avg_bleu is not None else " N/A"
        meteor_str = f"{self.avg_meteor:.1f}" if self.avg_meteor is not None else " N/A"
        teds_str = (
            f"{self.avg_teds_html:.3f}" if self.avg_teds_html is not None else "  N/A"
        )
        return (
            f"{self.backend:<12} {self.pdf_name:<35} "
            f"NED={ned_str}  TEDS-H={teds_str}  "
            f"BLEU={bleu_str}  METEOR={meteor_str}  "
            f"speed={self.avg_speed:.2f}s/p  words={self.total_words}"
        )


# ── 구조 지표 ─────────────────────────────────────────────────────────────────


def compute_structure(text: str) -> dict[str, bool | int]:
    """마크다운 구조 지표 반환."""
    return {
        "has_headers": bool(re.search(r"^#{1,4}\s", text, re.MULTILINE)),
        "has_tables": "|" in text,
        "has_formulas": "$" in text,
        "has_code": "```" in text,
        "word_count": len(text.split()),
        "line_count": len(text.splitlines()),
    }


# ── 페이지 단위 평가 래퍼 ─────────────────────────────────────────────────────


def evaluate_page(
    pred_text: str,
    gt_text: str | None,
    page_num: int,
    speed_s: float,
) -> PageScore:
    """단일 페이지 평가 — OmniDoc 메트릭만 사용."""
    struct = compute_structure(pred_text)

    omnidoc = None
    if gt_text:
        from autorag_pdf_eval.omnidoc_metrics import compute_omnidoc_scores

        omnidoc = compute_omnidoc_scores(pred_text, gt_text)

    return PageScore(
        page=page_num,
        speed_s=speed_s,
        word_count=struct["word_count"],
        has_headers=struct["has_headers"],
        has_tables=struct["has_tables"],
        has_formulas=struct["has_formulas"],
        omnidoc=omnidoc,
    )


def evaluate_document(
    pred_text: str,
    gt_text: str | None,
    speed_s: float,
    backend: str,
    pdf_name: str,
    mode: str,
) -> BenchResult:
    """
    문서 전체를 단일 페이지로 평가 (로컬 direct 모드 기본).

    페이지 분리가 필요하면 페이지별 evaluate_page()를 직접 호출하고
    BenchResult.pages에 누적하세요.
    """
    result = BenchResult(backend=backend, pdf_name=pdf_name, mode=mode)
    score = evaluate_page(pred_text, gt_text, page_num=1, speed_s=speed_s)
    result.pages.append(score)
    return result
