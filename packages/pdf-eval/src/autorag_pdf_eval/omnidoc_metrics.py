"""
OmniDocBench 호환 평가 메트릭 — Edit Distance, BLEU, METEOR, TEDS-HTML

CVPR 2025 OmniDocBench 벤치마크의 평가 방법론을 구현.
- Edit_dist (NED): Normalized Edit Distance (rapidfuzz Levenshtein)
- BLEU-4: n-gram 정밀도 (nltk)
- METEOR: 정밀도+재현율+어순 (nltk)
- TEDS-HTML: IBM 원본 Tree Edit Distance based Similarity (lxml + apted)

TEDS 구현은 OmniDocBench (Apache-2.0) 및 IBM TEDS 원본을 벤더링.
https://github.com/opendatalab/OmniDocBench
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass


@dataclass
class OmniDocScore:
    """OmniDocBench 호환 메트릭 결과."""

    edit_dist: float | None = None  # 0~1 (기존 NED)
    bleu: float | None = None  # 0~100
    meteor: float | None = None  # 0~100
    teds_html: float | None = None  # 0~1 (-1.0 = N/A)


# ── Edit Distance (NED) ──────────────────────────────────────────────────────


def compute_text_ned(pred: str, gt: str) -> float:
    """
    Normalized Edit Distance 기반 텍스트 정확도.

    OmniDocBench Edit_dist 방식: 1 - (levenshtein / max_len).
    반환값: 0.0 ~ 1.0  (1.0 = 완벽 일치)
    """
    from rapidfuzz.distance import Levenshtein

    pred = re.sub(r"\s+", " ", pred.strip())
    gt = re.sub(r"\s+", " ", gt.strip())

    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0

    dist = Levenshtein.distance(pred, gt)
    return max(0.0, 1.0 - dist / max_len)


# ── BLEU ─────────────────────────────────────────────────────────────────────


def compute_bleu(pred: str, gt: str) -> float | None:
    """
    BLEU-4 점수 (0~100).

    OmniDocBench 방식: sentence_bleu + SmoothingFunction.method1.
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return None

    pred_tokens = pred.strip().split()
    gt_tokens = gt.strip().split()

    if not gt_tokens:
        return 0.0
    if not pred_tokens:
        return 0.0

    smoothie = SmoothingFunction().method1
    score = sentence_bleu(
        [gt_tokens],
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )
    return round(score * 100, 2)


# ── METEOR ───────────────────────────────────────────────────────────────────


def compute_meteor(pred: str, gt: str) -> float | None:
    """
    METEOR 점수 (0~100).

    첫 호출 시 wordnet 자동 다운로드.
    """
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score as _meteor
    except ImportError:
        return None

    nltk.download("wordnet", quiet=True)

    pred_tokens = pred.strip().split()
    gt_tokens = gt.strip().split()

    if not gt_tokens:
        return 0.0
    if not pred_tokens:
        return 0.0

    score = _meteor([gt_tokens], pred_tokens)
    return round(score * 100, 2)


# ── HTML 테이블 변환 ─────────────────────────────────────────────────────────


def _extract_md_tables(text: str) -> list[list[list[str]]]:
    """마크다운에서 테이블 추출."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if re.match(r"^\|[-:\s|]+\|$", stripped):
                continue
            cells = [c.strip() for c in stripped[1:-1].split("|")]
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []

    if current:
        tables.append(current)

    return tables


def _md_table_to_html(rows: list[list[str]]) -> str:
    """마크다운 테이블 행 → HTML 문서 변환.

    IBM TEDS는 ``body/table`` xpath를 사용하므로 완전한 HTML 문서로 생성.
    """
    if not rows:
        return "<html><body><table></table></body></html>"

    parts = ["<html><body><table>"]
    for i, row in enumerate(rows):
        parts.append("<tr>")
        tag = "th" if i == 0 else "td"
        for cell in row:
            escaped = html.escape(cell)
            parts.append(f"<{tag}>{escaped}</{tag}>")
        parts.append("</tr>")
    parts.append("</table></body></html>")
    return "".join(parts)


# ── TEDS-HTML (IBM 원본 via OmniDocBench) ────────────────────────────────────


def compute_teds_html(
    pred: str, gt: str, *, structure_only: bool = False
) -> float | None:
    """
    IBM TEDS (Tree Edit Distance based Similarity) (0~1).

    OmniDocBench에서 벤더링한 IBM 원본 TEDS 구현 사용:
    - lxml HTML 파싱 (malformed HTML 허용)
    - colspan/rowspan 지원
    - Levenshtein 기반 셀 토큰 비교
    - apted Tree 서브클래스 (TableTree)

    마크다운에서 테이블 추출 → HTML 변환 → IBM TEDS evaluate.
    GT에 테이블 없으면 -1.0, pred에 테이블 없으면 0.0.
    structure_only=True → 셀 내용 무시, 구조만 비교.
    """
    try:
        from autorag_pdf_eval._vendor_teds import TEDS
    except ImportError:
        return None

    gt_tables = _extract_md_tables(gt)
    pred_tables = _extract_md_tables(pred)

    if not gt_tables:
        return -1.0
    if not pred_tables:
        return 0.0

    teds = TEDS(structure_only=structure_only)

    scores: list[float] = []
    for gt_tbl in gt_tables:
        gt_html = _md_table_to_html(gt_tbl)

        best = 0.0
        for pred_tbl in pred_tables:
            pred_html = _md_table_to_html(pred_tbl)
            score = teds.evaluate(pred_html, gt_html)
            best = max(best, score)

        scores.append(best)

    if not scores:
        return -1.0

    # 테이블 수 불일치 패널티
    count_penalty = min(len(pred_tables), len(gt_tables)) / max(
        len(pred_tables), len(gt_tables)
    )
    return round((sum(scores) / len(scores)) * count_penalty, 4)


# ── 통합 계산 ────────────────────────────────────────────────────────────────


def compute_omnidoc_scores(pred: str, gt: str) -> OmniDocScore:
    """전체 OmniDocBench 메트릭 한번에 계산."""
    return OmniDocScore(
        edit_dist=compute_text_ned(pred, gt),
        bleu=compute_bleu(pred, gt),
        meteor=compute_meteor(pred, gt),
        teds_html=compute_teds_html(pred, gt),
    )
