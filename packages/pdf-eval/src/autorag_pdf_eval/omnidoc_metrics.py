"""
OmniDocBench 호환 평가 메트릭 — Edit Distance, BLEU, METEOR, TEDS-HTML

CVPR 2025 OmniDocBench 벤치마크의 평가 방법론을 경량 구현.
- Edit_dist (NED): Normalized Edit Distance (Levenshtein)
- BLEU-4: n-gram 정밀도 (nltk)
- METEOR: 정밀도+재현율+어순 (nltk)
- TEDS-HTML: HTML 트리 편집 거리 (apted)
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


def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance (문자 단위)."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (c1 != c2),
                )
            )
        prev = curr
    return prev[-1]


def compute_text_ned(pred: str, gt: str) -> float:
    """
    Normalized Edit Distance 기반 텍스트 정확도.

    NED = 1 - (edit_distance / max(len(pred), len(gt)))
    반환값: 0.0 ~ 1.0  (1.0 = 완벽 일치)
    """
    pred = re.sub(r"\s+", " ", pred.strip())
    gt = re.sub(r"\s+", " ", gt.strip())

    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0

    if max_len > 5000:
        pred_words = pred.split()
        gt_words = gt.split()
        dist = _levenshtein(" ".join(pred_words), " ".join(gt_words))
        max_len = max(len(" ".join(pred_words)), len(" ".join(gt_words)))
    else:
        dist = _levenshtein(pred, gt)

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

    # wordnet 자동 다운로드 (quiet=True → 이미 있으면 무시)
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
    """마크다운에서 테이블 추출. evaluator.py _extract_tables와 동일 로직."""
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
    """마크다운 테이블 행 → HTML <table> 변환."""
    if not rows:
        return "<table></table>"

    parts = ["<table>"]
    for i, row in enumerate(rows):
        parts.append("<tr>")
        tag = "th" if i == 0 else "td"
        for cell in row:
            escaped = html.escape(cell)
            parts.append(f"<{tag}>{escaped}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


# ── TEDS-HTML ────────────────────────────────────────────────────────────────


def _html_to_tree(html_str: str):
    """HTML 문자열 → apted Node 트리."""

    # 간단한 재귀 파서: 태그 기반 트리 구축
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(html_str)
    except ET.ParseError:
        return None

    class _Node:
        def __init__(self, name: str, children: list | None = None):
            self.name = name
            self.children = children or []

    def _build(elem: ET.Element) -> _Node:
        tag = elem.tag
        text = (elem.text or "").strip()
        children = [_build(child) for child in elem]
        if text and not children:
            # 리프 노드: 태그 + 텍스트 내용
            return _Node(tag, [_Node(text)])
        return _Node(tag, children)

    return _build(root)


def _get_apted_config():
    """apted Config 서브클래스를 지연 생성."""
    from apted import Config as _BaseConfig

    class _Cfg(_BaseConfig):
        def rename(self, node1, node2):
            return 0 if node1.name == node2.name else 1

        def children(self, node):
            return node.children

    return _Cfg()


def compute_teds_html(pred: str, gt: str) -> float | None:
    """
    HTML 트리 기반 TEDS (0~1).

    마크다운에서 테이블을 추출 → HTML 변환 → apted 트리 편집 거리.
    GT에 테이블 없으면 -1.0, pred에 테이블 없으면 0.0.
    """
    try:
        from apted import APTED
    except ImportError:
        return None

    gt_tables = _extract_md_tables(gt)
    pred_tables = _extract_md_tables(pred)

    if not gt_tables:
        return -1.0
    if not pred_tables:
        return 0.0

    scores: list[float] = []
    for gt_tbl in gt_tables:
        gt_html = _md_table_to_html(gt_tbl)
        gt_tree = _html_to_tree(gt_html)
        if gt_tree is None:
            continue

        best = 0.0
        for pred_tbl in pred_tables:
            pred_html = _md_table_to_html(pred_tbl)
            pred_tree = _html_to_tree(pred_html)
            if pred_tree is None:
                continue

            apted_inst = APTED(pred_tree, gt_tree, _get_apted_config())
            edit_dist = apted_inst.compute_edit_distance()

            # 트리 크기: 노드 수
            def _count(node) -> int:
                return 1 + sum(_count(c) for c in node.children)

            max_nodes = max(_count(pred_tree), _count(gt_tree))
            if max_nodes == 0:
                sim = 1.0
            else:
                sim = max(0.0, 1.0 - edit_dist / max_nodes)
            best = max(best, sim)

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
