"""
벤치마크 평가 모듈 — Text NED / Table TEDS / 구조 지표

평가 지표:
  - Text NED  : 1 - (Levenshtein edit distance / max_len)  → 0~1, 높을수록 좋음
  - Table TEDS: 표 구조 유사도 (Tree Edit Distance 기반)   → 0~1, 높을수록 좋음
  - Word Count: 단어 수 (추출량 지표)
  - Structure : headers/tables/formulas 유무 (구조 보존)

참고:
  - TEDS 원본 논문: "PubTabNet" (Zhong et al., 2020)
  - 단순화 구현: HTML 변환 대신 마크다운 테이블 구조 비교
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


# ── 데이터 구조 ───────────────────────────────────────────────────────────────

@dataclass
class PageScore:
    """페이지 단위 평가 결과."""
    page: int
    text_ned: float          # 0.0 ~ 1.0 (1.0 = 완벽 일치)
    table_teds: float        # 0.0 ~ 1.0 (표 없으면 -1.0)
    speed_s: float           # 처리 시간 (초)
    word_count: int
    has_headers: bool = False
    has_tables: bool = False
    has_formulas: bool = False


@dataclass
class BenchResult:
    """전체 PDF 단위 평가 결과."""
    backend: str
    pdf_name: str
    mode: str
    pages: list[PageScore] = field(default_factory=list)

    @property
    def avg_text_ned(self) -> float:
        valid = [p.text_ned for p in self.pages if p.text_ned >= 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_table_teds(self) -> float:
        valid = [p.table_teds for p in self.pages if p.table_teds >= 0]
        return sum(valid) / len(valid) if valid else -1.0

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
        teds_str = f"{self.avg_table_teds:.3f}" if self.avg_table_teds >= 0 else "  N/A "
        return (
            f"{self.backend:<12} {self.pdf_name:<35} "
            f"NED={self.avg_text_ned:.3f}  TEDS={teds_str}  "
            f"speed={self.avg_speed:.2f}s/p  words={self.total_words}"
        )


# ── Text NED ──────────────────────────────────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance (문자 단위)."""
    # 짧은 문자열을 s1으로 고정해 메모리 절약
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            curr.append(min(
                prev[j] + 1,        # 삭제
                curr[j - 1] + 1,    # 삽입
                prev[j - 1] + (c1 != c2),  # 교체
            ))
        prev = curr
    return prev[-1]


def compute_text_ned(pred: str, gt: str) -> float:
    """
    Normalized Edit Distance 기반 텍스트 정확도.

    NED = 1 - (edit_distance / max(len(pred), len(gt)))
    반환값: 0.0 ~ 1.0  (1.0 = 완벽 일치)
    """
    # 공백 정규화 (연속 공백 → 단일 공백, 줄바꿈 통일)
    pred = re.sub(r'\s+', ' ', pred.strip())
    gt   = re.sub(r'\s+', ' ', gt.strip())

    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0

    # 긴 텍스트는 워드 단위로 계산 (속도)
    if max_len > 5000:
        pred_words = pred.split()
        gt_words   = gt.split()
        dist = _levenshtein(' '.join(pred_words), ' '.join(gt_words))
        max_len = max(len(' '.join(pred_words)), len(' '.join(gt_words)))
    else:
        dist = _levenshtein(pred, gt)

    return max(0.0, 1.0 - dist / max_len)


# ── Table TEDS ────────────────────────────────────────────────────────────────

def _extract_tables(text: str) -> list[list[list[str]]]:
    """
    마크다운 텍스트에서 테이블 추출.

    반환: 테이블 목록, 각 테이블은 행×열 문자열 2D 리스트
    """
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            # 구분자 행 (|---|---|) 무시
            if re.match(r'^\|[-:\s|]+\|$', stripped):
                continue
            cells = [c.strip() for c in stripped[1:-1].split('|')]
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []

    if current:
        tables.append(current)

    return tables


def _table_ned(t1: list[list[str]], t2: list[list[str]]) -> float:
    """두 테이블 간 셀 단위 NED 평균."""
    # 행/열 수 맞추기
    rows = max(len(t1), len(t2))
    cols = max(
        max((len(r) for r in t1), default=0),
        max((len(r) for r in t2), default=0),
    )
    if rows == 0 or cols == 0:
        return 1.0

    scores: list[float] = []
    for i in range(rows):
        for j in range(cols):
            c1 = t1[i][j] if i < len(t1) and j < len(t1[i]) else ""
            c2 = t2[i][j] if i < len(t2) and j < len(t2[i]) else ""
            scores.append(compute_text_ned(c1, c2))

    return sum(scores) / len(scores)


def compute_table_teds(pred: str, gt: str) -> float:
    """
    마크다운 기반 Table TEDS 근사.

    원본 TEDS(HTML 트리 기반)의 단순화 버전.
    - 테이블 수 불일치 시 패널티 적용
    - 각 테이블 쌍의 셀 NED 평균 반환

    반환값:
      -1.0  : GT에 테이블 없음 (지표 해당 없음)
      0.0~1.0: 유사도 (1.0 = 완벽)
    """
    gt_tables   = _extract_tables(gt)
    pred_tables = _extract_tables(pred)

    if not gt_tables:
        return -1.0   # GT에 표 없음 → 지표 미산출

    if not pred_tables:
        return 0.0    # 파서가 표를 추출하지 못함

    # 테이블 수 불일치 패널티
    count_penalty = min(len(pred_tables), len(gt_tables)) / max(len(pred_tables), len(gt_tables))

    # 각 GT 테이블과 가장 유사한 pred 테이블 매칭
    scores: list[float] = []
    for gt_tbl in gt_tables:
        best = max((_table_ned(p_tbl, gt_tbl) for p_tbl in pred_tables), default=0.0)
        scores.append(best)

    return (sum(scores) / len(scores)) * count_penalty


# ── 구조 지표 ─────────────────────────────────────────────────────────────────

def compute_structure(text: str) -> dict[str, bool | int]:
    """마크다운 구조 지표 반환."""
    return {
        "has_headers":   bool(re.search(r'^#{1,4}\s', text, re.MULTILINE)),
        "has_tables":    '|' in text,
        "has_formulas":  '$' in text,
        "has_code":      '```' in text,
        "word_count":    len(text.split()),
        "line_count":    len(text.splitlines()),
    }


# ── 페이지 단위 평가 래퍼 ─────────────────────────────────────────────────────

def evaluate_page(
    pred_text: str,
    gt_text: str | None,
    page_num: int,
    speed_s: float,
) -> PageScore:
    """
    단일 페이지 평가.

    Args:
        pred_text: 파서 출력 마크다운
        gt_text:   Ground Truth 마크다운 (없으면 None)
        page_num:  페이지 번호
        speed_s:   처리 시간 (초)

    Returns:
        PageScore
    """
    struct = compute_structure(pred_text)

    if gt_text:
        text_ned    = compute_text_ned(pred_text, gt_text)
        table_teds  = compute_table_teds(pred_text, gt_text)
    else:
        text_ned    = -1.0
        table_teds  = -1.0

    return PageScore(
        page=page_num,
        text_ned=text_ned,
        table_teds=table_teds,
        speed_s=speed_s,
        word_count=struct["word_count"],
        has_headers=struct["has_headers"],
        has_tables=struct["has_tables"],
        has_formulas=struct["has_formulas"],
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
