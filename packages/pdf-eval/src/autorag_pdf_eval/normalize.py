"""
마크다운 정규화 모듈 — 파서 출력과 GT 간 서식 차이를 통일.

각 정규화 규칙은 개별 함수로 분리되어 있으며,
normalize_markdown()은 모든 규칙을 순서대로 적용합니다.
diff_report()로 어떤 규칙이 실제로 텍스트를 변경했는지 확인할 수 있습니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class NormalizationLog:
    """정규화 적용 결과 — 각 규칙별 변경 여부와 변경 횟수."""

    applied: dict[str, int] = field(default_factory=dict)

    @property
    def total_changes(self) -> int:
        return sum(self.applied.values())

    def summary(self) -> str:
        if not self.applied:
            return "변경 없음"
        parts = [f"{name}: {count}건" for name, count in self.applied.items() if count]
        return ", ".join(parts) if parts else "변경 없음"


# ── 개별 정규화 규칙 ──────────────────────────────────────────────────────────


def strip_code_block_wrapper(text: str) -> tuple[str, int]:
    """코드블록 래퍼 제거 — VLM이 ```markdown ... ``` 으로 감싸는 경우.

    GPT-4o 등이 전체 출력을 코드블록으로 감싸는 패턴을 제거합니다.
    페이지 단위로 여러 코드블록이 있을 수 있어 반복 처리합니다.
    """
    # ```markdown 또는 ``` 으로 시작하여 ``` 으로 끝나는 블록
    pattern = r"^```(?:markdown|md)?\s*\n(.*?)^```\s*$"
    matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
    if not matches:
        return text, 0

    result = text
    count = 0
    for match in reversed(matches):  # 뒤에서부터 치환 (인덱스 보존)
        result = result[: match.start()] + match.group(1) + result[match.end() :]
        count += 1

    return result, count


def unify_bullet_markers(text: str) -> tuple[str, int]:
    """불릿 기호 통일 — *, +, • 를 모두 - 로 변환.

    마크다운 리스트에서 파서마다 다른 불릿 기호를 사용합니다:
    - GT: * (asterisk)
    - paddleocr-vl: • (bullet) 또는 -
    - pymupdf: • (bullet)
    - openai: - (hyphen)
    """
    count = 0

    def _replace_bullet(m: re.Match) -> str:
        nonlocal count
        count += 1
        return m.group(1) + "-" + m.group(2)

    # 줄 시작(또는 들여쓰기) + 불릿마커 + 공백
    result = re.sub(
        r"^(\s*)[*+•](\s)",
        _replace_bullet,
        text,
        flags=re.MULTILINE,
    )
    return result, count


def collapse_whitespace(text: str) -> tuple[str, int]:
    """연속 공백/줄바꿈 정규화.

    - 연속 공백(탭 포함) → 단일 공백 (줄 내부)
    - 3개 이상 연속 빈 줄 → 2개 빈 줄 (단락 구분 유지)
    - 각 줄의 후행 공백 제거
    """
    original = text
    # 줄 내부 연속 공백 → 단일 공백
    result = re.sub(r"[ \t]+", " ", text)
    # 각 줄 후행 공백 제거
    result = re.sub(r" +$", "", result, flags=re.MULTILINE)
    # 3+ 연속 빈 줄 → 2 빈 줄
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    count = 1 if result != original else 0
    return result, count


def strip_bold_in_headers(text: str) -> tuple[str, int]:
    """헤더 내 볼드 마커 제거 — # **제목** → # 제목.

    pymupdf가 헤더 텍스트를 볼드로 감싸는 패턴을 정규화합니다.
    """
    count = 0

    def _strip(m: re.Match) -> str:
        nonlocal count
        count += 1
        return m.group(1) + m.group(2)

    result = re.sub(
        r"^(#{1,6}\s+)\*\*(.+?)\*\*\s*$",
        _strip,
        text,
        flags=re.MULTILINE,
    )
    return result, count


def strip_blockquote_markers(text: str) -> tuple[str, int]:
    """인용구 마커(> ) 제거.

    일부 VLM(openai 등)이 강조 텍스트를 blockquote로 변환하지만,
    GT에는 인용구 마커가 없는 경우가 많습니다.
    마크다운 blockquote를 일반 텍스트로 변환합니다.
    """
    count = 0

    def _strip(m: re.Match) -> str:
        nonlocal count
        count += 1
        return m.group(1)

    result = re.sub(r"^>\s?(.*)$", _strip, text, flags=re.MULTILINE)
    return result, count


def strip_vlm_location_tokens(text: str) -> tuple[str, int]:
    """VLM 위치 토큰 제거 — <|LOC_XX|>, <|SEP|> 등 특수 토큰.

    PaddleOCR-VL 등 일부 VLM이 bounding box 좌표를 나타내는
    위치 토큰(<|LOC_숫자|>)을 텍스트에 삽입합니다.
    이 토큰은 마크다운 내용과 무관하므로 제거합니다.
    """
    pattern = r"<\|[A-Z_]+(?:_\d+)?\|>"
    matches = list(re.finditer(pattern, text))
    count = len(matches)
    if count == 0:
        return text, 0
    result = re.sub(pattern, "", text)
    return result, count


def normalize_table_whitespace(text: str) -> tuple[str, int]:
    """테이블 셀 내부 공백 정규화.

    테이블 셀의 앞뒤 공백을 정리하고, 셀 내부 연속 공백을 단일 공백으로 변환합니다.
    서로 다른 파서가 테이블 셀에 추가하는 패딩 차이를 제거합니다.
    """
    original = text
    lines = text.splitlines()
    result_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            # 셀 내부 공백 정규화
            cells = stripped.split("|")
            cells = [c.strip() for c in cells]
            result_lines.append("| " + " | ".join(cells[1:-1]) + " |")
        else:
            result_lines.append(line)
    result = "\n".join(result_lines)
    count = 1 if result != original else 0
    return result, count


# ── 통합 정규화 ────────────────────────────────────────────────────────────────


# 정규화 규칙 목록 (순서 중요: 코드블록 제거 → 불릿 → 공백 순)
RULES: list[tuple[str, callable]] = [
    ("code_block_wrapper", strip_code_block_wrapper),
    ("vlm_location_tokens", strip_vlm_location_tokens),
    ("bullet_markers", unify_bullet_markers),
    ("bold_in_headers", strip_bold_in_headers),
    ("blockquote_markers", strip_blockquote_markers),
    ("table_whitespace", normalize_table_whitespace),
    ("whitespace", collapse_whitespace),  # 마지막에 적용
]

# 보고서용 규칙 설명
RULE_DESCRIPTIONS: dict[str, str] = {
    "vlm_location_tokens": (
        "VLM 위치 토큰(<|LOC_숫자|> 등)을 제거합니다. "
        "PaddleOCR-VL 등이 bounding box 좌표를 나타내는 특수 토큰을 텍스트에 삽입하며, "
        "이는 마크다운 내용과 무관한 메타데이터로 NED를 크게 하락시킵니다."
    ),
    "code_block_wrapper": (
        "VLM이 출력 전체를 ```markdown...``` 코드블록으로 감싸는 패턴을 제거합니다. "
        "GPT-4o 등이 마크다운 응답을 코드 펜스로 래핑하여 NED 계산 시 불필요한 차이를 유발합니다."
    ),
    "bullet_markers": (
        "리스트 불릿 기호(*, +, •)를 하이픈(-)으로 통일합니다. "
        "GT는 * 사용, pymupdf는 •, openai는 - 등 파서마다 불릿 기호가 달라 "
        "실질적 내용 차이가 아닌 서식 차이로 NED가 하락합니다."
    ),
    "bold_in_headers": (
        "헤더 내부의 볼드 마커를 제거합니다 (# **제목** → # 제목). "
        "pymupdf가 헤더 텍스트를 볼드로 감싸는 패턴으로, 구조적 의미는 동일합니다."
    ),
    "blockquote_markers": (
        "인용구 마커(> )를 제거합니다. "
        "GPT-4o 등이 강조 텍스트를 blockquote로 변환하지만, GT에는 해당 마커가 없어 불필요한 차이를 유발합니다."
    ),
    "table_whitespace": (
        "마크다운 테이블 셀 내부의 공백을 정규화합니다. "
        "파서마다 테이블 셀의 패딩이 다르지만 실제 테이블 내용은 동일한 경우를 처리합니다."
    ),
    "whitespace": (
        "연속 공백을 단일 공백으로, 3개 이상 빈 줄을 2개로, 줄 끝 공백을 제거합니다. "
        "모든 파서에 공통적으로 존재하는 공백 차이로, 의미 있는 내용 차이가 아닙니다."
    ),
}


def normalize_markdown(text: str) -> tuple[str, NormalizationLog]:
    """모든 정규화 규칙을 순서대로 적용.

    Returns:
        (정규화된 텍스트, 적용 로그)
    """
    log = NormalizationLog()
    result = text

    for name, fn in RULES:
        result, count = fn(result)
        if count > 0:
            log.applied[name] = count

    return result.strip(), log
