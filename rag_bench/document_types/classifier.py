"""
문서 종류 분류기 (classifier).

1단계: 키워드 기반 빠른 분류 (점수 투표)
2단계 (optional): LLM 기반 정교한 분류

외부 인터페이스:
  classify_document(text: str, *, override: str | None) -> DocType
  classify_file(path: str | Path, *, override: str | None) -> DocType
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from rag_bench.document_types.types import DocType


# ---------------------------------------------------------------------------
# 키워드 사전 (카테고리 → 키워드 리스트)
# ---------------------------------------------------------------------------

_KEYWORD_MAP: Dict[DocType, list[str]] = {
    DocType.TECHNICAL: [
        # 영문 기술 용어
        "api", "sdk", "endpoint", "function", "class", "method", "parameter",
        "return", "exception", "interface", "module", "library", "framework",
        "http", "rest", "json", "xml", "yaml", "dockerfile", "kubernetes",
        "git", "ci/cd", "deploy", "server", "database", "sql", "nosql",
        # 한국어 기술 용어
        "함수", "메서드", "클래스", "라이브러리", "프레임워크", "데이터베이스",
        "서버", "배포", "빌드", "컴파일", "디버그", "개발 가이드", "기술 문서",
        "레퍼런스", "매뉴얼", "코드 예제", "소스코드",
    ],
    DocType.LEGAL: [
        # 한국어 법률 용어
        "계약서", "계약", "법률", "법령", "조항", "제1조", "제2조", "제3조",
        "당사자", "갑", "을", "병", "甲", "乙", "丙",
        "손해배상", "손해", "배상", "위약금", "위반", "분쟁", "소송", "판결",
        "법원", "헌법", "민법", "형법", "상법", "행정법",
        "계약기간", "해지", "해약", "해제", "효력", "서명", "날인",
        "원고", "피고", "항소", "상고", "판례", "판결문",
        # 영문 법률 용어
        "contract", "agreement", "clause", "party", "parties", "liable",
        "liability", "indemnify", "indemnification", "breach", "terminate",
        "arbitration", "jurisdiction", "governing law",
    ],
    DocType.BUSINESS: [
        # 한국어 금융/경영 용어
        "매출", "영업이익", "순이익", "자산", "부채", "자본", "재무제표",
        "손익계산서", "대차대조표", "현금흐름표", "감사보고서", "사업보고서",
        "주주", "배당", "주가", "시가총액", "공시", "상장",
        "예산", "결산", "분기", "연간", "전년대비", "전분기대비",
        "투자", "수익률", "roi", "ebitda", "per", "pbr",
        # 공공/행정 용어
        "정책", "공고", "공시", "입찰", "조달", "예산안", "기획",
        "행정", "규정", "지침", "고시", "훈령",
        # 영문
        "revenue", "profit", "financial", "quarterly", "annual report",
        "balance sheet", "income statement", "cash flow",
    ],
    DocType.MEDICAL: [
        # 한국어 의료/보건 용어
        "환자", "의사", "병원", "진단", "치료", "처방", "약물", "증상",
        "질병", "질환", "감염", "바이러스", "백신", "예방접종",
        "공중보건", "보건소", "의료", "임상", "수술",
        "혈압", "혈당", "콜레스테롤", "bmi",
        "코로나", "독감", "폐렴", "암", "당뇨", "고혈압",
        # 영문 의료 용어
        "patient", "doctor", "hospital", "diagnosis", "treatment",
        "medication", "symptom", "disease", "infection", "vaccine",
        "public health", "cdc", "who", "clinical", "medical",
        "covid", "influenza", "cancer", "diabetes", "hypertension",
    ],
    DocType.GENERAL: [
        # 백과사전/위키 스타일
        "위키백과", "wikipedia", "백과사전", "역사", "문화", "예술",
        "인물", "지리", "국가", "도시", "자연", "과학", "수학",
        "사회", "경제", "정치", "교육", "스포츠", "음식",
        # 일반 서술 패턴
        "~이다", "~이란", "~의 역사", "~의 개요", "~의 특징",
        "개요", "정의", "배경", "유래",
    ],
}


# ---------------------------------------------------------------------------
# 키워드 기반 분류
# ---------------------------------------------------------------------------

def _score_text(text: str) -> Dict[DocType, int]:
    """텍스트에서 각 카테고리별 키워드 점수를 계산한다."""
    text_lower = text.lower()
    scores: Dict[DocType, int] = {dt: 0 for dt in DocType}

    for doc_type, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            # 단어 경계 고려 (한국어는 단어 경계가 없으므로 단순 포함 검사)
            if kw.lower() in text_lower:
                scores[doc_type] += 1

    return scores


def _pick_winner(scores: Dict[DocType, int], min_score: int = 2) -> Tuple[DocType, int]:
    """점수 투표에서 승자를 결정한다.

    Returns:
        (DocType, score) — 최고 점수 카테고리와 해당 점수.
        최고 점수가 min_score 미만이면 DocType.GENERAL 반환.
    """
    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    if best_score < min_score:
        return DocType.GENERAL, best_score

    return best, best_score


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def classify_document(
    text: str,
    *,
    override: Optional[str] = None,
    min_score: int = 2,
    sample_chars: int = 5000,
) -> DocType:
    """텍스트를 분석하여 문서 종류를 반환한다.

    Args:
        text: 분류할 텍스트 (전체 또는 일부).
        override: 수동 지정 (DocType 문자열 값). 지정 시 분류 스킵.
        min_score: 이 점수 미만이면 GENERAL로 폴백.
        sample_chars: 분류에 사용할 최대 문자 수 (앞부분 우선).

    Returns:
        DocType 열거형 값.
    """
    if override:
        try:
            return DocType(override.lower())
        except ValueError:
            valid = [dt.value for dt in DocType]
            raise ValueError(f"유효하지 않은 doc_type: '{override}'. 허용 값: {valid}")

    # 앞부분 텍스트로 빠른 분류
    sample = text[:sample_chars] if len(text) > sample_chars else text
    scores = _score_text(sample)
    doc_type, score = _pick_winner(scores, min_score=min_score)

    return doc_type


def classify_file(
    path: "str | Path",
    *,
    override: Optional[str] = None,
    sample_chars: int = 5000,
) -> DocType:
    """파일 경로에서 문서 종류를 판별한다.

    파일명 힌트(키워드)를 1차로 확인하고, 내용 기반 분류를 보조한다.

    Args:
        path: 파일 경로.
        override: 수동 지정 (DocType 문자열 값).
        sample_chars: 분류에 사용할 최대 문자 수.

    Returns:
        DocType 열거형 값.
    """
    if override:
        return classify_document("", override=override)

    path = Path(path)
    filename = path.stem.lower()

    # 파일명 힌트 우선 확인
    _FILENAME_HINTS: Dict[str, DocType] = {
        "contract": DocType.LEGAL,
        "agreement": DocType.LEGAL,
        "law": DocType.LEGAL,
        "legal": DocType.LEGAL,
        "계약": DocType.LEGAL,
        "법률": DocType.LEGAL,
        "financial": DocType.BUSINESS,
        "finance": DocType.BUSINESS,
        "report": DocType.BUSINESS,
        "보고서": DocType.BUSINESS,
        "재무": DocType.BUSINESS,
        "medical": DocType.MEDICAL,
        "health": DocType.MEDICAL,
        "clinical": DocType.MEDICAL,
        "의료": DocType.MEDICAL,
        "보건": DocType.MEDICAL,
        "api": DocType.TECHNICAL,
        "guide": DocType.TECHNICAL,
        "manual": DocType.TECHNICAL,
        "readme": DocType.TECHNICAL,
        "개발": DocType.TECHNICAL,
        "기술": DocType.TECHNICAL,
    }

    for hint, doc_type in _FILENAME_HINTS.items():
        if hint in filename:
            return doc_type

    # 파일 내용 읽기 (지원 포맷: txt, md)
    try:
        if path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            return classify_document(text, sample_chars=sample_chars)
    except (OSError, PermissionError):
        pass

    return DocType.GENERAL


def describe_classification(text: str, sample_chars: int = 5000) -> Dict[str, int]:
    """분류 점수 상세 정보 반환 (디버깅/설명용)."""
    sample = text[:sample_chars] if len(text) > sample_chars else text
    scores = _score_text(sample)
    return {dt.value: score for dt, score in sorted(scores.items(), key=lambda x: -x[1])}
