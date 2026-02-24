"""
DocType — 문서 종류 열거형 + 메타데이터.

5개 카테고리와 각 카테고리별 샘플링 비율, 청킹 전략,
표준 HuggingFace 벤치마크 데이터셋 정보를 제공한다.
"""

from enum import Enum
from typing import Dict, List, Optional, Any


class DocType(str, Enum):
    """문서 종류 열거형."""

    TECHNICAL = "technical"   # API 문서, 개발 가이드, 기술 매뉴얼
    LEGAL     = "legal"       # 법률, 계약서, 판례 (markers_bm law)
    BUSINESS  = "business"    # 금융/공공/상업 보고서 (markers_bm finance+public+commerce)
    MEDICAL   = "medical"     # 의료/공중보건 FAQ (publichealth-qa CDC/WHO)
    GENERAL   = "general"     # 백과사전/위키 (MIRACL + Ko-StrategyQA + Belebele + MrTiDy)


# ---------------------------------------------------------------------------
# 카테고리별 메타데이터
# ---------------------------------------------------------------------------

DOC_TYPE_METADATA: Dict[DocType, Dict[str, Any]] = {
    DocType.TECHNICAL: {
        # 샘플링 비율 (전체 텍스트 대비)
        "sampling_ratio": 0.15,
        # 청킹 강조 전략
        "chunk_emphasis": "code+structure",
        # 대응 HuggingFace 데이터셋 (없으면 None)
        "hf_dataset": None,          # 사용자 업로드 문서 기반
        "hf_subset": None,
        "secondary_datasets": [],
        # 예상 상위 조합 (리서치 기반 사전 추정)
        "expected_top": "e5+korean_bm25",
        # 설명
        "description": "API 문서, 개발 가이드, 기술 매뉴얼. 코드+구조 밀집 문서.",
    },
    DocType.LEGAL: {
        "sampling_ratio": 0.20,
        "chunk_emphasis": "precision",
        "hf_dataset": "yjoonjang/markers_bm",
        "hf_subset": "law",
        "secondary_datasets": [],
        "expected_top": "snowflake-ko+korean_bm25",
        "description": "법률, 계약서, 판례. AutoRAG 법률 SOTA: snowflake-ko +9.5%.",
    },
    DocType.BUSINESS: {
        "sampling_ratio": 0.20,
        "chunk_emphasis": "summary+figures",
        "hf_dataset": "yjoonjang/markers_bm",
        "hf_subset": "finance+public+commerce",
        "secondary_datasets": [],
        "expected_top": "snowflake-ko+korean_bm25",
        "description": "금융/공공/상업 보고서. 수치+요약 밀집 문서.",
    },
    DocType.MEDICAL: {
        "sampling_ratio": 1.00,      # 소규모 데이터셋이므로 전량 사용
        "chunk_emphasis": "faq",
        "hf_dataset": "xhluca/publichealth-qa",
        "hf_subset": "korean",
        "secondary_datasets": [],
        "expected_top": "snowflake-ko+splade",
        "description": "의료/공중보건 FAQ. CDC/WHO 기반 77개 QA.",
    },
    DocType.GENERAL: {
        "sampling_ratio": 0.10,
        "chunk_emphasis": "balanced",
        "hf_dataset": "miracl/miracl",
        "hf_subset": "ko",           # primary
        "secondary_datasets": [
            ("taeminlee/Ko-StrategyQA", {}),
            ("facebook/belebele", {"name": "kor_Hang"}),
            ("mteb/mrtidy", {"name": "korean-corpus"}),
        ],
        "expected_top": "bge-m3+splade",
        "description": "백과사전/위키. MIRACL(primary) + Ko-StrategyQA + Belebele + MrTiDy.",
    },
}


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def get_hf_dataset(doc_type: DocType) -> Optional[str]:
    """카테고리의 주 HuggingFace 데이터셋 이름 반환."""
    return DOC_TYPE_METADATA[doc_type].get("hf_dataset")


def get_sampling_ratio(doc_type: DocType) -> float:
    """카테고리별 샘플링 비율 반환."""
    return DOC_TYPE_METADATA[doc_type]["sampling_ratio"]


def list_doc_types() -> List[str]:
    """사용 가능한 문서 종류 목록 반환."""
    return [dt.value for dt in DocType]
