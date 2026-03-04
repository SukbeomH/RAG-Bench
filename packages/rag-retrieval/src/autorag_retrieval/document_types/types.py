"""
DocType — 문서 종류 열거형 + 메타데이터.

5개 카테고리와 각 카테고리별 샘플링 비율, 청킹 전략,
표준 HuggingFace 벤치마크 데이터셋 정보를 제공한다.
"""

from enum import Enum
from typing import Dict, List, Optional, Any


class DocType(str, Enum):
    """문서 종류 열거형."""

    TECHNICAL = "technical"  # API 문서, 개발 가이드, 기술 매뉴얼
    LEGAL = "legal"  # 법률, 계약서, 판례 (markers_bm law)
    BUSINESS = "business"  # 금융/공공/상업 보고서 (markers_bm finance+public+commerce)
    MEDICAL = "medical"  # 의료/공중보건 FAQ (publichealth-qa CDC/WHO)
    GENERAL = "general"  # 백과사전/위키 (MIRACL + Ko-StrategyQA + Belebele + MrTiDy)


# ---------------------------------------------------------------------------
# 카테고리별 메타데이터
# ---------------------------------------------------------------------------

DOC_TYPE_METADATA: Dict[DocType, Dict[str, Any]] = {
    DocType.TECHNICAL: {
        "sampling_ratio": 0.15,
        "chunk_emphasis": "code+structure",
        "hf_dataset": None,
        "hf_subset": None,
        "secondary_datasets": [],
        "expected_top": "e5+korean_bm25",
        "description": "API 문서, 개발 가이드, 기술 매뉴얼. 코드+구조 밀집 문서.",
        "label": "기술 (API·매뉴얼)",
        "data_source": "사용자 업로드 기술 문서 (별도 확보 필요)",
        "characteristic": "코드·명령어·버전 번호 등 구조화된 기술 용어 정밀 매칭 필요.",
    },
    DocType.LEGAL: {
        "sampling_ratio": 0.20,
        "chunk_emphasis": "precision",
        "hf_dataset": "yjoonjang/markers_bm",
        "hf_subset": "law",
        "secondary_datasets": [],
        "expected_top": "bge-m3+korean_bm25",
        "description": "법률, 계약서, 판례. 조문 번호·용어 정밀 매칭이 품질을 결정.",
        "label": "법률 (계약서·판례)",
        "data_source": "markers_bm 법률 판례·계약서 데이터셋",
        "characteristic": "'제N조 제N항'처럼 조문·번호 정확 매칭이 답변 품질을 결정.",
    },
    DocType.BUSINESS: {
        "sampling_ratio": 0.20,
        "chunk_emphasis": "summary+figures",
        "hf_dataset": "yjoonjang/markers_bm",
        "hf_subset": "finance+public+commerce",
        "secondary_datasets": [],
        "expected_top": "bge-m3+korean_bm25",
        "description": "금융/공공/상업 보고서. 수치+요약 밀집 문서.",
        "label": "금융·비즈니스 (보고서·공시)",
        "data_source": "markers_bm 금융·공공·상업 보고서 데이터셋",
        "characteristic": "수치·지표·회사명 등 고유명사 정확도가 높아야 하는 도메인.",
    },
    DocType.MEDICAL: {
        "sampling_ratio": 1.00,
        "chunk_emphasis": "faq",
        "hf_dataset": "xhluca/publichealth-qa",
        "hf_subset": "korean",
        "secondary_datasets": [],
        "expected_top": "bge-m3+splade",
        "description": "의료/공중보건 FAQ. CDC/WHO 기반 77개 QA.",
        "label": "의료 (FAQ·공중보건)",
        "data_source": "publichealth-qa CDC/WHO 한국어 FAQ, 77개 QA",
        "characteristic": "질환명·약어 등 전문 용어와 FAQ 형식 응답 정확도가 중요.",
    },
    DocType.GENERAL: {
        "sampling_ratio": 0.10,
        "chunk_emphasis": "balanced",
        "hf_dataset": "miracl/miracl",
        "hf_subset": "ko",
        "secondary_datasets": [
            ("taeminlee/Ko-StrategyQA", {}),
            ("facebook/belebele", {"name": "kor_Hang"}),
            ("mteb/mrtidy", {"name": "korean-corpus"}),
        ],
        "expected_top": "bge-m3+splade",
        "description": "백과사전/위키. MIRACL(primary) + Ko-StrategyQA + Belebele + MrTiDy.",
        "label": "일반 (백과사전·위키)",
        "data_source": "MIRACL(ko) 위키피디아 기반 국제 표준 데이터셋, 쿼리 1,081개",
        "characteristic": "위키피디아 수준의 대용량 범용 텍스트. 광범위한 주제 커버가 핵심.",
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
