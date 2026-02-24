"""
document_types — 문서 종류 분류 시스템.

서비스 벤치마크에서 문서 종류별 최적 RAG 모델 선정을 위해
문서를 5개 카테고리로 분류하고 샘플링 전략을 제공한다.

카테고리:
  - TECHNICAL : API 문서, 개발 가이드, 기술 매뉴얼
  - LEGAL     : 법률, 계약서, 판례
  - BUSINESS  : 금융/공공/상업 보고서
  - MEDICAL   : 의료/공중보건 FAQ
  - GENERAL   : 백과사전/위키
"""

from rag_bench.document_types.types import DocType, DOC_TYPE_METADATA
from rag_bench.document_types.classifier import classify_document, classify_file
from rag_bench.document_types.sampler import sample_document, sample_text

__all__ = [
    "DocType",
    "DOC_TYPE_METADATA",
    "classify_document",
    "classify_file",
    "sample_document",
    "sample_text",
]
