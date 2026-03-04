"""
document_types — 문서 종류 분류 시스템.

5개 카테고리로 문서를 분류하고 메타데이터를 제공한다.
"""

from autorag_retrieval.document_types.types import DocType, DOC_TYPE_METADATA

__all__ = [
    "DocType",
    "DOC_TYPE_METADATA",
]
