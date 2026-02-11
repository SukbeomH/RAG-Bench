"""
rag_bench — 모듈화 RAG 벤치마크 시스템

Strategy Pattern 기반으로 다양한 RAG 방식(Dense+Sparse, ColBERT, GraphRAG 등)을
비교 벤치마크할 수 있는 LangChain/LangGraph 통합 패키지.
"""

from rag_bench import config  # 전역 설정 및 .env 로드
from rag_bench.base import BaseRAGStrategy
from rag_bench.runner import BenchmarkRunner

__all__ = [
    "BaseRAGStrategy",
    "BenchmarkRunner",
]
