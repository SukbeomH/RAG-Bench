"""
evaluation 서브패키지 — RAGAS 기반 RAG 평가.

하위 호환:
    from rag_bench.evaluation import RAGEvaluator  # legacy.py
    from rag_bench.evaluation import ExtendedRAGEvaluator  # evaluator.py
"""

from rag_bench.evaluation.legacy import RAGEvaluator
from rag_bench.evaluation.evaluator import ExtendedRAGEvaluator, EvaluationReport, rank_strategies

__all__ = [
    "RAGEvaluator",
    "ExtendedRAGEvaluator",
    "EvaluationReport",
    "rank_strategies",
]
