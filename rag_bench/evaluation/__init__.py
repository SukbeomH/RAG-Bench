"""
evaluation 서브패키지 — RAGAS 기반 RAG 평가.

하위 호환:
    from rag_bench.evaluation import RAGEvaluator  # legacy.py
    from rag_bench.evaluation import ExtendedRAGEvaluator  # evaluator.py
"""

from rag_bench.evaluation.legacy import RAGEvaluator
from rag_bench.evaluation.evaluator import (
    ExtendedRAGEvaluator,
    EvaluationReport,
    SCORING_PROFILES,
    rank_strategies,
)
from rag_bench.evaluation.metrics import MetricPreset, create_metrics

__all__ = [
    "RAGEvaluator",
    "ExtendedRAGEvaluator",
    "EvaluationReport",
    "SCORING_PROFILES",
    "rank_strategies",
    "MetricPreset",
    "create_metrics",
]
