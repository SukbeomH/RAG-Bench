"""autorag-pdf-eval: PDF parsing benchmark — NED/TEDS metrics."""

from autorag_pdf_eval.evaluator import (
    BenchResult,
    PageScore,
    compute_text_ned,
    compute_table_teds,
    evaluate_document,
    evaluate_page,
)
from autorag_pdf_eval.spec import Backend, BenchSpec, GT_MAP, PRESETS, get_preset

__all__ = [
    "BenchResult",
    "PageScore",
    "Backend",
    "BenchSpec",
    "GT_MAP",
    "PRESETS",
    "compute_text_ned",
    "compute_table_teds",
    "evaluate_document",
    "evaluate_page",
    "get_preset",
]
