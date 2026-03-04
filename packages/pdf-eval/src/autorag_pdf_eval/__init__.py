"""autorag-pdf-eval: PDF parsing benchmark — OmniDoc metrics."""

from autorag_pdf_eval.evaluator import (
    BenchResult,
    PageScore,
    evaluate_document,
    evaluate_page,
)
from autorag_pdf_eval.omnidoc_metrics import (
    OmniDocScore,
    compute_bleu,
    compute_meteor,
    compute_omnidoc_scores,
    compute_teds_html,
    compute_text_ned,
)
from autorag_pdf_eval.spec import Backend, BenchSpec, GT_MAP, PRESETS, get_preset

__all__ = [
    "BenchResult",
    "PageScore",
    "Backend",
    "BenchSpec",
    "GT_MAP",
    "PRESETS",
    "OmniDocScore",
    "compute_text_ned",
    "compute_bleu",
    "compute_meteor",
    "compute_teds_html",
    "compute_omnidoc_scores",
    "evaluate_document",
    "evaluate_page",
    "get_preset",
]
