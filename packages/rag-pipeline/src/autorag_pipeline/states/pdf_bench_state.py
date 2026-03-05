"""PDF parser benchmark pipeline state definition."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class PDFBenchState(TypedDict, total=False):
    """State for the PDF parser benchmark graph."""

    # --- Input ---
    specs: list[dict]  # serialised BenchSpec list (backend × pdf combinations)
    results_dir: str
    skip_parse: bool  # True → 기존 output.md에 정규화만 재적용

    # --- Output ---
    parse_results: list[dict]
    normalize_results: list[dict]
    eval_results: list[dict]
    summary: str

    # --- Error tracking ---
    errors: Annotated[list[str], operator.add]
