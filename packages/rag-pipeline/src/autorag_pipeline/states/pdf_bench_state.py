"""PDF parser benchmark pipeline state definition."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class PDFBenchState(TypedDict, total=False):
    """State for the PDF parser benchmark graph."""

    # --- Input ---
    specs: list[dict]  # serialised BenchSpec list (backend × pdf combinations)
    results_dir: str

    # --- Output ---
    parse_results: list[dict]
    eval_results: list[dict]
    summary: str

    # --- Error tracking ---
    errors: Annotated[list[str], operator.add]
