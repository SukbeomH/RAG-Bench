"""Core protocol and data classes for PDF parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class PageResult:
    """Single page conversion result."""

    page_num: int
    markdown: str
    backend: str
    bbox_data: list[dict] | None = None  # [{x0,y0,x1,y1,text}] for citation
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversionResult:
    """Full document conversion result."""

    pdf_path: str
    pages: list[PageResult]
    total_time_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def full_markdown(self) -> str:
        return "\n\n".join(p.markdown for p in self.pages)


@runtime_checkable
class PDFParser(Protocol):
    """Protocol that all PDF parser backends must implement."""

    @property
    def name(self) -> str: ...

    def convert(self, pdf_path: str, **kwargs: object) -> ConversionResult: ...
