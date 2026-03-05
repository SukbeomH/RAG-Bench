"""autorag-parsers: PDF → Markdown conversion with multiple backends."""

from autorag_parsers._protocol import ConversionResult, PageResult, PDFParser
from autorag_parsers.chunking import ChunkConfig, chunk_document, chunk_page
from autorag_parsers.provenance import ChunkProvenance, CitedAnswer, Citation
from autorag_parsers.registry import available_backends, get_parser, register

# Auto-register built-in backends on import
import autorag_parsers.pymupdf as _pymupdf  # noqa: F401

# Lazy-loaded backends (import triggers registration)
_LAZY_BACKENDS = {
    "docling": "autorag_parsers.docling",
    "openai": "autorag_parsers.openai_vision",
    "openai-4.1": "autorag_parsers.openai_vision",
    "upstage": "autorag_parsers.upstage",
    "upstage-enhanced": "autorag_parsers.upstage",
    "paddleocr-vl": "autorag_parsers.paddleocr_vl",
    "deepseek-ocr2": "autorag_parsers.openai_compat",
}

_original_get_parser = get_parser


def _lazy_get_parser(name: str, **kwargs):  # type: ignore[no-untyped-def]
    """Try direct registry first; if not found, lazy-import the module."""
    try:
        return _original_get_parser(name, **kwargs)
    except KeyError:
        if name in _LAZY_BACKENDS:
            import importlib

            importlib.import_module(_LAZY_BACKENDS[name])
            return _original_get_parser(name, **kwargs)
        raise


# Monkey-patch for lazy loading
import autorag_parsers.registry as _reg

_reg.get_parser = _lazy_get_parser
get_parser = _lazy_get_parser

__all__ = [
    "ChunkConfig",
    "ChunkProvenance",
    "CitedAnswer",
    "Citation",
    "ConversionResult",
    "PageResult",
    "PDFParser",
    "available_backends",
    "chunk_document",
    "chunk_page",
    "get_parser",
    "register",
]
