"""Parser registry — factory pattern for PDF backends."""

from __future__ import annotations

from typing import Any, Callable

from autorag_parsers._protocol import PDFParser

_REGISTRY: dict[str, Callable[..., PDFParser]] = {}


def register(name: str) -> Callable:
    """Decorator to register a parser factory under a name."""

    def decorator(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_parser(name: str, **kwargs: Any) -> PDFParser:
    """Instantiate a parser by registered name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown parser '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def available_backends() -> list[str]:
    """Return all registered parser names."""
    return list(_REGISTRY.keys())
