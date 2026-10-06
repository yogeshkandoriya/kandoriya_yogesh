"""Summary AI - a document intelligence assistant.

A clean, modular reimplementation of the summarisation pipeline:

    PDF -> extraction/OCR -> cleaning -> chunking -> embeddings -> FAISS
        -> hybrid retrieval -> OpenRouter LLM -> grounded summaries & Q&A

Public entry point is :class:`DocumentAssistant`, which owns a document
library and exposes high-level operations the UI can call directly.
"""

from .config import DEFAULT_MODEL, FALLBACK_MODELS, FREE_MODELS, Settings, get_settings
from .schemas import Chunk, Citation, Document, SearchResult

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_MODEL",
    "FALLBACK_MODELS",
    "FREE_MODELS",
    "Settings",
    "get_settings",
    "Chunk",
    "Citation",
    "Document",
    "SearchResult",
    "DocumentAssistant",
    "__version__",
]


def __getattr__(name: str):
    """Import DocumentAssistant lazily to keep ``import summaryai`` cheap."""
    if name == "DocumentAssistant":
        from .pipeline import DocumentAssistant

        return DocumentAssistant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
