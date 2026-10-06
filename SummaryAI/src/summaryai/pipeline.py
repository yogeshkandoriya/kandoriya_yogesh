"""The :class:`DocumentAssistant` orchestrator.

Owns a library of loaded documents and their vector indexes, and exposes the
high-level operations the UI needs. All UI code talks to this class, never to
the internals, which keeps the Streamlit layer thin and the logic testable.
"""

from __future__ import annotations

from typing import Optional

from .analysis import compare_documents, extract_keywords, find_term_pages
from .config import FALLBACK_MODELS, Settings, get_settings
from .generation import (
    AnswerResult,
    SummaryResult,
    answer_question,
    explain_topic,
    quality_report,
    summarize,
)
from .ingest import load_document
from .llm.base import LLMEngine
from .llm.openrouter import OpenRouterEngine
from .retrieval import (
    VectorStore,
    chunk_document,
    chunk_stats,
    get_embedder,
    hybrid_search,
)
from .schemas import Document


class DocumentAssistant:
    """Loads documents, indexes them, and answers questions about them."""

    def __init__(
        self,
        engine: Optional[LLMEngine] = None,
        settings: Optional[Settings] = None,
        *,
        lightweight_embeddings: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or self._default_engine()
        self.embedder = get_embedder(
            self.settings.embedding_model, lightweight=lightweight_embeddings
        )
        #: doc name -> Document
        self.documents: dict[str, Document] = {}
        #: doc name -> VectorStore
        self.indexes: dict[str, VectorStore] = {}

    def _default_engine(self) -> LLMEngine:
        if not self.settings.has_key():
            return OpenRouterEngine(api_key="", model=self.settings.model)
        return OpenRouterEngine(
            api_key=self.settings.api_key,
            model=self.settings.model,
            fallback_models=FALLBACK_MODELS,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            timeout=self.settings.request_timeout,
            max_retries=self.settings.max_retries,
            app_url=self.settings.app_url,
            app_title=self.settings.app_title,
        )

    # ------------------------------------------------------------------ #
    # Library management
    # ------------------------------------------------------------------ #
    def add_document(
        self,
        file_obj,
        name: str,
        *,
        enable_ocr: bool = True,
        file_size: int = 0,
    ) -> Document:
        """Ingest, chunk and index a PDF, replacing any existing entry."""
        document = load_document(file_obj, name, enable_ocr=enable_ocr, file_size=file_size)
        self.documents[name] = document

        chunks = chunk_document(
            document.pages,
            max_chars=self.settings.chunk_chars,
            overlap=self.settings.chunk_overlap,
        )
        self.indexes[name] = VectorStore(self.embedder).build(chunks)
        return document

    def remove_document(self, name: str) -> bool:
        """Drop a document and free its index."""
        self.documents.pop(name, None)
        return self.indexes.pop(name, None) is not None

    def clear(self) -> None:
        self.documents.clear()
        self.indexes.clear()

    @property
    def names(self) -> list[str]:
        return list(self.documents)

    def has_documents(self) -> bool:
        return bool(self.documents)

    def index_stats(self, name: str) -> dict:
        """Chunk statistics for the Developer/Audit panel."""
        store = self.indexes.get(name)
        if store is None:
            return {}
        stats = chunk_stats(store.chunks)
        stats["Pages indexed"] = len({c.page for c in store.chunks})
        return stats

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #
    def summarize(
        self,
        name: str,
        *,
        length: str = "Medium",
        fmt: str = "Bullet Points",
        simple_language: bool = False,
    ) -> SummaryResult:
        """Summarise a loaded document."""
        document = self._require(name)
        context_tokens = max(8000, self.settings.model_context() // 2)
        result = summarize(
            document,
            self.engine,
            length=length,
            fmt=fmt,
            simple_language=simple_language,
            max_context_tokens=context_tokens,
        )
        result.coverage = quality_report(result, document)
        return result

    def ask(
        self,
        name: str,
        question: str,
        *,
        top_k: Optional[int] = None,
    ) -> AnswerResult:
        """Answer a question about a loaded document, with citations."""
        self._require(name)
        return answer_question(
            question,
            self.documents[name],
            self.indexes[name],
            self.engine,
            top_k=top_k or self.settings.top_k,
        )

    def explain(self, name: str, topic: str) -> AnswerResult:
        """Explain a topic, refusing if the document does not cover it."""
        self._require(name)
        return explain_topic(topic, self.documents[name], self.indexes[name], self.engine)

    def search(self, name: str, query: str, top_k: int = 5):
        """Raw hybrid retrieval, exposed for the debug panel."""
        self._require(name)
        return hybrid_search(query, self.indexes[name], top_k=top_k)

    def keywords(self, name: str, top_n: int = 12) -> list:
        """TF-IDF keywords, computed entirely offline."""
        return extract_keywords(self._require(name), top_n)

    def find(self, name: str, term: str, top_n: int = 5) -> list[dict]:
        """Locate a term within a document and return page-level excerpts."""
        return find_term_pages(self._require(name), term, top_n)

    def compare(self, name_a: str, name_b: str) -> str:
        """Compare two loaded documents."""
        doc_a, doc_b = self._require(name_a), self._require(name_b)
        return compare_documents(doc_a, doc_b, self.engine)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _require(self, name: str) -> Document:
        if name not in self.documents:
            raise KeyError(f"Document '{name}' is not loaded.")
        return self.documents[name]
