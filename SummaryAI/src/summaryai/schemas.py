"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Chunk:
    """A page-aware slice of a document with retrieval metadata."""

    text: str
    page: int                      # 1-based page number
    index: int = 0                 # position within the document
    section: str = ""              # detected heading, if any
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "page": self.page,
            "index": self.index,
            "section": self.section,
            "semantic_score": round(self.semantic_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "score": round(self.score, 4),
        }


@dataclass
class Citation:
    """A verifiable pointer back to a source page."""

    page: int
    score: float
    excerpt: str
    doc_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "score": round(self.score, 4),
            "excerpt": self.excerpt,
            "doc_name": self.doc_name,
        }


@dataclass
class Document:
    """A fully ingested document ready for indexing and summarisation."""

    name: str
    full_text: str
    pages: list[str]
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0
    file_size: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    ocr_pages: list[int] = field(default_factory=list)
    added_at: str = ""

    @classmethod
    def from_pages(
        cls,
        name: str,
        pages: list[str],
        *,
        meta: Optional[dict] = None,
        file_size: int = 0,
        ocr_pages: Optional[list[int]] = None,
    ) -> "Document":
        full_text = "\n\n".join(pages)
        return cls(
            name=name,
            full_text=full_text,
            pages=pages,
            page_count=len(pages),
            word_count=len(full_text.split()),
            char_count=len(full_text),
            file_size=file_size,
            meta=meta or {},
            ocr_pages=ocr_pages or [],
        )

    def stats(self) -> dict[str, Any]:
        return {
            "Pages": self.page_count,
            "Words": f"{self.word_count:,}",
            "Characters": f"{self.char_count:,}",
            "Size": f"{self.file_size / 1024:.1f} KB" if self.file_size else "n/a",
            "OCR pages": len(self.ocr_pages),
        }


@dataclass
class SearchResult:
    """Outcome of a retrieval query."""

    query: str
    chunks: list[Chunk]
    query_type: str = "general"
    refused: bool = False
    reason: str = ""

    @property
    def context(self) -> str:
        """Chunks formatted for inclusion in a prompt, with page markers."""
        return "\n\n".join(
            f"[Page {c.page}]{' — ' + c.section if c.section else ''}\n{c.text}" for c in self.chunks
        )

    @property
    def pages(self) -> list[int]:
        seen: list[int] = []
        for chunk in self.chunks:
            if chunk.page not in seen:
                seen.append(chunk.page)
        return sorted(seen)

    def citations(self, limit: int = 3) -> list[Citation]:
        return [
            Citation(page=c.page, score=c.score, excerpt=c.text[:450], doc_name="")
            for c in self.chunks[:limit]
        ]
