"""Page-aware chunking.

Chunks are built per page (never spanning a page boundary) so every chunk can
be cited precisely. Each chunk also carries the nearest detected heading, which
improves both retrieval quality and the readability of citations.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from ..schemas import Chunk
from ..ingest.cleaner import looks_like_heading

# Prefer breaking on sentence boundaries for retrieval quality.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _slice_page(page_text: str, max_chars: int, overlap: int) -> list[str]:
    """Greedily pack sentences into ~``max_chars`` slices with ``overlap`` chars."""
    if len(page_text) <= max_chars:
        return [page_text]

    pieces: list[str] = []
    current = ""
    for sentence in _split_sentences(page_text):
        # A single oversized sentence is hard-split rather than dropped.
        if len(sentence) > max_chars:
            if current:
                pieces.append(current.strip())
                current = ""
            for i in range(0, len(sentence), max_chars):
                pieces.append(sentence[i : i + max_chars])
            continue

        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            pieces.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip() if tail else sentence

    if current.strip():
        pieces.append(current.strip())
    return [p for p in pieces if p]


def chunk_document(
    pages: list[str],
    *,
    max_chars: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Split a document into page-anchored chunks.

    Empty pages yield no chunks, but the surviving chunks keep their true
    page numbers so citations remain correct.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        overlap = max(0, min(overlap, max_chars // 3))

    chunks: list[Chunk] = []
    current_heading = ""

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text or not page_text.strip():
            continue

        # Track the most recent heading so later chunks can be labelled.
        heading = current_heading
        for line in page_text.split("\n"):
            if looks_like_heading(line):
                heading = line.strip()[:90]
                break
        current_heading = heading

        for piece in _slice_page(page_text, max_chars, overlap):
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    page=page_number,
                    index=len(chunks),
                    section=heading,
                )
            )
    return chunks


def chunk_stats(chunks: Iterable[Chunk]) -> dict[str, float]:
    """Summary statistics for the Developer/Audit panel."""
    lengths = [len(c.text) for c in chunks]
    if not lengths:
        return {"total_chunks": 0, "min_chars": 0, "max_chars": 0, "avg_chars": 0.0}
    return {
        "total_chunks": len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
    }


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used for prompt budgeting."""
    return max(1, len(text) // 4)


def fit_to_budget(
    chunks: Iterable[Chunk],
    max_tokens: int,
    *,
    chars_per_token: int = 4,
) -> list[Chunk]:
    """Trim a chunk list to fit a token budget, highest scoring first."""
    budget_chars = max_tokens * chars_per_token
    kept: list[Chunk] = []
    used = 0
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        cost = len(chunk.text)
        if used + cost > budget_chars:
            continue
        kept.append(chunk)
        used += cost
    return sorted(kept, key=lambda c: c.index)


def find_page_text(doc_pages: list[str], page: int) -> str:
    """Return the text of a 1-based page number, or ``""`` if out of range."""
    if 1 <= page <= len(doc_pages):
        return doc_pages[page - 1]
    return ""


def locate_phrase(doc_pages: list[str], phrase: str) -> Optional[int]:
    """Find the first page containing ``phrase`` (case-insensitive)."""
    needle = re.sub(r"\s+", " ", phrase.strip().lower())
    if not needle:
        return None
    for number, text in enumerate(doc_pages, start=1):
        if needle in re.sub(r"\s+", " ", text.lower()):
            return number
    return None
