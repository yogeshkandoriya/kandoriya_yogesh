"""FAISS vector index wrapper (cosine similarity via inner product)."""

from __future__ import annotations

from typing import Optional, Sequence

import faiss
import numpy as np

from ..schemas import Chunk
from .embeddings import Embedder, l2_normalize


class VectorStore:
    """Builds and queries a FAISS ``IndexFlatIP`` over normalised embeddings."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.index: Optional["faiss.Index"] = None
        self.chunks: list[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(self, chunks: Sequence[Chunk], *, batch_size: int = 32) -> "VectorStore":
        """Encode chunks and populate the index. Safe to call repeatedly."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.index = None
            self.embeddings = None
            return self

        vectors = l2_normalize(self.embedder.encode([c.text for c in self.chunks]))
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        self.index = index
        self.embeddings = vectors
        return self

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 4) -> list[tuple[int, float]]:
        """Return ``(chunk_index, cosine_score)`` pairs, best first."""
        if self.index is None or not self.chunks or not query.strip():
            return []

        query_vec = l2_normalize(self.embedder.encode([query]))
        limit = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, limit)

        results: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def semantic_scores(self, query: str) -> dict[int, float]:
        """Score for *every* chunk. Used by hybrid retrieval, which needs the
        full distribution to normalise lexical scores consistently."""
        if self.index is None or not self.chunks or not query.strip():
            return {}
        query_vec = l2_normalize(self.embedder.encode([query]))
        scores, indices = self.index.search(query_vec, len(self.chunks))
        return {int(idx): float(score) for score, idx in zip(scores[0], indices[0]) if idx != -1}

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return len(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)
