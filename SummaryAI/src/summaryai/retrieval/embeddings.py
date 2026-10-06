"""Dense embedding helpers backed by sentence-transformers.

The model is loaded lazily and cached, because it is the single most expensive
object in the pipeline. Tests can swap in a deterministic hash embedder.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

_DIM = 384  # all-MiniLM-L6-v2 output size


class Embedder(Protocol):
    """Anything that turns text into fixed-size vectors."""

    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        ...


class SentenceTransformerEmbedder:
    """Real embeddings from ``sentence-transformers/all-MiniLM-L6-v2``."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.dimension = _DIM
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbedder:
    """Deterministic, dependency-free bag-of-words hashing embedder.

    Used by the test-suite: it preserves lexical similarity (words shared
    between a query and a chunk score higher) without downloading a model.
    """

    def __init__(self, dimension: int = _DIM) -> None:
        self.dimension = dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                # Two independent buckets give both a slot and a sign.
                slot = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, slot] += sign
        return matrix

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation so dot product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# Process-wide cache so Streamlit reruns do not reload the model.
_EMBEDDER_CACHE: dict[str, Embedder] = {}


def get_embedder(model_name: str = "all-MiniLM-L6-v2", *, lightweight: bool = False) -> Embedder:
    """Return a cached embedder.

    ``lightweight=True`` returns the :class:`HashEmbedder` (no model download).
    """
    key = f"{model_name}:{'light' if lightweight else 'full'}"
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = HashEmbedder() if lightweight else SentenceTransformerEmbedder(model_name)
    return _EMBEDDER_CACHE[key]
