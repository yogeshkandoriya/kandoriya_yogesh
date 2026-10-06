import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Embeds page-aware text chunks and retrieves the most relevant ones."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        # Each chunk: {"text": str, "page": int, "position": int}
        self.chunks: list[dict] = []
        self.chunk_words = 180
        self.overlap_words = 30

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def create_chunks(self, pages: list[tuple[int, str]]) -> list[dict]:
        """Split each page into overlapping word windows, keeping page numbers."""
        chunks: list[dict] = []
        step = max(1, self.chunk_words - self.overlap_words)
        for page_num, page_text in pages:
            words = page_text.split()
            if not words:
                continue
            if len(words) <= self.chunk_words:
                chunks.append({"text": page_text.strip(), "page": page_num,
                               "position": len(chunks)})
                continue
            for i in range(0, len(words), step):
                window = words[i:i + self.chunk_words]
                if len(window) < self.overlap_words and chunks and chunks[-1]["page"] == page_num:
                    break  # tail already covered by the previous window
                chunks.append({
                    "text": " ".join(window),
                    "page": page_num,
                    "position": len(chunks),
                })
        return chunks

    def build_index(self, pages: list[tuple[int, str]]) -> int:
        """Embed chunks and build the FAISS index. Returns the chunk count."""
        self.chunks = self.create_chunks(pages)
        if not self.chunks:
            self.index = None
            return 0
        embeddings = self.model.encode(
            [c["text"] for c in self.chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity via inner product
            show_progress_bar=False,
        )
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        return len(self.chunks)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """Return the top_k most similar chunks (copies with a 'score' field)."""
        if self.index is None or not self.chunks:
            return []
        top_k = min(top_k, len(self.chunks))
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        seen_positions = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks) or idx in seen_positions:
                continue
            seen_positions.add(idx)
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results
