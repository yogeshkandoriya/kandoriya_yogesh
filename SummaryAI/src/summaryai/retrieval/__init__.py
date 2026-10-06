"""Retrieval: chunking, embeddings, FAISS index, and hybrid ranking."""

from .chunker import chunk_document, chunk_stats, estimate_tokens, find_page_text, fit_to_budget, locate_phrase
from .embeddings import HashEmbedder, SentenceTransformerEmbedder, get_embedder, l2_normalize
from .hybrid import analyze_query, hybrid_search, lexical_score, pages_mentioning
from .vector_store import VectorStore

__all__ = [
    "chunk_document",
    "chunk_stats",
    "estimate_tokens",
    "find_page_text",
    "fit_to_budget",
    "locate_phrase",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
    "l2_normalize",
    "analyze_query",
    "hybrid_search",
    "lexical_score",
    "pages_mentioning",
    "VectorStore",
]
