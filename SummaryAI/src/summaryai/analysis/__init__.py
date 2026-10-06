"""Document analysis: keyword extraction, term location, and comparison."""

from .keywords import Keyword, compare_documents, extract_keywords, find_term_pages

__all__ = ["Keyword", "compare_documents", "extract_keywords", "find_term_pages"]
