"""Document ingestion: extraction, OCR fallback, and text cleaning."""

from .cleaner import (
    clean_page,
    clean_pages,
    dedupe_sentences,
    detect_running_headers,
    is_page_number,
    looks_like_heading,
    normalize_whitespace,
)
from .extractor import extract_metadata, extract_pages, load_document, load_document_from_bytes
from .ocr import is_ocr_available, merge_native_and_ocr, page_needs_ocr

__all__ = [
    "clean_page",
    "clean_pages",
    "dedupe_sentences",
    "detect_running_headers",
    "is_page_number",
    "looks_like_heading",
    "normalize_whitespace",
    "extract_metadata",
    "extract_pages",
    "load_document",
    "load_document_from_bytes",
    "is_ocr_available",
    "merge_native_and_ocr",
    "page_needs_ocr",
]
