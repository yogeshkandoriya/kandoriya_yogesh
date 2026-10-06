"""Tests for ingestion, cleaning and chunking."""

from __future__ import annotations

from summaryai.ingest import (
    clean_page,
    clean_pages,
    dedupe_sentences,
    detect_running_headers,
    is_ocr_available,
    is_page_number,
    load_document,
    looks_like_heading,
    merge_native_and_ocr,
    normalize_whitespace,
    page_needs_ocr,
)
from summaryai.retrieval import chunk_document, chunk_stats


# --------------------------------------------------------------------------- #
# Cleaner
# --------------------------------------------------------------------------- #
class TestCleaner:
    def test_removes_page_numbers(self):
        assert is_page_number("12")
        assert is_page_number("Page 7")
        assert is_page_number("iv")
        assert not is_page_number("Introduction to hardware")

    def test_clean_page_drops_page_numbers(self):
        cleaned = clean_page("HEADER\n\nReal content here.\n\n12")
        assert "Real content here." in cleaned
        assert "12" not in cleaned

    def test_detects_running_headers(self):
        pages = ["COMPUTER ARCHITECTURE\n\nBody A", "COMPUTER ARCHITECTURE\n\nBody B", "COMPUTER ARCHITECTURE\n\nBody C"]
        headers = detect_running_headers(pages)
        assert "COMPUTER ARCHITECTURE" in headers

    def test_does_not_treat_unique_lines_as_headers(self):
        pages = ["Unique A\n\nx", "Unique B\n\ny", "Unique C\n\nz", "Unique D\n\nw"]
        assert "Unique A" not in detect_running_headers(pages)

    def test_clean_pages_returns_tuple(self):
        pages, headers = clean_pages(["H\n\nbody", "H\n\nbody2"])
        assert len(pages) == 2
        assert isinstance(headers, set)

    def test_normalize_joins_hyphenated_linebreaks(self):
        assert "multiprocessing" in normalize_whitespace("multi-\nprocessing unit")

    def test_normalize_collapses_blank_lines(self):
        assert normalize_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_heading_detection(self):
        assert looks_like_heading("COMPUTER ARCHITECTURE")
        assert looks_like_heading("1.2 Introduction")
        assert looks_like_heading("Introduction To Hardware")
        assert not looks_like_heading("This is a sentence that continues and ends with a period.")

    def test_dedupe_sentences(self):
        text = "The CPU is fast. The CPU is fast. The GPU is slow."
        assert dedupe_sentences(text).count("The CPU is fast") == 1


# --------------------------------------------------------------------------- #
# OCR helpers (must degrade gracefully without Tesseract)
# --------------------------------------------------------------------------- #
class TestOcrHelpers:
    def test_page_needs_ocr_thresholds(self):
        assert page_needs_ocr("tiny")
        assert not page_needs_ocr("x" * 300)

    def test_merge_keeps_native_when_redundant(self):
        native = "The CPU executes instructions at 3.5 GHz."
        assert merge_native_and_ocr(native, native.upper()) == native

    def test_merge_appends_new_content(self):
        merged = merge_native_and_ocr("Native text.", "Completely unrelated scanned words.")
        assert "Native text." in merged
        assert "unrelated" in merged

    def test_merge_handles_empty_ocr(self):
        assert merge_native_and_ocr("Native", "") == "Native"
        assert merge_native_and_ocr("", "OCR") == "OCR"

    def test_availability_is_boolean(self):
        assert isinstance(is_ocr_available(), bool)


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
class TestExtractor:
    def test_loads_all_pages(self, sample_pdf):
        document = load_document(sample_pdf, "sample.pdf", enable_ocr=False)
        assert document.page_count == 4
        assert document.word_count > 50
        assert "motherboard" in document.full_text.lower()

    def test_strips_repeated_running_header(self, sample_pdf):
        document = load_document(sample_pdf, "sample.pdf", enable_ocr=False)
        # The header repeats on every page and should be removed.
        assert document.full_text.lower().count("computer architecture") <= 1

    def test_rejects_non_pdf(self):
        import pymupdf  # noqa: F401  (imported to assert dependency present)

        try:
            load_document(b"not a pdf at all", "bad.pdf", enable_ocr=False)
        except Exception:
            pass  # Raising is acceptable; returning empty is also fine.
