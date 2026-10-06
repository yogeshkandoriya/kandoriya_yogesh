"""Text normalisation for extracted PDF pages.

PDF extraction produces a lot of layout noise: repeated running headers,
page numbers, hyphenated line breaks, and ragged whitespace. This module
strips that so downstream chunking and retrieval see clean prose.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

# A short line made only of digits/roman numerals is almost always a page number.
_PAGE_NUMBER = re.compile(r"^\s*(page\s+)?\d{1,4}\s*(of\s+\d{1,4}\s*)?$", re.I)
_ROMAN = re.compile(r"^\s*(page\s+)?[ivxlcdm]{1,7}\s*$", re.I)
# "Chapter 3", "UNIT - 2", "1.2 Introduction"
_NUMBERED_HEADING = re.compile(r"^\s*((\d+\.?\d*)|([a-z]))\s*[\.\)]?\s+[A-Z].{0,90}$")
# Short uppercase-ish lines are heading-like.
_SHORT_UPPER = re.compile(r"^\s*[A-Z][A-Z0-9\s\-\&\,\.\:\/]{3,70}$")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines without losing paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)      # re-join words split across lines
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def is_page_number(line: str) -> bool:
    """True if the line looks like a footer/header page marker."""
    return bool(_PAGE_NUMBER.match(line) or _ROMAN.match(line))


def looks_like_heading(line: str) -> bool:
    """Heuristic heading detector used to label chunks with a section name."""
    stripped = line.strip()
    if not (3 <= len(stripped) <= 90):
        return False
    if is_page_number(stripped) or stripped.endswith((".", ",", ";")):
        return False
    if _SHORT_UPPER.match(stripped):
        return True
    if _NUMBERED_HEADING.match(stripped):
        return True
    # Title Case line with no terminal punctuation.
    words = stripped.split()
    if 1 < len(words) <= 12 and not stripped.endswith((".", ",", ";")):
        capitals = sum(1 for w in words if w[:1].isupper())
        if capitals / len(words) >= 0.6:
            return True
    return False


def detect_running_headers(pages: Iterable[str], min_ratio: float = 0.4) -> set[str]:
    """Find short lines that repeat across many pages — running headers/footers.

    Only lines appearing on at least ``min_ratio`` of pages are removed, which
    keeps legitimate repeated content (e.g. a table column) intact.
    """
    page_list = [p for p in pages if p.strip()]
    if len(page_list) < 3:
        return set()

    first_lines: list[str] = []
    last_lines: list[str] = []
    for page in page_list:
        lines = [ln.strip() for ln in page.split("\n") if ln.strip()]
        if not lines:
            continue
        first_lines.append(lines[0])
        last_lines.append(lines[-1])
        # A header can also sit on line 2 when line 1 is a title.
        if len(lines) > 1 and is_page_number(lines[0]):
            first_lines.append(lines[1])
        if len(lines) > 1 and is_page_number(lines[-1]):
            last_lines.append(lines[-2])

    total = len(page_list)
    threshold = max(3, int(total * min_ratio))
    repeated = set()
    for line in Counter(first_lines + last_lines).most_common():
        text, count = line
        if count < threshold:
            break
        if len(text) <= 90 and text:
            repeated.add(text)
    return repeated


def clean_page(text: str, running_headers: Optional[set[str]] = None) -> str:
    """Clean a single page, dropping page numbers and known running headers."""
    running_headers = running_headers or set()
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if is_page_number(stripped) or stripped in running_headers:
            continue
        kept.append(stripped)
    return normalize_whitespace("\n".join(kept))


def clean_pages(pages: list[str]) -> tuple[list[str], set[str]]:
    """Clean every page, auto-detecting running headers first.

    Returns the cleaned pages and the set of headers that were stripped.
    """
    headers = detect_running_headers(pages)
    return [clean_page(p, headers) for p in pages], headers


def dedupe_sentences(text: str) -> str:
    """Drop immediately repeated sentences, a common OCR/extraction artefact."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        key = re.sub(r"\W+", "", part.lower())
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(part)
    return " ".join(out).strip()
