"""PDF text extraction with metadata, OCR fallback, and page cleaning."""

from __future__ import annotations

import io
from typing import Any, BinaryIO, Union

from pathlib import Path

import pymupdf

from ..schemas import Document
from . import ocr as ocr_mod
from .cleaner import clean_pages

FileLike = Union[bytes, bytearray, BinaryIO, str, Path]


def _open(file_obj: FileLike) -> "pymupdf.Document":
    """PyMuPDF accepts bytes, paths and streams interchangeably."""
    if isinstance(file_obj, (bytes, bytearray)):
        return pymupdf.open(stream=bytes(file_obj), filetype="pdf")
    return pymupdf.open(file_obj)


def extract_metadata(file_obj: FileLike) -> dict[str, Any]:
    """Pull title/subject/author from the PDF info dictionary, never raising."""
    try:
        with _open(file_obj) as doc:
            return {k: (v or "").strip() for k, v in (doc.metadata or {}).items() if v}
    except Exception:
        return {}


def extract_pages(
    file_obj: FileLike,
    *,
    enable_ocr: bool = True,
    ocr_dpi: int = 200,
) -> tuple[list[str], list[int], dict[str, Any]]:
    """Extract per-page text, OCR-ing pages that appear to be scanned images.

    Returns ``(raw_pages, ocr_page_numbers, metadata)``.
    """
    raw_pages: list[str] = []
    ocr_pages: list[int] = []
    metadata: dict[str, Any] = {}

    doc = _open(file_obj)
    try:
        metadata = {k: (v or "").strip() for k, v in (doc.metadata or {}).items() if v}
        use_ocr = enable_ocr and ocr_mod.is_ocr_available()

        for index in range(doc.page_count):
            page = doc.load_page(index)
            native = page.get_text("text") or ""

            if use_ocr and ocr_mod.page_needs_ocr(native):
                ocr_text = ""
                for image_info in page.get_images(full=True):
                    try:
                        extracted = doc.extract_image(image_info[0])
                        pil_image = _to_pil(extracted["image"])
                        if pil_image is not None:
                            ocr_text = ocr_mod.ocr_image_object(pil_image, dpi=ocr_dpi)
                            if ocr_text:
                                break
                    except Exception:
                        continue
                if ocr_text:
                    native = ocr_mod.merge_native_and_ocr(native, ocr_text)
                    ocr_pages.append(index + 1)

            raw_pages.append(native)
    finally:
        doc.close()

    return raw_pages, ocr_pages, metadata


def _to_pil(image_bytes: bytes):
    """Convert raw image bytes to a PIL image, or ``None`` if unsupported."""
    try:
        import io as _io

        from PIL import Image

        return Image.open(_io.BytesIO(image_bytes))
    except Exception:
        return None


def load_document(
    file_obj: FileLike,
    name: str,
    *,
    enable_ocr: bool = True,
    file_size: int = 0,
) -> Document:
    """Full ingestion pipeline: extract -> OCR -> clean -> :class:`Document`."""
    raw_pages, ocr_pages, metadata = extract_pages(file_obj, enable_ocr=enable_ocr)
    cleaned, _headers = clean_pages(raw_pages)

    # Keep pages positionally aligned so citations stay accurate even when a
    # page cleans down to an empty string.
    doc = Document.from_pages(
        name=name,
        pages=cleaned,
        meta=metadata,
        file_size=file_size,
        ocr_pages=ocr_pages,
    )
    return doc


def load_document_from_bytes(data: bytes, name: str, **kwargs) -> Document:
    """Convenience wrapper for Streamlit uploads."""
    return load_document(io.BytesIO(data), name, file_size=len(data), **kwargs)
