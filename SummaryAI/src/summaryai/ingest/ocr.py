"""Optional OCR support for scanned / image-only PDF pages.

Tesseract is not a hard dependency. If the binary or traineddata is missing,
:func:`is_ocr_available` returns ``False`` and the pipeline simply skips OCR
for that page instead of failing.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional

#: Repo-bundled traineddata (see tessdata/eng.traineddata) used if present.
_BUNDLED_TESSDATA = Path(__file__).resolve().parents[3] / "tessdata"


@lru_cache(maxsize=1)
def is_ocr_available() -> bool:
    """Whether pytesseract plus a Tesseract binary can actually run."""
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False

    if os.environ.get("TESSDATA_PREFIX") and _BUNDLED_TESSDATA.exists():
        return True
    return shutil.which("tesseract") is not None


def configure_tessdata() -> None:
    """Point Tesseract at the repo's tessdata folder when it exists."""
    if _BUNDLED_TESSDATA.exists():
        os.environ.setdefault("TESSDATA_PREFIX", str(_BUNDLED_TESSDATA))


def ocr_image_object(image, dpi: int = 200) -> str:
    """Run Tesseract on a PIL image. Returns ``""`` on any failure."""
    if not is_ocr_available():
        return ""
    try:
        import pytesseract

        configure_tessdata()
        # PSM 6 = assume a uniform block of text, which suits page scans.
        config = "--oem 3 --psm 6"
        return (pytesseract.image_to_string(image, config=config) or "").strip()
    except Exception:
        return ""


def page_needs_ocr(native_text: str, threshold: int = 120) -> bool:
    """Heuristic: pages with almost no embedded text are scanned images."""
    return len(native_text.strip()) < threshold


def merge_native_and_ocr(native: str, ocr: str, threshold: float = 0.85) -> str:
    """Combine embedded text with OCR output, avoiding duplicated content.

    If the OCR text is mostly already present in the native text we keep the
    native version (it has correct spacing and reading order).
    """
    native = (native or "").strip()
    ocr = (ocr or "").strip()
    if not ocr:
        return native
    if not native:
        return ocr

    native_words = set(native.lower().split())
    ocr_words = ocr.lower().split()
    if not ocr_words:
        return native

    overlap = sum(1 for w in ocr_words if w in native_words) / len(ocr_words)
    if overlap >= threshold:
        return native

    return f"{native}\n\n{ocr}".strip()
