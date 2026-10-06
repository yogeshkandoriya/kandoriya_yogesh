import os

import pymupdf


def extract_text(file_path: str) -> list[tuple[int, str]]:
    """
    Extracts text from a PDF page by page.

    Returns a list of ``(page_number, text)`` tuples (1-indexed pages).
    """
    pages: list[tuple[int, str]] = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Uploaded file not found: {file_path}")

    doc = pymupdf.open(file_path)
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError("The PDF is password-protected and cannot be read.")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text().strip()
            if text:
                pages.append((page_num + 1, text))
    finally:
        doc.close()
    return pages


def get_text_from_pdf(file_path: str) -> str:
    """Returns the full text of the PDF, with page markers."""
    return "\n\n".join(f"[Page {p}]\n{t}" for p, t in extract_text(file_path))
