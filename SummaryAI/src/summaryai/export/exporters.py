"""Export generated content to Markdown, Word, PowerPoint and PDF.

Each function returns raw ``bytes`` so the UI can hand them straight to a
download button without touching the filesystem.
"""

from __future__ import annotations

import io
import re
from typing import Iterable, Optional

from ..schemas import Citation

#: Rough characters-per-slide budget for the PowerPoint exporter.
SLIDE_CHAR_BUDGET = 900


def safe_filename(name: str, fallback: str = "summary") -> str:
    """Strip characters that are illegal in filenames on Windows/POSIX."""
    cleaned = re.sub(r"[^\w\s\-\.]", "", (name or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_.")
    return cleaned[:80] or fallback


# --------------------------------------------------------------------------- #
# Markdown / plain text
# --------------------------------------------------------------------------- #
def to_markdown(
    title: str,
    content: str,
    citations: Optional[Iterable[Citation]] = None,
    stats: Optional[dict] = None,
) -> str:
    """Render a full report as markdown, including sources and statistics."""
    parts = [f"# {title}", "", content, ""]

    citations = list(citations or [])
    if citations:
        parts += ["## Sources", ""]
        for cite in citations:
            doc = f"{cite.doc_name}, " if cite.doc_name else ""
            parts.append(f"- **{doc}Page {cite.page}** (relevance {cite.score:.2f})")
            if cite.excerpt:
                parts.append(f"  > {cite.excerpt.strip()}")
        parts.append("")

    if stats:
        parts += ["## Document Statistics", ""]
        parts += [f"- **{key}**: {value}" for key, value in stats.items()]
        parts.append("")

    return "\n".join(parts)


def to_plain_text(title: str, content: str) -> str:
    """Strip the most common markdown syntax for a clean .txt download."""
    text = re.sub(r"^#{1,6}\s*", "", content or "", flags=re.M)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)
    text = re.sub(r"^[-•]\s*", "- ", text, flags=re.M)
    return f"{title}\n{'=' * len(title)}\n\n{text.strip()}\n"


# --------------------------------------------------------------------------- #
# Word
# --------------------------------------------------------------------------- #
def to_docx(title: str, content: str, citations: Optional[Iterable[Citation]] = None) -> bytes:
    """Build a .docx with headings, bullet lists and a sources section."""
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("python-docx is required for Word export: pip install python-docx") from exc

    document = docx.Document()
    document.add_heading(title, level=0)

    for raw_line in (content or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            level = min(6, len(line) - len(line.lstrip("#")))
            document.add_heading(line.lstrip("#").strip(), level=level)
        elif re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line):
            document.add_paragraph(
                re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line), style="List Bullet"
            )
        else:
            document.add_paragraph(line)

    citations = list(citations or [])
    if citations:
        document.add_heading("Sources", level=1)
        for cite in citations:
            label = f"{cite.doc_name}, Page {cite.page}" if cite.doc_name else f"Page {cite.page}"
            document.add_paragraph(label, style="List Bullet")

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# PowerPoint
# --------------------------------------------------------------------------- #
def _split_into_slides(content: str) -> list[tuple[str, list[str]]]:
    """Group content into ``(heading, bullet_lines)`` slides."""
    slides: list[tuple[str, list[str]]] = []
    heading = "Overview"
    bullets: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal bullets, size
        if bullets:
            slides.append((heading, bullets))
        bullets, size = [], 0

    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip() or "Overview"
            continue
        if re.match(r"^(?:[-*•]|\d+[.)])\s+", line):
            bullets.append(re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line))
        else:
            bullets.append(line)
        size += len(line)
        if size >= SLIDE_CHAR_BUDGET:
            flush()
    flush()
    return slides or [("Overview", ["(No content)"])]


def to_pptx(title: str, content: str, citations: Optional[Iterable[Citation]] = None) -> bytes:
    """Build a .pptx deck: one title slide plus a slide per section."""
    try:
        from pptx import Presentation
        from pptx.util import Pt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("python-pptx is required for PowerPoint export") from exc

    presentation = Presentation()

    title_slide = presentation.slides.add_slide(presentation.slide_layouts[0])
    title_slide.shapes.title.text = title
    title_slide.placeholders[1].text = "Generated by Summary AI"

    for heading, bullets in _split_into_slides(content):
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = heading
        frame = slide.placeholders[1].text_frame
        frame.word_wrap = True
        frame.text = bullets[0]
        for item in bullets[1:]:
            frame.add_paragraph().text = item
        for paragraph in frame.paragraphs:
            paragraph.font.size = Pt(18)

    citations = list(citations or [])
    if citations:
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "Sources"
        frame = slide.placeholders[1].text_frame
        frame.text = f"Page {citations[0].page}"
        for cite in citations[1:]:
            frame.add_paragraph().text = f"Page {cite.page}"

    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def to_pdf(title: str, content: str, citations: Optional[Iterable[Citation]] = None) -> bytes:
    """Render to PDF using PyMuPDF, with automatic pagination."""
    import pymupdf

    document = pymupdf.open()
    state = {"page": document.new_page(), "y": 56.0}
    BOTTOM, MARGIN, WIDTH = 780.0, 50.0, 500.0

    def write(text: str, size: float, gap: float = 6.0) -> None:
        """Word-wrap ``text`` to the page width and advance the cursor."""
        font = "hebo" if size >= 12 else "helv"
        line = ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if pymupdf.get_text_length(trial, fontname=font, fontsize=size) <= WIDTH:
                line = trial
                continue
            if line:  # flush the full line before starting a new one
                if state["y"] > BOTTOM:
                    state["page"] = document.new_page()
                    state["y"] = 56.0
                state["page"].insert_text((MARGIN, state["y"]), line, fontsize=size, fontname=font)
                state["y"] += size + 4
            line = word
        if line:
            if state["y"] > BOTTOM:
                state["page"] = document.new_page()
                state["y"] = 56.0
            state["page"].insert_text((MARGIN, state["y"]), line, fontsize=size, fontname=font)
            state["y"] += size + 4
        state["y"] += gap

    write(title, 15, 12)
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            state["y"] += 6
            continue
        if line.startswith("#"):
            write(line.lstrip("# ").strip(), 12, 8)
            continue
        bullet = bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line))
        if bullet:
            write(f"•  {re.sub(r'^(?:[-*•]|\d+[.)])\s+', '', line)}", 10)
        else:
            write(line, 10)

    citations = list(citations or [])
    if citations:
        state["y"] += 10
        write("Sources", 12, 8)
        for cite in citations:
            label = f"{cite.doc_name}, Page {cite.page}" if cite.doc_name else f"Page {cite.page}"
            write(f"•  {label}", 9)

    data = document.tobytes()
    document.close()
    return data
