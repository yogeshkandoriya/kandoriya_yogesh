"""Length-targeted document summarisation.

Long documents are summarised in a map-reduce fashion: the text is split into
token-budgeted sections, each section is summarised, and the partial summaries
are merged into one coherent result. This keeps quality high regardless of
document length while staying inside the model's context window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.base import LLMEngine
from ..schemas import Document
from .prompts import SUMMARY_SYSTEM, build_summary_prompt, strip_markdown_noise

#: Human label -> fraction of the document's own word count to target.
LENGTH_TARGETS: dict[str, float] = {
    "Short": 0.15,
    "Medium": 0.30,
    "Detailed": 0.50,
}

CHARS_PER_TOKEN = 4


@dataclass
class SummaryResult:
    """A generated summary plus the metadata needed to audit it."""

    text: str
    length_label: str
    fmt: str
    target_words: int
    actual_words: int
    model: str = ""
    sections_summarised: int = 0
    error: str = ""
    coverage: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())


def target_word_count(document: Document, length_label: str) -> int:
    """Words to aim for, scaled to the document and clamped to a sane band."""
    ratio = LENGTH_TARGETS.get(length_label, LENGTH_TARGETS["Medium"])
    return max(60, min(int(document.word_count * ratio), 1500))


def split_into_sections(text: str, max_tokens: int) -> list[str]:
    """Split text into ~``max_tokens`` sections on paragraph boundaries."""
    if not text.strip():
        return []
    budget = max(1, max_tokens) * CHARS_PER_TOKEN
    if len(text) <= budget:
        return [text]

    sections: list[str] = []
    current = ""
    for paragraph in re.split(r"\n{2,}", text):
        if not paragraph.strip():
            continue
        if len(paragraph) > budget:
            # A single oversized paragraph still has to be split.
            if current:
                sections.append(current)
                current = ""
            for i in range(0, len(paragraph), budget):
                sections.append(paragraph[i : i + budget])
            continue
        if len(current) + len(paragraph) + 2 <= budget:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        else:
            sections.append(current)
            current = paragraph
    if current:
        sections.append(current)
    return sections


def summarize(
    document: Document,
    engine: LLMEngine,
    *,
    length: str = "Medium",
    fmt: str = "Bullet Points",
    simple_language: bool = False,
    max_context_tokens: int = 100_000,
) -> SummaryResult:
    """Summarise a document, reducing in stages when it is very long."""
    target = target_word_count(document, length)

    if not document.full_text.strip():
        return SummaryResult(
            text="",
            length_label=length,
            fmt=fmt,
            target_words=target,
            actual_words=0,
            error="The document contains no extractable text.",
        )

    sections = split_into_sections(document.full_text, max_context_tokens)
    per_section_words = max(40, target // max(1, len(sections)))

    partials: list[str] = []
    used_model = ""
    errors: list[str] = []

    # ---- Stage 1: summarise each section independently -------------- #
    for section in sections:
        prompt = build_summary_prompt(
            section,
            length_label=f"{per_section_words}-word",
            target_words=per_section_words,
            fmt=fmt,
            simple_language=simple_language,
        )
        response = engine.generate(prompt, system=SUMMARY_SYSTEM)
        if response.ok:
            partials.append(response.text)
            used_model = used_model or response.model
        else:
            errors.append(response.error)

    if not partials:
        return SummaryResult(
            text="",
            length_label=length,
            fmt=fmt,
            target_words=target,
            actual_words=0,
            error=errors[0] if errors else "Summarisation failed.",
        )

    # ---- Stage 2: merge partials when we produced more than one ----- #
    if len(partials) == 1:
        final_text = partials[0]
    else:
        joined = "\n\n".join(f"- {p.strip()}" for p in partials)
        merge_prompt = build_summary_prompt(
            joined,
            length_label=length,
            target_words=target,
            fmt=fmt,
            simple_language=simple_language,
        )
        response = engine.generate(merge_prompt, system=SUMMARY_SYSTEM)
        if response.ok:
            final_text = response.text
            used_model = response.model
        else:
            # Reduction failed (usually rate limiting) - keep the partials.
            final_text = "\n\n".join(partials)
            errors.append(f"Merge step skipped: {response.error}")

    final_text = strip_markdown_noise(final_text)
    return SummaryResult(
        text=final_text,
        length_label=length,
        fmt=fmt,
        target_words=target,
        actual_words=len(final_text.split()),
        model=used_model,
        sections_summarised=len(sections),
        error="; ".join(errors) if len(partials) < len(sections) else "",
    )


def quality_report(result: SummaryResult, document: Document) -> dict:
    """Compare the produced summary against its target for the debug panel."""
    target = max(1, result.target_words)
    ratio = result.actual_words / target
    if ratio < 0.6:
        verdict = "Under target - the model may have been too terse."
    elif ratio > 1.6:
        verdict = "Over target - consider a shorter length setting."
    else:
        verdict = "On target."

    return {
        "Target words": result.target_words,
        "Actual words": result.actual_words,
        "Coverage": f"{ratio:.0%}",
        "Verdict": verdict,
        "Document compression": (
            f"{result.actual_words / document.word_count:.1%}" if document.word_count else "n/a"
        ),
        "Sections merged": result.sections_summarised,
    }
