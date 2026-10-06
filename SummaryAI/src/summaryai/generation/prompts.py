"""Prompt templates and anti-hallucination guards.

All generation is grounded: the model only ever sees retrieved excerpts and is
instructed to refuse rather than guess. :func:`is_refusal` detects attempts to
answer without evidence so the pipeline can override the model.
"""

from __future__ import annotations

import re

#: The single canonical refusal. The model is told to emit this verbatim.
STANDARD_REFUSAL = (
    "I could not find sufficient information in the uploaded PDF to answer this question."
)

GROUNDED_SYSTEM = (
    "You are a precise document-analysis assistant.\n"
    "Rules you must follow without exception:\n"
    "1. Answer ONLY from the excerpts you are given. Never use outside knowledge.\n"
    "2. Never invent facts, numbers, names or conclusions that are not in the excerpts.\n"
    "3. If the excerpts do not contain the answer, reply with exactly:\n"
    f'   "{STANDARD_REFUSAL}"\n'
    "4. Keep the language clear and direct. Do not mention the excerpts themselves.\n"
    "5. Preserve important figures, dates and terminology exactly as written."
)

SUMMARY_SYSTEM = (
    "You are an expert technical summariser. Produce faithful, well-organised "
    "summaries of the supplied document excerpts.\n"
    "Rules:\n"
    "1. Use ONLY the provided content. Do not add outside information.\n"
    "2. Cover the document's actual structure: purpose, methods, findings, conclusions.\n"
    "3. Keep every number, date and proper noun exactly as written in the source.\n"
    "4. Do not invent sections the document does not contain."
)

# Phrases that indicate the model is hedging instead of answering.
_REFUSAL_MARKERS = (
    "not mentioned in the document",
    "not specified in the document",
    "not provided in the document",
    "not available in the document",
    "does not contain information",
    "doesn't contain information",
    "no information about",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "i don't know",
    "i do not know",
    "not enough information",
    "insufficient information",
    "no relevant information",
    "not found in the document",
    "cannot be determined",
    "cannot find",
    "unable to find",
)


def is_refusal(text: str) -> bool:
    """True if the model declined to answer for lack of evidence."""
    if not text or not text.strip():
        return True
    lowered = text.lower()
    if STANDARD_REFUSAL.lower() in lowered:
        return True
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def format_excerpts(context: str, *, max_chars: int = 12000) -> str:
    """Trim retrieved context to a prompt-safe size."""
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rsplit("\n\n", 1)[0] + "\n\n[...truncated...]"


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #
def build_summary_prompt(
    content: str,
    *,
    length_label: str,
    target_words: int,
    fmt: str,
    simple_language: bool = False,
) -> str:
    """Prompt for a length-targeted summary in the requested format."""
    format_rules = {
        "Bullet Points": (
            "Write the summary as concise bullet points, one idea per bullet. "
            "Start each bullet with a capital letter and end it with a period."
        ),
        "Key Findings": (
            "Write the summary as a numbered list of key findings, ordered from "
            "most to least important."
        ),
        "Paragraph": (
            "Write the summary as three to five flowing paragraphs. Do not use "
            "bullet points or headings."
        ),
        "FAQ": (
            "Write the summary as a FAQ: five to seven questions a reader would "
            "genuinely ask, each followed by a short direct answer."
        ),
        "Structured": (
            "Organise the summary under short markdown headings such as "
            "'Overview', 'Key Concepts', 'Details' and 'Conclusions', using "
            "bullet points under each heading."
        ),
    }.get(fmt, "Write the summary as concise bullet points.")

    language = (
        "Use very simple, easy-to-understand language, as if explaining to a beginner. "
        if simple_language
        else ""
    )

    return (
        f"Write a {length_label.lower()} summary of the document below.\n\n"
        f"Requirements:\n"
        f"- Aim for approximately {target_words} words.\n"
        f"- {format_rules}\n"
        f"- Every claim must be supported by the document text.\n"
        f"- Do not add an introduction or a conclusion about the summary itself; "
        f"begin directly with the content.\n"
        f"{language}\n"
        f"DOCUMENT:\n\"\"\"\n{content}\n\"\"\""
    )


def build_qa_prompt(context: str, question: str) -> str:
    """Prompt for a grounded, citation-aware answer."""
    return (
        f"Answer the question using ONLY the document excerpts below.\n\n"
        f"Excerpts:\n\"\"\"\n{format_excerpts(context)}\n\"\"\"\n\n"
        f"Question: {question}\n\n"
        f"If the excerpts contain the answer, answer clearly and completely. "
        f"Otherwise reply with exactly: \"{STANDARD_REFUSAL}\""
    )


def build_explain_prompt(context: str, topic: str) -> str:
    return (
        f"Using ONLY the excerpts below, explain '{topic}' from the document.\n\n"
        f"Structure your response as:\n"
        f"1. Simple Explanation\n"
        f"2. Important Points\n"
        f"3. Relevant Details\n\n"
        f"Do not invent facts. If the excerpts do not cover '{topic}', reply with "
        f"exactly: \"{STANDARD_REFUSAL}\"\n\n"
        f"Excerpts:\n\"\"\"\n{format_excerpts(context)}\n\"\"\""
    )


def build_keyword_prompt(context: str, count: int) -> str:
    return (
        f"List the {count} most important keywords or key terms in this document.\n"
        f"Return them as a plain list, one per line, with no numbering and no "
        f"extra commentary.\n\n"
        f"Excerpts:\n\"\"\"\n{format_excerpts(context, max_chars=8000)}\n\"\"\""
    )


def build_compare_prompt(doc_a: str, doc_b: str, name_a: str, name_b: str) -> str:
    return (
        f"Compare two documents and describe their similarities and differences.\n\n"
        f"Document 1 - {name_a}:\n\"\"\"\n{doc_a}\n\"\"\"\n\n"
        f"Document 2 - {name_b}:\n\"\"\"\n{doc_b}\n\"\"\"\n\n"
        f"Answer under these headings:\n"
        f"1. Overview of {name_a}\n"
        f"2. Overview of {name_b}\n"
        f"3. Key Similarities\n"
        f"4. Key Differences\n"
        f"Base every statement strictly on the two summaries provided."
    )


def build_faq_prompt(context: str, count: int) -> str:
    return (
        f"Write {count} question-and-answer pairs about this document, covering "
        f"the most important content.\n"
        f"Format each pair as:\nQ: <question>\nA: <answer>\n\n"
        f"Use only information present in the excerpts.\n\n"
        f"Excerpts:\n\"\"\"\n{format_excerpts(context, max_chars=10000)}\n\"\"\""
    )


def strip_markdown_noise(text: str) -> str:
    """Tidy model output: collapse excess blank lines and stray artefacts."""
    text = re.sub(r"\n{3,}", "\n\n", text or "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()
