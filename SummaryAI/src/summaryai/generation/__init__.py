"""Generation: prompts, summarisation, and grounded question answering."""

from .prompts import (
    GROUNDED_SYSTEM,
    STANDARD_REFUSAL,
    SUMMARY_SYSTEM,
    build_compare_prompt,
    build_explain_prompt,
    build_faq_prompt,
    build_keyword_prompt,
    build_qa_prompt,
    build_summary_prompt,
    format_excerpts,
    is_refusal,
    strip_markdown_noise,
)
from .qa import AnswerResult, answer_question, explain_topic
from .summarizer import LENGTH_TARGETS, SummaryResult, quality_report, summarize, target_word_count

__all__ = [
    "GROUNDED_SYSTEM",
    "STANDARD_REFUSAL",
    "SUMMARY_SYSTEM",
    "build_compare_prompt",
    "build_explain_prompt",
    "build_faq_prompt",
    "build_keyword_prompt",
    "build_qa_prompt",
    "build_summary_prompt",
    "format_excerpts",
    "is_refusal",
    "strip_markdown_noise",
    "AnswerResult",
    "answer_question",
    "explain_topic",
    "LENGTH_TARGETS",
    "SummaryResult",
    "quality_report",
    "summarize",
    "target_word_count",
]
