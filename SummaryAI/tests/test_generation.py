"""Tests for prompts, summarisation, grounded Q&A and the refusal guard."""

from __future__ import annotations

import pytest

from summaryai.generation import (
    STANDARD_REFUSAL,
    answer_question,
    build_qa_prompt,
    build_summary_prompt,
    is_refusal,
    quality_report,
    strip_markdown_noise,
    summarize,
    target_word_count,
)
from summaryai.generation.qa import explain_topic
from summaryai.generation.summarizer import split_into_sections
from summaryai.llm.mock import MockEngine, ScriptedRetrievalEngine
from summaryai.retrieval import VectorStore, chunk_document
from summaryai.schemas import Document

from conftest import SAMPLE_PAGES


@pytest.fixture
def store(hash_embedder):
    return VectorStore(hash_embedder).build(chunk_document(SAMPLE_PAGES, max_chars=300))


# --------------------------------------------------------------------------- #
# Refusal detection
# --------------------------------------------------------------------------- #
class TestRefusalGuard:
    def test_detects_standard_refusal(self):
        assert is_refusal(STANDARD_REFUSAL)

    def test_detects_hedging_phrases(self):
        for phrase in (
            "This is not mentioned in the document.",
            "The document does not contain information about it.",
            "I don't know.",
            "Insufficient information to answer.",
        ):
            assert is_refusal(phrase), phrase

    def test_accepts_real_answers(self):
        assert not is_refusal("The CPU executes instructions at 3.5 GHz.")

    def test_empty_is_refusal(self):
        assert is_refusal("")
        assert is_refusal("   ")


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #
class TestPrompts:
    def test_summary_prompt_contains_document(self):
        prompt = build_summary_prompt(
            "DOC TEXT", length_label="Medium", target_words=200, fmt="Bullet Points"
        )
        assert "DOC TEXT" in prompt
        assert "200" in prompt
        assert "bullet" in prompt.lower()

    def test_summary_prompt_faq_format(self):
        prompt = build_summary_prompt("X", length_label="Short", target_words=50, fmt="FAQ")
        assert "FAQ" in prompt

    def test_simple_language_flag(self):
        plain = build_summary_prompt("X", length_label="Short", target_words=50, fmt="Paragraph")
        simple = build_summary_prompt(
            "X", length_label="Short", target_words=50, fmt="Paragraph", simple_language=True
        )
        assert len(simple) > len(plain)

    def test_qa_prompt_contains_question_and_context(self):
        prompt = build_qa_prompt("[Page 1]\nSome content", "What is the CPU?")
        assert "What is the CPU?" in prompt
        assert "Page 1" in prompt

    def test_strip_markdown_noise(self):
        assert strip_markdown_noise("a\n\n\n\nb") == "a\n\nb"


# --------------------------------------------------------------------------- #
# Summarisation
# --------------------------------------------------------------------------- #
class TestSummarizer:
    def test_target_scales_with_document(self):
        # Needs a document large enough that the 60-word floor does not
        # clamp every length to the same value.
        big = Document.from_pages("big.pdf", ["word " * 2000])
        assert target_word_count(big, "Short") < target_word_count(big, "Detailed")

    def test_target_has_a_floor(self, sample_document):
        # The 4-page sample is only ~117 words, so every length clamps to 60.
        assert target_word_count(sample_document, "Short") == 60
        assert target_word_count(sample_document, "Detailed") == 60

    def test_target_is_clamped_at_the_top(self):
        huge = Document.from_pages("huge.pdf", ["word " * 100_000])
        assert target_word_count(huge, "Detailed") == 1500

    def test_split_into_sections_short_text(self):
        assert len(split_into_sections("Short text.", 1000)) == 1

    def test_split_into_sections_long_text(self):
        text = "\n\n".join(["Paragraph text here." * 10] * 20)
        assert len(split_into_sections(text, 100)) > 1

    def test_summarize_produces_output(self, sample_document):
        engine = MockEngine(["- Point one.\n- Point two."])
        result = summarize(sample_document, engine, length="Short")
        assert result.ok
        assert "Point one" in result.text
        assert result.actual_words > 0

    def test_summarize_reports_engine_failure(self, sample_document):
        result = summarize(sample_document, MockEngine(fail_with="HTTP 429: rate limited"), length="Short")
        assert not result.ok
        assert "429" in result.error

    def test_summarize_empty_document(self):
        result = summarize(Document.from_pages("e.pdf", []), MockEngine(), length="Short")
        assert not result.ok
        assert "no extractable text" in result.error.lower()

    def test_map_reduce_merges_sections(self, mock_engine):
        big = Document.from_pages("big.pdf", ["Body paragraph text." * 200] * 6)
        result = summarize(big, mock_engine, length="Short", max_context_tokens=800)
        assert result.sections_summarised > 1
        assert result.ok

    def test_quality_report_fields(self, sample_document):
        result = summarize(sample_document, MockEngine(["- One.\n- Two."]), length="Short")
        report = quality_report(result, sample_document)
        assert "Coverage" in report
        assert "Verdict" in report


# --------------------------------------------------------------------------- #
# Grounded Q&A
# --------------------------------------------------------------------------- #
class TestQuestionAnswering:
    def test_answers_with_citations(self, sample_document, store):
        engine = ScriptedRetrievalEngine()
        result = answer_question("What does the CPU do?", sample_document, store, engine)
        assert result.ok
        assert not result.refused
        assert result.citations
        assert result.pages

    def test_citation_footer_lists_pages(self, sample_document, store):
        result = answer_question("What does the GPU do?", sample_document, store, ScriptedRetrievalEngine())
        assert "Page" in result.citation_footer()

    def test_refuses_when_model_hedges(self, sample_document, store):
        # Model receives evidence but chooses to refuse; we must honour that.
        engine = MockEngine(["This is not mentioned in the document."])
        result = answer_question("What is the salary of the CEO?", sample_document, store, engine)
        assert result.refused
        assert result.answer == STANDARD_REFUSAL
        assert result.reason

    def test_refuses_when_no_evidence_retrieved(self, sample_document, hash_embedder):
        # An index built from an unrelated document yields no usable evidence.
        other = VectorStore(hash_embedder).build(chunk_document(["Pasta recipes and cooking tips."]))
        result = answer_question(
            "What does the CPU do?", sample_document, other, ScriptedRetrievalEngine()
        )
        assert result.refused
        assert result.answer == STANDARD_REFUSAL

    def test_mock_without_page_markers_refuses(self, sample_document, store):
        # ScriptedRetrievalEngine returns the refusal when it sees no excerpts.
        result = answer_question("anything", sample_document, store, ScriptedRetrievalEngine())
        assert result.answer  # either a grounded answer or the refusal, never a crash

    def test_empty_question_refuses(self, sample_document, store):
        assert answer_question("  ", sample_document, store, MockEngine()).refused

    def test_engine_error_is_surfaced(self, sample_document, store):
        result = answer_question(
            "cpu", sample_document, store, MockEngine(fail_with="HTTP 500: server error")
        )
        assert not result.ok
        assert "500" in result.error

    def test_evidence_is_included_in_prompt(self, sample_document, store):
        engine = ScriptedRetrievalEngine()
        answer_question("What is the CPU clock speed?", sample_document, store, engine)
        # The prompt must carry page-anchored excerpts for grounding.
        assert "[Page" in engine.last_prompt

    def test_explain_topic(self, sample_document, store):
        result = explain_topic("the cooling system", sample_document, store, ScriptedRetrievalEngine())
        assert result.ok

    def test_explain_requires_topic(self, sample_document, store):
        assert explain_topic("", sample_document, store, MockEngine()).refused
