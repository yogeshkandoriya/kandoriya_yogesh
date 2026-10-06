"""Tests for the OpenRouter provider, exports, analysis and the pipeline.

All tests are offline: HTTP is exercised through a fake session object rather
than real network calls.
"""

from __future__ import annotations

import json

import pytest

from summaryai.analysis import extract_keywords, find_term_pages
from summaryai.export import (
    safe_filename,
    to_docx,
    to_markdown,
    to_pdf,
    to_plain_text,
    to_pptx,
)
from summaryai.llm.mock import MockEngine
from summaryai.llm.openrouter import OpenRouterEngine
from summaryai.schemas import Citation


# --------------------------------------------------------------------------- #
# Fake HTTP layer
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class FakeSession:
    """Replays a scripted list of responses and records the payloads sent."""

    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0) if self.responses else FakeResponse(200, _ok("default"))


def _ok(content: str, model: str = "test/model") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _err(status: int, message: str) -> dict:
    return {"error": {"message": message}}


def _engine(responses: list[FakeResponse], **kwargs) -> OpenRouterEngine:
    session = FakeSession(responses)
    engine = OpenRouterEngine(
        api_key="sk-or-test", model="vendor/model", session=session,
        max_retries=kwargs.pop("max_retries", 2), **kwargs,
    )
    engine._session = session
    return engine


# --------------------------------------------------------------------------- #
# OpenRouter provider
# --------------------------------------------------------------------------- #
class TestOpenRouterEngine:
    def test_successful_call(self):
        engine = _engine([FakeResponse(200, _ok("Hello there"))])
        result = engine.generate("Hi")
        assert result.ok
        assert result.text == "Hello there"
        assert result.completion_tokens == 5

    def test_sends_auth_and_attribution_headers(self):
        engine = _engine([FakeResponse(200, _ok("ok"))], app_url="http://x", app_title="T")
        engine.generate("Hi")
        headers = engine._session.calls[0]["headers"]
        assert headers["Authorization"] == "Bearer sk-or-test"
        assert headers["HTTP-Referer"] == "http://x"
        assert headers["X-Title"] == "T"

    def test_disables_reasoning_by_default(self):
        engine = _engine([FakeResponse(200, _ok("ok"))])
        engine.generate("Hi")
        assert engine._session.calls[0]["json"]["reasoning"] == {"enabled": False}

    def test_missing_key_is_reported(self):
        result = OpenRouterEngine(api_key="", model="m").generate("Hi")
        assert not result.ok
        assert "OPENROUTER_API_KEY" in result.error

    def test_blank_key_is_reported(self):
        assert not OpenRouterEngine(api_key="not-a-key", model="m").is_configured()

    def test_empty_prompt_is_reported(self):
        assert not _engine([FakeResponse(200, _ok("x"))]).generate("   ").ok

    def test_empty_completion_is_an_error(self):
        result = _engine([FakeResponse(200, _ok("   "))]).generate("Hi")
        assert not result.ok
        assert "empty" in result.error.lower()

    def test_malformed_payload_is_an_error(self):
        result = _engine([FakeResponse(200, {"unexpected": True})]).generate("Hi")
        assert not result.ok
        assert "Malformed" in result.error

    def test_non_retryable_error_stops_immediately(self):
        engine = _engine([FakeResponse(402, _err(402, "insufficient credits"))])
        result = engine.generate("Hi")
        assert not result.ok
        assert "402" in result.error
        assert len(engine._session.calls) == 1

    def test_rate_limit_falls_back_to_another_model(self):
        engine = _engine(
            [FakeResponse(429, _err(429, "rate limited"))],
            fallback_models=("backup/model",),
            max_retries=1,
        )
        # First model exhausts retries, then the fallback succeeds.
        engine._session.responses = [
            FakeResponse(429, _err(429, "rate limited")),
            FakeResponse(200, _ok("from backup", "backup/model")),
        ]
        result = engine.generate("Hi")
        assert result.ok
        assert result.model == "backup/model"
        assert "vendor/model" in result.models_tried

    def test_retired_model_404_rotates_without_retrying(self):
        """Regression: OpenRouter retires free models (404). We must NOT burn
        retries on a model that no longer exists - rotate to the next one."""
        engine = _engine(
            [FakeResponse(404, _err(404, "Model not found"))],
            fallback_models=("alive/model",),
            max_retries=3,
        )
        engine._session.responses = [
            FakeResponse(404, _err(404, "Model not found")),      # dead model
            FakeResponse(200, _ok("ok", "alive/model")),          # fallback
        ]
        result = engine.generate("Hi")
        assert result.ok
        assert result.model == "alive/model"
        # Exactly one attempt against the dead model - no pointless retries.
        assert len(engine._session.calls) == 2

    def test_every_default_model_has_a_fallback(self):
        """The primary must never be duplicated in the fallback list."""
        from summaryai.config import DEFAULT_MODEL, FALLBACK_MODELS

        assert DEFAULT_MODEL not in FALLBACK_MODELS
        assert len(FALLBACK_MODELS) >= 2

    def test_every_configured_model_looks_free(self):
        """Every advertised model must be a zero-cost `:free` id."""
        from summaryai.config import DEFAULT_MODEL, FALLBACK_MODELS, FREE_MODELS

        for model_id in list(FREE_MODELS) + [DEFAULT_MODEL] + list(FALLBACK_MODELS):
            assert model_id.endswith(":free"), model_id

    def test_strips_think_blocks(self):
        engine = _engine([FakeResponse(200, _ok("<think>secret</think>The answer."))])
        assert engine.generate("Hi").text == "The answer."

    def test_strips_unterminated_think(self):
        engine = _engine([FakeResponse(200, _ok("<think>never ends"))])
        assert not engine.generate("Hi").ok  # nothing left after stripping

    def test_strips_reasoning_preamble(self):
        # The preamble marker is removed; the content after it is preserved.
        # We deliberately do NOT delete following numbered lists, because a
        # summary may legitimately contain them.
        engine = _engine([FakeResponse(200, _ok("Here's my thinking process:\n\nReal answer."))])
        assert engine.generate("Hi").text == "Real answer."

    def test_preserves_numbered_content(self):
        # A numbered list is valid output and must survive untouched.
        engine = _engine([FakeResponse(200, _ok("1. First finding\n2. Second finding"))])
        assert engine.generate("Hi").text == "1. First finding\n2. Second finding"

    def test_health_check(self):
        assert _engine([]).health() == (True, "Ready.")
        assert _engine([]).__class__(api_key="", model="m").health()[0] is False


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
class TestAnalysis:
    def test_extracts_keywords(self, sample_document):
        terms = [k.term for k in extract_keywords(sample_document, top_n=8)]
        assert len(terms) > 0
        assert any("motherboard" in t or "cpu" in t for t in terms)

    def test_keywords_respect_top_n(self, sample_document):
        assert len(extract_keywords(sample_document, top_n=3)) <= 3

    def test_keywords_on_empty_document(self):
        from summaryai.schemas import Document

        assert extract_keywords(Document.from_pages("e.pdf", [])) == []

    def test_find_term_pages(self, sample_document):
        hits = find_term_pages(sample_document, "CUDA cores")
        assert hits and hits[0]["page"] == 2
        assert "CUDA" in hits[0]["excerpt"]

    def test_find_missing_term(self, sample_document):
        assert find_term_pages(sample_document, "zzzznotpresent") == []

    def test_find_blank_term(self, sample_document):
        assert find_term_pages(sample_document, "  ") == []


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
class TestExports:
    CONTENT = "# Overview\n\n- First point about hardware.\n- Second point.\n\nBody paragraph text."

    def test_safe_filename(self):
        assert safe_filename("My Report: v1!") == "My_Report_v1"
        assert safe_filename("") == "summary"
        assert "/" not in safe_filename("a/b\\c")

    def test_markdown_includes_citations_and_stats(self):
        md = to_markdown("T", "body", [Citation(3, 0.9, "excerpt", "doc.pdf")], {"Pages": 4})
        assert "# T" in md
        assert "Page 3" in md
        assert "Pages" in md

    def test_plain_text_strips_markdown(self):
        text = to_plain_text("T", "**bold** and # Head")
        assert "**" not in text
        assert "Head" in text

    def test_docx_is_a_valid_zip(self):
        data = to_docx("Title", self.CONTENT, [Citation(1, 0.5, "x")])
        assert data[:2] == b"PK"  # zip magic number

    def test_pptx_is_a_valid_zip(self):
        assert to_pptx("Title", self.CONTENT)[:2] == b"PK"

    def test_pdf_is_a_valid_document(self):
        data = to_pdf("Title", self.CONTENT, [Citation(2, 0.4, "x")])
        assert data[:5] == b"%PDF-"

    def test_pdf_paginates_long_content(self):
        data = to_pdf("Title", "\n\n".join(["Line of text here."] * 400))
        assert data[:5] == b"%PDF-"
        assert len(data) > 3000

    def test_exports_tolerate_empty_content(self):
        assert to_docx("T", "")[:2] == b"PK"
        assert to_pptx("T", "")[:2] == b"PK"
        assert to_pdf("T", "")[:5] == b"%PDF-"


# --------------------------------------------------------------------------- #
# Pipeline orchestrator
# --------------------------------------------------------------------------- #
class TestPipeline:
    def test_add_and_list_document(self, assistant, sample_pdf):
        document = assistant.add_document(sample_pdf, "a.pdf", enable_ocr=False)
        assert document.page_count == 4
        assert assistant.names == ["a.pdf"]
        assert assistant.has_documents()

    def test_index_stats(self, loaded_assistant):
        stats = loaded_assistant.index_stats("sample.pdf")
        assert stats["total_chunks"] > 0
        assert stats["Pages indexed"] == 4

    def test_remove_document(self, loaded_assistant):
        assert loaded_assistant.remove_document("sample.pdf")
        assert not loaded_assistant.has_documents()
        assert "sample.pdf" not in loaded_assistant.indexes

    def test_remove_unknown_document(self, loaded_assistant):
        assert loaded_assistant.remove_document("nope.pdf") is False

    def test_clear(self, loaded_assistant):
        loaded_assistant.clear()
        assert not loaded_assistant.has_documents()

    def test_summarize_via_pipeline(self, loaded_assistant):
        result = loaded_assistant.summarize("sample.pdf", length="Short")
        assert result.ok
        assert result.coverage

    def test_ask_via_pipeline(self, loaded_assistant):
        result = loaded_assistant.ask("sample.pdf", "What does the CPU do?")
        assert result.answer

    def test_keywords_via_pipeline(self, loaded_assistant):
        assert len(loaded_assistant.keywords("sample.pdf", 5)) > 0

    def test_find_via_pipeline(self, loaded_assistant):
        assert loaded_assistant.find("sample.pdf", "CUDA")

    def test_search_via_pipeline(self, loaded_assistant):
        chunks, info = loaded_assistant.search("sample.pdf", "cooling")
        assert chunks
        assert "query_type" in info

    def test_operations_raise_for_unknown_document(self, loaded_assistant):
        with pytest.raises(KeyError):
            loaded_assistant.summarize("missing.pdf")

    def test_adding_twice_replaces_document(self, loaded_assistant, sample_pdf):
        loaded_assistant.add_document(sample_pdf, "sample.pdf", enable_ocr=False)
        assert loaded_assistant.names == ["sample.pdf"]
