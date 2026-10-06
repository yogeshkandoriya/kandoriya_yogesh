"""Deterministic in-memory engine used by tests and offline demos.

Because every generation call in the app goes through :class:`LLMEngine`,
substituting this class makes the entire test-suite run with no network
access and no model downloads.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .base import LLMEngine, LLMResponse


class MockEngine(LLMEngine):
    """Returns canned responses and records every prompt it received."""

    name = "mock"

    def __init__(
        self,
        responses: Optional[list[str]] = None,
        *,
        handler: Optional[Callable[[str], str]] = None,
        fail_with: str = "",
    ) -> None:
        #: Consumed in order; the last entry repeats once exhausted.
        self.responses = list(responses or ["Mock response."])
        self.handler = handler
        self.fail_with = fail_with
        #: Every prompt passed to :meth:`generate`, for assertions.
        self.calls: list[dict] = []
        self._cursor = 0

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system})
        if self.fail_with:
            return LLMResponse(text="", model=self.name, error=self.fail_with)

        if self.handler is not None:
            text = self.handler(prompt)
        else:
            idx = min(self._cursor, len(self.responses) - 1)
            text = self.responses[idx]
            self._cursor += 1

        # Honour max_tokens roughly, so truncation behaviour can be exercised.
        if max_tokens:
            text = self._truncate(text, max_tokens)
        return LLMResponse(
            text=text,
            model=self.name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
        )

    @staticmethod
    def _truncate(text: str, max_tokens: int) -> str:
        """Cut to ~4 chars/token, mirroring real token accounting."""
        limit = max_tokens * 4
        return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."

    # Convenience accessors for tests ------------------------------- #
    @property
    def last_prompt(self) -> str:
        return self.calls[-1]["prompt"] if self.calls else ""

    @property
    def call_count(self) -> int:
        return len(self.calls)


class ScriptedRetrievalEngine(MockEngine):
    """Mock that echoes back document content found in the prompt.

    Useful for verifying that grounding prompts actually contain the
    retrieved excerpts.
    """

    def __init__(self) -> None:
        super().__init__()

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system})
        # Pull sentences out of the "[Page N]" blocks the app supplies.
        pages = re.findall(r"\[Page (\d+)\][^\n]*\n(.*?)(?=\n\n\[Page |\Z)", prompt, re.S)
        if pages:
            body = " ".join(f"Page {p}: {t.strip()[:200]}" for p, t in pages[:2])
            return LLMResponse(text=body, model=self.name, prompt_tokens=len(prompt.split()))
        return LLMResponse(
            text="I could not find sufficient information in the uploaded PDF to answer this question.",
            model=self.name,
            prompt_tokens=len(prompt.split()),
        )
