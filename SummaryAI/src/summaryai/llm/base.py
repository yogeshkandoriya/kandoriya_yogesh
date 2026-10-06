"""Abstract LLM interface.

Every generation backend implements :class:`LLMEngine`, so the rest of the
application never needs to know which provider is active. This is what makes
the test-suite fast and offline: tests inject :class:`MockEngine`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Normalised response from any provider."""

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    error_kind: Optional[int] = None   # HTTP status when the failure was remote
    attempts: int = 1
    models_tried: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMEngine(abc.ABC):
    """Common interface for all text-generation backends."""

    name: str = "base"

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Generate a completion for ``prompt``.

        Implementations must never raise for expected failures; they return an
        :class:`LLMResponse` with ``error`` populated instead.
        """

    def health(self) -> tuple[bool, str]:
        """Cheap readiness probe. Returns ``(ok, message)``."""
        if not self.is_configured():
            return False, "No API key configured."
        return True, "Ready."

    def is_configured(self) -> bool:
        """Whether this engine has everything it needs to run."""
        return True
