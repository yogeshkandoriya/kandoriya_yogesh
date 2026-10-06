"""OpenRouter chat-completions provider.

Handles: authentication, retries with exponential backoff, rate-limit (429)
failover across a list of free models, and token accounting.
"""

from __future__ import annotations

import random
import re
import time
from typing import Optional, Sequence

import requests

from .base import LLMEngine, LLMResponse

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

#: Errors worth retrying on the same model.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 522, 524}


class OpenRouterEngine(LLMEngine):
    """Generate text through the OpenRouter API."""

    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        fallback_models: Sequence[str] = (),
        temperature: float = 0.2,
        max_tokens: int = 1400,
        timeout: int = 90,
        max_retries: int = 3,
        app_url: str = "http://localhost:8501",
        app_title: str = "Summary AI",
        session: Optional[requests.Session] = None,
        disable_reasoning: bool = True,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self.fallback_models = tuple(m for m in fallback_models if m != model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.app_url = app_url
        self.app_title = app_title
        # Many free models emit visible chain-of-thought by default, which would
        # pollute summaries. We turn it off and also strip it defensively.
        self.disable_reasoning = disable_reasoning
        self._session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def is_configured(self) -> bool:
        return self.api_key.startswith("sk-or-")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter uses these for dashboard attribution.
            "HTTP-Referer": self.app_url,
            "X-Title": self.app_title,
        }

    @property
    def _candidate_models(self) -> list[str]:
        return [self.model, *self.fallback_models]

    # ------------------------------------------------------------------ #
    # Single request attempt
    # ------------------------------------------------------------------ #
    def _call_once(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Issue one HTTP request. Never raises; failures come back as an error field."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.disable_reasoning:
            # Ask the provider not to expose chain-of-thought in the content.
            payload["reasoning"] = {"enabled": False}
        try:
            resp = self._session.post(
                ENDPOINT,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            return LLMResponse(text="", model=model, error="Request timed out.")
        except requests.exceptions.RequestException as exc:
            return LLMResponse(text="", model=model, error=f"Network error: {exc}")

        if resp.status_code != 200:
            return LLMResponse(
                text="",
                model=model,
                error=f"HTTP {resp.status_code}: {self._extract_error(resp)}",
                # Expose the status so the caller can decide to rotate models.
                error_kind=resp.status_code,
            )

        try:
            data = resp.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except (ValueError, KeyError, IndexError) as exc:
            return LLMResponse(text="", model=model, error=f"Malformed response: {exc}")

        text = self._strip_reasoning(text)
        if not text:
            return LLMResponse(text="", model=model, error="Model returned an empty response.")

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", model),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Remove leaked chain-of-thought that some models emit in-band.

        Even with ``reasoning.enabled=False`` a few free models still prefix
        their scratchpad, so this is a second line of defence.
        """
        if not text:
            return text
        # Explicit think blocks.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
        text = re.sub(r"^.*?</think>", "", text, flags=re.S) if "</think>" in text else text
        # Unterminated block, or an unclosed opening tag.
        if "<think>" in text:
            text = re.sub(r"<think>.*", "", text, flags=re.S | re.I)
        # Prefixed prose such as "Here's my thinking process:\n\n1. ..."
        text = re.sub(
            r"^\s*(here'?s?\s+(is\s+)?(my\s+)?(thinking|reasoning|analysis)\b.*?:\s*)+",
            "",
            text,
            flags=re.I,
        )
        return text.strip()

    @staticmethod
    def _extract_error(resp: "requests.Response") -> str:
        """Pull a human-readable message out of an OpenRouter error body."""
        try:
            body = resp.json().get("error", {})
            return str(body.get("message") or body)[:300]
        except Exception:
            return resp.text[:300]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        if not self.is_configured():
            return LLMResponse(
                text="",
                model=self.model,
                error="OPENROUTER_API_KEY is missing or malformed. Add it to .env and restart.",
            )
        if not prompt.strip():
            return LLMResponse(text="", model=self.model, error="Prompt is empty.")

        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]

        models_tried: list[str] = []
        last_error = "Unknown error."

        for model in self._candidate_models:
            models_tried.append(model)
            for attempt in range(1, self.max_retries + 1):
                result = self._call_once(model, messages, max_tokens, temperature)

                if result.ok:
                    result.attempts = attempt
                    result.models_tried = tuple(models_tried)
                    return result

                last_error = result.error

                if result.error_kind in RETRYABLE_STATUS and attempt < self.max_retries:
                    # Exponential backoff with jitter; free tiers are contended.
                    time.sleep(min(2 ** attempt + random.uniform(0, 1), 30))
                    continue

                # Non-retryable (402 quota, 404 model retired, 400 bad request)
                # or retries exhausted -> abandon this model and try the next.
                break

        return LLMResponse(
            text="",
            model=models_tried[-1] if models_tried else self.model,
            error=last_error,
            attempts=self.max_retries,
            models_tried=tuple(models_tried),
        )
