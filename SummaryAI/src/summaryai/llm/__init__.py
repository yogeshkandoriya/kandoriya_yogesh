"""LLM backends. All generation funnels through the :class:`LLMEngine` protocol."""

from .base import LLMEngine, LLMResponse
from .mock import MockEngine, ScriptedRetrievalEngine
from .openrouter import OpenRouterEngine

__all__ = [
    "LLMEngine",
    "LLMResponse",
    "MockEngine",
    "ScriptedRetrievalEngine",
    "OpenRouterEngine",
    "build_engine",
]


def build_engine(settings, fallback_models=()) -> LLMEngine:
    """Construct the default engine described by ``settings``.

    Falls back to a :class:`MockEngine` when no API key is available so the
    application still starts and the UI can explain the problem.
    """
    if not settings.has_key():
        return MockEngine([f"No API key configured. Set OPENROUTER_API_KEY in {settings.app_title} .env file."])
    return OpenRouterEngine(
        api_key=settings.api_key,
        model=settings.model,
        fallback_models=fallback_models,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        app_url=settings.app_url,
        app_title=settings.app_title,
    )
