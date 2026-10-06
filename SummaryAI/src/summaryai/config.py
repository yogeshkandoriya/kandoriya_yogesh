"""Application-wide configuration loaded from environment / .env file.

Priority order for secrets:
    1. Explicit argument
    2. Process environment variable
    3. ``.env`` file in the project root
    4. Streamlit secrets (only when running inside Streamlit)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Project root = .../SummaryAI (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env manually so we do not hard-depend on python-dotenv.
if ENV_FILE.exists():
    # utf-8-sig transparently strips a BOM if the file was written by PowerShell.
    for _line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _val = _line.split("=", 1)
        _key = _key.strip().lstrip("﻿").strip()
        if _key:
            os.environ.setdefault(_key, _val.strip().strip("'\""))


# --------------------------------------------------------------------------- #
# Model catalogue
# --------------------------------------------------------------------------- #

#: Verified working, zero-cost OpenRouter models suited to long-document work.
#: ``context`` is in tokens; used to decide how much text fits in one prompt.
#:
#: NOTE: OpenRouter retires free models without notice - `qwen/qwen3.8-27b:free`
#: was our original default and now returns 404. Keep a healthy fallback list
#: and treat 404 as a rotate-to-next-model signal (see llm/openrouter.py).
FREE_MODELS: dict[str, dict] = {
    "nvidia/nemotron-3-super-120b-a12b:free": {
        "label": "Nemotron 3 Super 120B (free)",
        "context": 262_144,
        "note": "Strong reasoning and long context. Default choice.",
    },
    "inclusionai/ling-3.0-flash-sante:free": {
        "label": "Ling 3.0 Flash (free)",
        "context": 262_144,
        "note": "Very fast responses.",
    },
    "nvidia/nemotron-3.5-lightning:free": {
        "label": "Nemotron 3.5 Lightning (free)",
        "context": 1_000_000,
        "note": "Huge context window for very long documents.",
    },
    "apodex/apodex-1.1-mini:free": {
        "label": "Apodex 1.1 Mini (free)",
        "context": 262_144,
        "note": "Research oriented.",
    },
    "google/gemma-4-31b-it:free": {
        "label": "Gemma 4 31B IT (free)",
        "context": 262_144,
        "note": "Google model. May return 429 under load - auto-retries.",
    },
}

DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

#: Tried in order when the primary model is rate-limited, retired, or unavailable.
FALLBACK_MODELS: tuple[str, ...] = (
    "inclusionai/ling-3.0-flash-sante:free",
    "nvidia/nemotron-3.5-lightning:free",
    "apodex/apodex-1.1-mini:free",
    "google/gemma-4-31b-it:free",
)


def _streamlit_secret(key: str) -> str:
    """Read a secret from Streamlit's secret store if we are inside Streamlit."""
    try:
        import streamlit as st  # noqa: WPS433 (optional dependency)

        return str(st.secrets.get(key, "") or "")
    except Exception:
        return ""


def get_api_key(explicit: str = "") -> str:
    """Resolve the OpenRouter API key from any supported source."""
    if explicit and explicit.strip():
        return explicit.strip()
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key
    return _streamlit_secret("OPENROUTER_API_KEY").strip()


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass
class Settings:
    """Runtime settings for the whole application."""

    api_key: str = field(default_factory=get_api_key)
    model: str = field(default_factory=lambda: os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    app_url: str = field(default_factory=lambda: os.environ.get("OPENROUTER_APP_URL", "http://localhost:8501"))
    app_title: str = field(default_factory=lambda: os.environ.get("OPENROUTER_APP_TITLE", "Summary AI"))

    # Retrieval
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_chars: int = 900
    chunk_overlap: int = 120
    top_k: int = 4

    # Generation
    temperature: float = 0.2
    max_tokens: int = 1400
    request_timeout: int = 90
    max_retries: int = 3

    # Upload limits
    max_files: int = 10
    max_file_mb: int = 200

    def model_context(self, model: Optional[str] = None) -> int:
        """Context window (tokens) for a model id, with a safe default."""
        return FREE_MODELS.get(model or self.model, {}).get("context", 128_000)

    def has_key(self) -> bool:
        return bool(self.api_key)


def get_settings(**overrides) -> Settings:
    """Build a :class:`Settings` instance, re-reading the environment each call."""
    base = Settings()
    for key, value in overrides.items():
        if value is not None and hasattr(base, key):
            setattr(base, key, value)
    return base
