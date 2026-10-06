"""Shared pytest fixtures.

Every test runs fully offline: a synthetic PDF is generated in-process, and
``MockEngine`` stands in for the LLM. No model downloads, no network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summaryai.llm.mock import MockEngine  # noqa: E402
from summaryai.pipeline import DocumentAssistant  # noqa: E402
from summaryai.retrieval.embeddings import HashEmbedder  # noqa: E402
from summaryai.schemas import Document  # noqa: E402

SAMPLE_PAGES = [
    "COMPUTER ARCHITECTURE\n\nPage 1\n\n"
    "The motherboard is the main circuit board of a computer. It connects the CPU, "
    "RAM and the GPU. The CPU executes program instructions at 3.5 GHz.\n\n"
    "The RAM stores temporary data while programs are running.",

    "COMPUTER ARCHITECTURE\n\nPage 2\n\n"
    "The GPU handles graphics rendering using thousands of CUDA cores. "
    "It is essential for video editing and 3D modelling workloads.",

    "COMPUTER ARCHITECTURE\n\nPage 3\n\n"
    "The cooling system uses fans and heatsinks. Thermal paste between the "
    "processor and the cooler improves heat transfer. Dust removal is advised yearly.",

    "COMPUTER ARCHITECTURE\n\nPage 4\n\n"
    "The power supply unit converts AC mains to regulated DC. A 650W unit is "
    "recommended for a mid-range gaming build assembled in 2023.",
]


def make_pdf_bytes(pages: list[str]) -> bytes:
    """Build a minimal multi-page PDF containing ``pages`` as text."""
    import pymupdf

    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 780), text, fontsize=10)
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def sample_pdf() -> bytes:
    """A real 4-page PDF with repeatable headers and page numbers."""
    return make_pdf_bytes(SAMPLE_PAGES)


@pytest.fixture
def mock_engine() -> MockEngine:
    return MockEngine(
        [
            "- The motherboard connects the CPU, RAM and GPU.\n"
            "- The CPU executes instructions at 3.5 GHz.\n"
            "- The GPU renders graphics using CUDA cores."
        ]
    )


@pytest.fixture
def assistant(mock_engine) -> DocumentAssistant:
    """An assistant with lightweight (hash) embeddings and a mock LLM."""
    return DocumentAssistant(engine=mock_engine, lightweight_embeddings=True)


@pytest.fixture
def loaded_assistant(assistant, sample_pdf) -> DocumentAssistant:
    assistant.add_document(sample_pdf, "sample.pdf", enable_ocr=False, file_size=len(sample_pdf))
    return assistant


@pytest.fixture
def sample_document() -> Document:
    return Document.from_pages("sample.pdf", SAMPLE_PAGES)


@pytest.fixture
def hash_embedder() -> HashEmbedder:
    return HashEmbedder()
