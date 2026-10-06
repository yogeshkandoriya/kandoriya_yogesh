"""Grounded question answering with verifiable page citations.

The flow is: retrieve evidence -> check the evidence is good enough -> ask the
model -> validate the answer -> attach citations. If any stage decides the
evidence is insufficient, we return the standard refusal instead of a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import LLMEngine
from ..retrieval.hybrid import hybrid_search
from ..retrieval.vector_store import VectorStore
from ..schemas import Citation, Document
from .prompts import (
    GROUNDED_SYSTEM,
    STANDARD_REFUSAL,
    build_explain_prompt,
    build_qa_prompt,
    is_refusal,
    strip_markdown_noise,
)

#: Below these retrieval scores the evidence is considered too weak to trust.
MIN_SEMANTIC_SCORE = 0.12
MIN_COMBINED_SCORE = 0.18


@dataclass
class AnswerResult:
    """A grounded answer plus its supporting citations and debug trace."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    model: str = ""
    query_type: str = "general"
    evidence: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.answer.strip())

    @property
    def pages(self) -> list[int]:
        return sorted({c.page for c in self.citations})

    def citation_footer(self) -> str:
        """Markdown source line rendered under the answer."""
        if not self.citations:
            return ""
        pages = ", ".join(f"Page {p}" for p in self.pages)
        return f"\n\n**Source:** {pages}"


def _evidence_is_sufficient(chunks: list) -> tuple[bool, str]:
    """Guard against answering from irrelevant retrieved text."""
    if not chunks:
        return False, "No relevant passages were found in the document."

    best = chunks[0]
    if best.semantic_score < MIN_SEMANTIC_SCORE and best.score < MIN_COMBINED_SCORE:
        return False, "The closest passages in the document were not relevant enough to answer this."

    return True, ""


def answer_question(
    question: str,
    document: Document,
    store: VectorStore,
    engine: LLMEngine,
    *,
    top_k: int = 4,
    max_tokens: int = 800,
) -> AnswerResult:
    """Answer a free-form question grounded in ``document``."""
    question = (question or "").strip()
    if not question:
        return AnswerResult(question="", answer="", refused=True, reason="Empty question.")

    chunks, info = hybrid_search(question, store, top_k=top_k)

    sufficient, why = _evidence_is_sufficient(chunks)
    if not sufficient:
        return AnswerResult(
            question=question,
            answer=STANDARD_REFUSAL,
            refused=True,
            reason=why,
            query_type=info["query_type"],
        )

    context = "\n\n".join(
        f"[Page {c.page}]{' - ' + c.section if c.section else ''}\n{c.text}" for c in chunks
    )
    response = engine.generate(build_qa_prompt(context, question), system=GROUNDED_SYSTEM, max_tokens=max_tokens)

    if not response.ok:
        return AnswerResult(
            question=question,
            answer="",
            error=response.error,
            query_type=info["query_type"],
            model=response.model,
        )

    answer = strip_markdown_noise(response.text)

    # Post-hoc validation: the model may still have hedged despite good evidence.
    if is_refusal(answer):
        return AnswerResult(
            question=question,
            answer=STANDARD_REFUSAL,
            citations=[Citation(c.page, c.score, c.text[:450], document.name) for c in chunks[:3]],
            refused=True,
            reason="The retrieved passages did not contain a definitive answer.",
            model=response.model,
            query_type=info["query_type"],
            evidence=[c.to_dict() for c in chunks],
        )

    return AnswerResult(
        question=question,
        answer=answer,
        citations=[Citation(c.page, c.score, c.text[:450], document.name) for c in chunks[:3]],
        refused=False,
        model=response.model,
        query_type=info["query_type"],
        evidence=[c.to_dict() for c in chunks],
    )


def explain_topic(
    topic: str,
    document: Document,
    store: VectorStore,
    engine: LLMEngine,
    *,
    top_k: int = 4,
    max_tokens: int = 900,
) -> AnswerResult:
    """Explain a specific topic, refusing if the document does not cover it."""
    topic = (topic or "").strip()
    if not topic:
        return AnswerResult(question="", answer="", refused=True, reason="No topic supplied.")

    chunks, info = hybrid_search(f"explain {topic}", store, top_k=top_k)
    sufficient, why = _evidence_is_sufficient(chunks)
    if not sufficient:
        return AnswerResult(
            question=topic,
            answer=STANDARD_REFUSAL,
            refused=True,
            reason=why,
            query_type=info["query_type"],
        )

    context = "\n\n".join(f"[Page {c.page}]\n{c.text}" for c in chunks)
    response = engine.generate(
        build_explain_prompt(context, topic), system=GROUNDED_SYSTEM, max_tokens=max_tokens
    )
    if not response.ok:
        return AnswerResult(question=topic, answer="", error=response.error, model=response.model)

    answer = strip_markdown_noise(response.text)
    refused = is_refusal(answer)
    return AnswerResult(
        question=topic,
        answer=STANDARD_REFUSAL if refused else answer,
        citations=[Citation(c.page, c.score, c.text[:450], document.name) for c in chunks[:3]],
        refused=refused,
        reason="The document does not cover this topic." if refused else "",
        model=response.model,
        query_type=info["query_type"],
        evidence=[c.to_dict() for c in chunks],
    )
