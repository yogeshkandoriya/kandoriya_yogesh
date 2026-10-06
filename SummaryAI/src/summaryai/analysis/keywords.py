"""Keyword extraction and multi-document comparison."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..llm.base import LLMEngine
from ..schemas import Document
from ..generation.prompts import GROUNDED_SYSTEM, build_compare_prompt, strip_markdown_noise

#: Extremely common words that carry no topical signal in academic documents.
_BOILERPLATE = {
    "figure", "table", "chapter", "page", "section", "unit", "note", "example",
    "however", "therefore", "thus", "hence", "also", "may", "can", "will",
    "using", "used", "use", "based", "shown", "given", "following", "above",
    "below", "figure", "respectively", "etc", "ie", "eg", "within", "via",
    "paper", "report", "study", "author", "authors", "abstract", "keywords",
}

_WORD = re.compile(r"\b[a-zA-Z][a-zA-Z-]{2,}\b")


@dataclass
class Keyword:
    """A keyword with its document frequency weight."""

    term: str
    score: float
    count: int

    def to_dict(self) -> dict:
        return {"term": self.term, "score": round(self.score, 4), "count": self.count}


def extract_keywords(document: Document, top_n: int = 12) -> list[Keyword]:
    """Rank terms by TF-IDF, falling back to frequency on short documents.

    Fully offline and deterministic, so keyword extraction works even with no
    API key configured.
    """
    text = document.full_text
    if not text.strip():
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
        )
        matrix = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        raw_counts = Counter(_WORD.findall(text.lower()))
        pairs = sorted(zip(terms, scores), key=lambda kv: kv[1], reverse=True)

        results: list[Keyword] = []
        for term, score in pairs:
            if score <= 0:
                continue
            if all(part.lower() not in _BOILERPLATE for part in term.split()):
                results.append(Keyword(term, float(score), raw_counts.get(term.split()[0], 0)))
            if len(results) >= top_n:
                break
        if results:
            return results
    except Exception:
        pass  # Fall through to the frequency-based path.

    counts = Counter(
        w for w in _WORD.findall(text.lower()) if w not in _BOILERPLATE
    )
    return [
        Keyword(term, float(count), count)
        for term, count in counts.most_common(top_n)
    ]


def find_term_pages(document: Document, term: str, top_n: int = 5) -> list[dict]:
    """Pages where ``term`` appears, with a short surrounding excerpt."""
    needle = term.strip().lower()
    if not needle:
        return []

    hits: list[dict] = []
    for number, page_text in enumerate(document.pages, start=1):
        lowered = page_text.lower()
        index = lowered.find(needle)
        if index == -1:
            continue
        start = max(0, index - 180)
        end = min(len(page_text), index + len(needle) + 220)
        hits.append(
            {
                "page": number,
                "excerpt": ("..." if start else "") + page_text[start:end].strip() + ("..." if end < len(page_text) else ""),
            }
        )
        if len(hits) >= top_n:
            break
    return hits


def compare_documents(
    doc_a: Document,
    doc_b: Document,
    engine: LLMEngine,
    *,
    summary_a: str = "",
    summary_b: str = "",
    max_tokens: int = 1200,
) -> str:
    """Compare two documents.

    When no pre-computed summaries are supplied they are generated first, so
    the comparison prompt stays small even for long documents.
    """
    from ..generation.summarizer import summarize

    text_a = summary_a or summarize(doc_a, engine, length="Short").text
    text_b = summary_b or summarize(doc_b, engine, length="Short").text

    if not text_a.strip() or not text_b.strip():
        return "One of the documents could not be summarised, so a comparison is not possible."

    prompt = build_compare_prompt(text_a, text_b, doc_a.name, doc_b.name)
    response = engine.generate(prompt, system=GROUNDED_SYSTEM, max_tokens=max_tokens)
    if not response.ok:
        return f"Comparison failed: {response.error}"
    return strip_markdown_noise(response.text)
