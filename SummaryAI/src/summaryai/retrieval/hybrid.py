"""Hybrid retrieval: dense semantics + lexical precision, then re-ranking.

Pure vector search is bad at questions containing exact numbers, dates or
quoted terms ("what rate was applied in 2024?"), because MiniLM blurs precise
tokens. This module classifies the query and weights the two signals
accordingly, then applies thresholding, deduplication and neighbour expansion.
"""

from __future__ import annotations

import re

from ..schemas import Chunk
from .vector_store import VectorStore

# Words that indicate a factual, lookup-style question.
_NUMERIC = re.compile(r"\d")
_DATE = re.compile(
    r"\b(19|20)\d{2}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.I,
)
_DEFINITION = re.compile(
    r"\b(what\s+is|define|definition|meaning|means|refers\s+to|explain\s+the\s+concept)\b", re.I
)
_FORMULA = re.compile(r"[=≈≤≥<>]|\b(formula|equation|theorem|calculate|compute)\b", re.I)
_QUOTED = re.compile(r"[\"“']([^\"”']{3,60})[\"”']")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "that", "this",
    "these", "those", "it", "its", "what", "which", "who", "whom", "how", "why",
    "when", "where", "does", "do", "did", "can", "could", "will", "would", "should",
    "there", "their", "his", "her", "about", "into", "over", "than", "then", "so",
    "if", "not", "no", "yes", "you", "your", "we", "our", "they", "them", "he", "she",
    "me", "my", "have", "has", "had", "am", "up", "out", "else",
}


# --------------------------------------------------------------------------- #
# Query analysis
# --------------------------------------------------------------------------- #
def analyze_query(query: str) -> dict:
    """Extract signals that drive weighting and boosting during retrieval."""
    text = (query or "").strip()
    lowered = text.lower()

    numbers = re.findall(r"\b\d+(?:[.,]\d+)?\s*%?\b", text)
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    words = [w for w in re.findall(r"\b[a-z]{2,}\b", lowered) if w not in _STOPWORDS]
    quoted = _QUOTED.findall(text)

    query_type = "general"
    if _FORMULA.search(text):
        query_type = "formula"
    elif _DEFINITION.search(text):
        query_type = "definition"
    elif _NUMERIC.search(text) or years:
        query_type = "numerical"
    elif _DATE.search(text):
        query_type = "date_time"

    return {
        "raw": text,
        "query_type": query_type,
        "numbers": list(dict.fromkeys(numbers + years)),
        "dates_years": list(dict.fromkeys(years)),
        "quoted_phrases": [q.strip() for q in quoted],
        "content_words": list(dict.fromkeys(words)),
    }


# --------------------------------------------------------------------------- #
# Lexical scoring
# --------------------------------------------------------------------------- #
def lexical_score(chunk: str, info: dict) -> float:
    """Score exact-term matches: phrases > numbers > dates > word overlap."""
    lowered = chunk.lower()
    score = 0.0

    for phrase in info.get("quoted_phrases", []):
        if phrase.lower() in lowered:
            score += 3.5

    for number in info.get("numbers", []):
        if re.search(r"(?<![\w.])" + re.escape(number) + r"(?![\w])", chunk):
            score += 3.0

    for year in info.get("dates_years", []):
        if year in lowered:
            score += 2.5

    words = info.get("content_words") or []
    if words:
        matched = [w for w in words if w in lowered]
        score += (len(matched) / len(words)) * 2.0
        if len(matched) == len(words) and len(words) > 1:
            score += 2.0  # full coverage bonus

    return score


def _jaccard(a: str, b: str) -> float:
    """Word-set overlap, used to detect near-duplicate chunks."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# --------------------------------------------------------------------------- #
# Hybrid search
# --------------------------------------------------------------------------- #
def hybrid_search(
    query: str,
    store: VectorStore,
    *,
    top_k: int = 4,
    pool_size: int = 12,
    expand_neighbour: bool = True,
) -> tuple[list[Chunk], dict]:
    """Retrieve the most relevant chunks for ``query``.

    Returns ``(selected_chunks, query_info)``.
    """
    if store is None or not store.chunks or not query.strip():
        return [], analyze_query(query)

    info = analyze_query(query)
    chunks = store.chunks

    # --- 1. Semantic scores for the whole document ------------------- #
    semantic = store.semantic_scores(query)
    if not semantic:
        return [], info

    # --- 2. Lexical scores over a candidate pool --------------------- #
    candidates: set[int] = {
        idx for idx, _ in sorted(semantic.items(), key=lambda kv: kv[1], reverse=True)[:pool_size]
    }
    lexical: dict[int, float] = {i: lexical_score(chunks[i].text, info) for i in candidates}
    lexical = {i: v for i, v in lexical.items() if v > 0}

    # Pull in high-lexical chunks the embedding missed.
    extra_scores: dict[int, float] = {}
    for idx in range(len(chunks)):
        if idx in candidates:
            continue
        score = lexical_score(chunks[idx].text, info)
        if score > 0:
            extra_scores[idx] = score
    for idx in sorted(extra_scores, key=lambda i: extra_scores[i], reverse=True)[:pool_size]:
        candidates.add(idx)
        lexical[idx] = extra_scores[idx]

    # --- 3. Weighted fusion ------------------------------------------ #
    divisor = max(max(lexical.values(), default=0.0), 3.0)
    exact_type = info["query_type"] in {"numerical", "date_time", "definition", "formula"}

    scored: list[Chunk] = []
    for idx in candidates:
        sem = max(0.0, semantic.get(idx, 0.0))
        lex = lexical.get(idx, 0.0) / divisor
        # Exact-lookup questions favour lexical evidence far more.
        combined = (0.35 * sem + 0.65 * lex) if exact_type else (0.60 * sem + 0.40 * lex)

        if info["numbers"] and any(
            re.search(r"(?<![\w.])" + re.escape(n) + r"(?![\w])", chunks[idx].text)
            for n in info["numbers"]
        ):
            combined += 0.35  # boost chunks that literally contain the number

        scored.append(
            Chunk(
                text=chunks[idx].text,
                page=chunks[idx].page,
                index=chunks[idx].index,
                section=chunks[idx].section,
                score=round(combined, 6),
                semantic_score=round(sem, 6),
                lexical_score=round(lex, 6),
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)

    # --- 4. Drop weak / irrelevant chunks ---------------------------- #
    kept = [c for c in scored if c.semantic_score >= 0.18 or c.lexical_score >= 0.12 or c.score >= 0.22]
    if not kept:
        kept = [c for c in scored if c.score >= 0.15][:1] or scored[:1]

    # --- 5. Deduplicate near-identical chunks ------------------------ #
    selected: list[Chunk] = []
    for cand in kept:
        if len(selected) >= top_k:
            break
        if all(_jaccard(cand.text, chosen.text) <= 0.75 for chosen in selected):
            selected.append(cand)

    # --- 6. Expand with the adjacent chunk for context --------------- #
    if expand_neighbour and selected and len(selected) < top_k:
        best = selected[0]
        nxt = best.index + 1
        if nxt < len(chunks) and chunks[nxt].page == best.page:
            if all(c.index != nxt for c in selected):
                selected.append(
                    Chunk(
                        text=chunks[nxt].text,
                        page=chunks[nxt].page,
                        index=chunks[nxt].index,
                        section=chunks[nxt].section,
                        score=round(best.score * 0.75, 6),
                        semantic_score=round(semantic.get(nxt, 0.0), 6),
                    )
                )

    return selected, info


def pages_mentioning(chunks: list[Chunk], term: str) -> list[int]:
    """Distinct page numbers whose text contains ``term``."""
    needle = term.strip().lower()
    if not needle:
        return []
    pages: list[int] = []
    for chunk in chunks:
        if needle in chunk.text.lower() and chunk.page not in pages:
            pages.append(chunk.page)
    return sorted(pages)
