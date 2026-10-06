"""Tests for chunking, embeddings, the FAISS store and hybrid ranking."""

from __future__ import annotations

import pytest

from summaryai.retrieval import (
    VectorStore,
    analyze_query,
    chunk_document,
    chunk_stats,
    find_page_text,
    get_embedder,
    hybrid_search,
    l2_normalize,
    lexical_score,
    locate_phrase,
    pages_mentioning,
)
from summaryai.retrieval.embeddings import HashEmbedder

from conftest import SAMPLE_PAGES


# --------------------------------------------------------------------------- #
# Chunker
# --------------------------------------------------------------------------- #
class TestChunker:
    def test_produces_chunks_per_page(self):
        chunks = chunk_document(SAMPLE_PAGES, max_chars=400)
        assert len(chunks) >= 4
        assert {c.page for c in chunks} == {1, 2, 3, 4}

    def test_page_numbers_are_one_based(self):
        chunks = chunk_document(SAMPLE_PAGES)
        assert min(c.page for c in chunks) == 1
        assert max(c.page for c in chunks) == len(SAMPLE_PAGES)

    def test_chunks_respect_max_chars(self):
        chunks = chunk_document(SAMPLE_PAGES, max_chars=300)
        # A single oversized sentence may exceed the budget by design.
        assert all(len(c.text) <= 500 for c in chunks)

    def test_small_pages_are_not_split(self):
        assert len(chunk_document(["Short page."], max_chars=900)) == 1

    def test_empty_pages_are_skipped(self):
        chunks = chunk_document(["", "Real content.", "   "])
        assert len(chunks) == 1
        assert chunks[0].page == 2

    def test_indices_are_sequential(self):
        chunks = chunk_document(SAMPLE_PAGES, max_chars=200)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_rejects_invalid_max_chars(self):
        with pytest.raises(ValueError):
            chunk_document(["text"], max_chars=0)

    def test_overlap_larger_than_chunk_is_clamped(self):
        assert len(chunk_document(SAMPLE_PAGES, max_chars=100, overlap=999)) > 0

    def test_heading_is_attached_to_chunks(self):
        assert any(c.section for c in chunk_document(SAMPLE_PAGES, max_chars=300))

    def test_chunk_stats(self):
        stats = chunk_stats(chunk_document(SAMPLE_PAGES))
        assert stats["total_chunks"] > 0
        assert stats["avg_chars"] > 0

    def test_chunk_stats_on_empty(self):
        assert chunk_stats([])["total_chunks"] == 0


# --------------------------------------------------------------------------- #
# Page helpers
# --------------------------------------------------------------------------- #
class TestPageHelpers:
    def test_find_page_text(self, sample_document):
        assert "motherboard" in find_page_text(sample_document.pages, 1).lower()
        assert find_page_text(sample_document.pages, 99) == ""

    def test_locate_phrase(self, sample_document):
        assert locate_phrase(sample_document.pages, "CUDA cores") == 2
        assert locate_phrase(sample_document.pages, "nonexistent topic") is None


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class TestEmbeddings:
    def test_l2_normalize_gives_unit_rows(self):
        import numpy as np

        matrix = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
        assert pytest.approx(float((matrix[0] ** 2).sum()), abs=0.01) == 1.0
        assert (matrix[1] == 0).all()  # zero rows are left untouched

    def test_hash_embedder_is_deterministic(self):
        assert (HashEmbedder().encode(["motherboard cpu"]) == HashEmbedder().encode(["motherboard cpu"])).all()

    def test_hash_embedder_shape(self):
        assert HashEmbedder(64).encode(["one", "two", "three"]).shape == (3, 64)

    def test_shared_words_increase_similarity(self):
        embedder = HashEmbedder()
        base = l2_normalize(embedder.encode(["cpu motherboard ram"]))
        related = l2_normalize(embedder.encode(["cpu motherboard gpu"]))
        unrelated = l2_normalize(embedder.encode(["cooking pasta recipe"]))
        assert float((base * related).sum()) > float((base * unrelated).sum())

    def test_empty_input(self):
        assert HashEmbedder().encode([]).shape[0] == 0

    def test_get_embedder_is_cached(self):
        assert get_embedder(lightweight=True) is get_embedder(lightweight=True)


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
class TestVectorStore:
    @pytest.fixture
    def store(self, hash_embedder):
        return VectorStore(hash_embedder).build(chunk_document(SAMPLE_PAGES, max_chars=300))

    def test_build_populates_index(self, store):
        assert store.size > 0
        assert store.index is not None

    def test_search_returns_sorted_scores(self, store):
        results = store.search("GPU graphics rendering", top_k=3)
        assert 1 <= len(results) <= 3
        assert results == sorted(results, key=lambda kv: kv[1], reverse=True)

    def test_search_on_empty_store(self, hash_embedder):
        assert VectorStore(hash_embedder).search("anything") == []

    def test_search_blank_query(self, store):
        assert store.search("   ") == []

    def test_semantic_scores_cover_every_chunk(self, store):
        assert len(store.semantic_scores("cpu")) == store.size

    def test_rebuild_replaces_contents(self, store):
        store.build(chunk_document(["one small page"]))
        assert store.size == 1


# --------------------------------------------------------------------------- #
# Query analysis
# --------------------------------------------------------------------------- #
class TestQueryAnalysis:
    def test_detects_numbers(self):
        assert "2024" in analyze_query("What is the clock speed in 2024?")["numbers"]

    def test_detects_definition_questions(self):
        assert analyze_query("What is a motherboard?")["query_type"] == "definition"

    def test_detects_formula_questions(self):
        assert analyze_query("calculate the value of x = 5")["query_type"] == "formula"

    def test_detects_quoted_phrases(self):
        assert analyze_query('what does "thermal paste" do')["quoted_phrases"] == ["thermal paste"]

    def test_stopwords_removed(self):
        assert "the" not in analyze_query("What is the motherboard?")["content_words"]


# --------------------------------------------------------------------------- #
# Lexical scoring and hybrid search
# --------------------------------------------------------------------------- #
class TestHybridSearch:
    @pytest.fixture
    def store(self, hash_embedder):
        return VectorStore(hash_embedder).build(chunk_document(SAMPLE_PAGES, max_chars=300))

    def test_exact_number_boosts_matching_chunk(self):
        info = analyze_query("What clock speed does the CPU run at 3.5 GHz?")
        assert lexical_score("The CPU executes instructions at 3.5 GHz.", info) > lexical_score(
            "The GPU renders graphics using CUDA cores.", info
        )

    def test_finds_the_correct_page(self, store):
        chunks, _ = hybrid_search("How does the GPU render graphics?", store, top_k=3)
        assert any(c.page == 2 for c in chunks)

    def test_finds_power_supply_page(self, store):
        chunks, _ = hybrid_search("what wattage power supply for gaming build", store, top_k=3)
        assert any(c.page == 4 for c in chunks)

    def test_respects_top_k(self, store):
        chunks, _ = hybrid_search("computer hardware", store, top_k=2)
        assert len(chunks) <= 3  # top_k plus possible neighbour expansion

    def test_blank_query_returns_nothing(self, store):
        chunks, info = hybrid_search("  ", store)
        assert chunks == []
        assert info["query_type"] == "general"

    def test_empty_store_returns_nothing(self, hash_embedder):
        assert hybrid_search("anything", VectorStore(hash_embedder))[0] == []

    def test_scores_are_populated(self, store):
        chunks, _ = hybrid_search("CPU instructions", store)
        assert all(c.score >= 0 for c in chunks)
        assert any(c.semantic_score > 0 for c in chunks)

    def test_deduplicates_near_identical_chunks(self, store):
        chunks, _ = hybrid_search("motherboard", store, top_k=4)
        assert len({c.text for c in chunks}) == len(chunks)

    def test_pages_mentioning(self, store):
        chunks, _ = hybrid_search("cooling system", store)
        assert pages_mentioning(chunks, "fans") == [3]
