"""
test_indexer.py - Unit tests for the Indexer module.

Tests cover:
  - Tokenisation (lowercasing, punctuation stripping, edge cases)
  - Index building (structure, frequency counts, position tracking)
  - Multi-page indexing and case insensitivity
  - Save / load round-trip persistence
  - TF-IDF scoring properties
"""

import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexer import Indexer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SIMPLE_PAGES = {
    "https://example.com/p1": "The quick brown fox jumps over the lazy dog",
    "https://example.com/p2": "A quick brown bird sat on the fence",
    "https://example.com/p3": "The dog barked loudly at the bird",
}


def fresh_indexer(pages=None):
    """Return an Indexer pre-built from pages (default: SIMPLE_PAGES)."""
    idx = Indexer()
    idx.build_index(pages if pages is not None else SIMPLE_PAGES)
    return idx


# ---------------------------------------------------------------------------
# TestTokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    """Tests for Indexer._tokenize."""

    def test_lowercase_conversion(self):
        idx = Indexer()
        tokens = idx._tokenize("Hello WORLD")
        assert "hello" in tokens
        assert "world" in tokens
        assert "Hello" not in tokens
        assert "WORLD" not in tokens

    def test_returns_list(self):
        idx = Indexer()
        assert isinstance(idx._tokenize("hello"), list)

    def test_empty_string_returns_empty_list(self):
        idx = Indexer()
        assert idx._tokenize("") == []

    def test_punctuation_stripped(self):
        idx = Indexer()
        tokens = idx._tokenize("hello, world! it's great.")
        assert "," not in tokens
        assert "!" not in tokens
        assert "." not in tokens
        # The apostrophe splits "it's" into "it" and "s"
        assert "hello" in tokens
        assert "world" in tokens

    def test_numbers_excluded(self):
        idx = Indexer()
        tokens = idx._tokenize("page 42 has errors")
        assert "page" in tokens
        assert "has" in tokens
        assert "errors" in tokens
        assert "42" not in tokens

    def test_preserves_order(self):
        idx = Indexer()
        tokens = idx._tokenize("apple banana cherry")
        assert tokens == ["apple", "banana", "cherry"]

    def test_whitespace_only_returns_empty_list(self):
        idx = Indexer()
        assert idx._tokenize("   \t\n  ") == []

    def test_mixed_case_and_punctuation(self):
        idx = Indexer()
        tokens = idx._tokenize("Hello, World! GREAT.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "great" in tokens

    def test_hyphenated_words_split(self):
        """Hyphens are treated as delimiters, splitting compound words."""
        idx = Indexer()
        tokens = idx._tokenize("well-known fact")
        assert "well" in tokens
        assert "known" in tokens


# ---------------------------------------------------------------------------
# TestBuildIndex
# ---------------------------------------------------------------------------


class TestBuildIndex:
    """Tests for Indexer.build_index."""

    def test_returns_dict(self):
        idx = fresh_indexer()
        assert isinstance(idx.index, dict)

    def test_indexed_words_present(self):
        idx = fresh_indexer()
        assert "quick" in idx.index
        assert "dog" in idx.index
        assert "fox" in idx.index

    def test_index_entry_has_frequency(self):
        idx = fresh_indexer()
        entry = idx.index["fox"]["https://example.com/p1"]
        assert "frequency" in entry
        assert isinstance(entry["frequency"], int)

    def test_index_entry_has_positions(self):
        idx = fresh_indexer()
        entry = idx.index["fox"]["https://example.com/p1"]
        assert "positions" in entry
        assert isinstance(entry["positions"], list)

    def test_frequency_single_occurrence(self):
        idx = Indexer()
        idx.build_index({"https://example.com": "cat sat on mat"})
        assert idx.index["cat"]["https://example.com"]["frequency"] == 1

    def test_frequency_multiple_occurrences(self):
        idx = Indexer()
        idx.build_index({"https://example.com": "the cat and the dog and the bird"})
        assert idx.index["the"]["https://example.com"]["frequency"] == 3
        assert idx.index["and"]["https://example.com"]["frequency"] == 2

    def test_position_tracking_single(self):
        idx = Indexer()
        idx.build_index({"https://example.com": "hello world"})
        assert idx.index["hello"]["https://example.com"]["positions"] == [0]
        assert idx.index["world"]["https://example.com"]["positions"] == [1]

    def test_position_tracking_multiple(self):
        idx = Indexer()
        idx.build_index({"https://example.com": "hello world hello"})
        positions = idx.index["hello"]["https://example.com"]["positions"]
        assert 0 in positions
        assert 2 in positions

    def test_case_insensitive_merging(self):
        idx = Indexer()
        idx.build_index({"https://example.com": "Good GOOD good"})
        freq = idx.index["good"]["https://example.com"]["frequency"]
        assert freq == 3

    def test_word_across_multiple_pages(self):
        idx = fresh_indexer()
        # "quick" appears in p1 and p2
        assert "https://example.com/p1" in idx.index["quick"]
        assert "https://example.com/p2" in idx.index["quick"]

    def test_word_not_in_other_pages(self):
        idx = fresh_indexer()
        # "fox" only in p1
        assert "https://example.com/p2" not in idx.index.get("fox", {})
        assert "https://example.com/p3" not in idx.index.get("fox", {})

    def test_empty_pages_dict(self):
        idx = Indexer()
        result = idx.build_index({})
        assert result == {}

    def test_page_with_empty_text(self):
        idx = Indexer()
        result = idx.build_index({"https://example.com": ""})
        assert result == {}

    def test_page_word_counts_populated(self):
        idx = fresh_indexer()
        for url in SIMPLE_PAGES:
            assert url in idx.page_word_counts
            assert idx.page_word_counts[url] > 0

    def test_rebuild_clears_previous_index(self):
        idx = Indexer()
        idx.build_index({"https://a.com": "hello world"})
        idx.build_index({"https://b.com": "foo bar"})
        # "hello" should NOT be present after rebuild with different pages
        assert "hello" not in idx.index
        assert "foo" in idx.index

    def test_special_characters_only(self):
        idx = Indexer()
        result = idx.build_index({"https://example.com": "!!! ??? ---"})
        assert result == {}


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------


class TestPersistence:
    """Tests for Indexer.save_index and Indexer.load_index."""

    def test_save_produces_valid_json(self):
        idx = fresh_indexer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            idx.save_index(path)
            with open(path, "r") as f:
                data = json.load(f)
            assert "index" in data
            assert "page_word_counts" in data
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_index(self):
        idx = fresh_indexer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            idx.save_index(path)
            idx2 = Indexer()
            idx2.load_index(path)
            assert idx2.index == idx.index
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_page_word_counts(self):
        idx = fresh_indexer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            idx.save_index(path)
            idx2 = Indexer()
            idx2.load_index(path)
            assert idx2.page_word_counts == idx.page_word_counts
        finally:
            os.unlink(path)

    def test_load_nonexistent_file_raises(self):
        idx = Indexer()
        with pytest.raises(FileNotFoundError):
            idx.load_index("/nonexistent/path/does_not_exist.json")

    def test_load_returns_index(self):
        idx = fresh_indexer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            idx.save_index(path)
            idx2 = Indexer()
            result = idx2.load_index(path)
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_save_overwrites_previous_file(self):
        idx1 = Indexer()
        idx1.build_index({"https://a.com": "hello world"})
        idx2 = Indexer()
        idx2.build_index({"https://b.com": "foo bar"})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            idx1.save_index(path)
            idx2.save_index(path)  # overwrite

            idx3 = Indexer()
            idx3.load_index(path)

            assert "foo" in idx3.index
            assert "hello" not in idx3.index
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestTFIDF
# ---------------------------------------------------------------------------


class TestTFIDF:
    """Tests for Indexer.compute_tfidf."""

    def test_missing_word_returns_zero(self):
        idx = fresh_indexer()
        assert idx.compute_tfidf("zzznonsense", "https://example.com/p1") == 0.0

    def test_missing_url_for_word_returns_zero(self):
        idx = fresh_indexer()
        # "fox" only in p1; querying p2 should return 0
        assert idx.compute_tfidf("fox", "https://example.com/p2") == 0.0

    def test_valid_score_is_nonnegative_float(self):
        idx = fresh_indexer()
        score = idx.compute_tfidf("quick", "https://example.com/p1")
        assert isinstance(score, float)
        assert score >= 0.0

    def test_rare_word_scores_higher_than_common_word(self):
        """
        A word that appears in fewer pages should have a higher IDF and
        therefore, for the same TF, a higher TF-IDF score than a very
        common word.
        """
        idx = fresh_indexer()
        # "the" appears in p1 and p3 (common); "fox" only in p1 (rare)
        common_score = idx.compute_tfidf("the", "https://example.com/p1")
        rare_score = idx.compute_tfidf("fox", "https://example.com/p1")
        assert rare_score >= common_score

    def test_word_in_all_pages_has_zero_idf(self):
        """IDF = log(N/N) = log(1) = 0, so TF-IDF should be 0."""
        pages = {
            "https://a.com": "common word here",
            "https://b.com": "common phrase",
            "https://c.com": "another common thing",
        }
        idx = Indexer()
        idx.build_index(pages)
        # "common" appears in all 3 pages → IDF = 0
        score = idx.compute_tfidf("common", "https://a.com")
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_empty_page_word_count_does_not_crash(self):
        """Edge case: if a page somehow has 0 tokens, TF-IDF should be 0."""
        idx = Indexer()
        idx.build_index({"https://example.com": "hello"})
        # Manually corrupt the count to simulate a degenerate case
        idx.page_word_counts["https://example.com"] = 0
        score = idx.compute_tfidf("hello", "https://example.com")
        assert score == 0.0
