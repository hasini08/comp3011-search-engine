"""
test_search.py - Unit tests for the SearchEngine module.

Tests cover:
  - print_index: existing word, missing word, case insensitivity, edge cases
  - find: single word, multi-word AND, missing word, case, ranking, edge cases
  - Edge cases: empty index, empty query, whitespace-only query
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexer import Indexer
from search import SearchEngine


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_PAGES = {
    "https://example.com/p1": "The quick brown fox jumps over the lazy dog",
    "https://example.com/p2": "A quick brown bird sat on the fence and the dog",
    "https://example.com/p3": "The dog barked loudly at the bird in the morning",
    "https://example.com/p4": "Life is beautiful and full of wonder and joy",
}


def make_engine(pages=None) -> SearchEngine:
    """Build a SearchEngine from pages (default: SAMPLE_PAGES)."""
    idx = Indexer()
    idx.build_index(pages if pages is not None else SAMPLE_PAGES)
    return SearchEngine(idx)


# ---------------------------------------------------------------------------
# TestPrintIndex
# ---------------------------------------------------------------------------


class TestPrintIndex:
    """Tests for SearchEngine.print_index."""

    def test_found_word_shows_url(self, capsys):
        engine = make_engine()
        engine.print_index("fox")
        out = capsys.readouterr().out
        assert "p1" in out

    def test_missing_word_reports_not_found(self, capsys):
        engine = make_engine()
        engine.print_index("xyznonexistent")
        out = capsys.readouterr().out
        assert "not found" in out.lower()

    def test_empty_string_reports_error(self, capsys):
        engine = make_engine()
        engine.print_index("")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "please" in out.lower()

    def test_whitespace_only_reports_error_or_not_found(self, capsys):
        engine = make_engine()
        engine.print_index("   ")
        out = capsys.readouterr().out
        # After strip the word is empty → error, or "not found" either is fine
        assert len(out.strip()) > 0

    def test_case_insensitive_uppercase(self, capsys):
        engine = make_engine()
        engine.print_index("FOX")
        out = capsys.readouterr().out
        assert "not found" not in out.lower()
        assert "p1" in out

    def test_case_insensitive_mixed(self, capsys):
        engine = make_engine()
        engine.print_index("QuIcK")
        out = capsys.readouterr().out
        assert "not found" not in out.lower()

    def test_frequency_displayed(self, capsys):
        engine = make_engine()
        engine.print_index("dog")
        out = capsys.readouterr().out
        assert "frequency" in out.lower() or any(c.isdigit() for c in out)

    def test_positions_displayed(self, capsys):
        engine = make_engine()
        engine.print_index("dog")
        out = capsys.readouterr().out
        assert "position" in out.lower()

    def test_tfidf_displayed(self, capsys):
        engine = make_engine()
        engine.print_index("fox")
        out = capsys.readouterr().out
        assert "tfidf" in out.lower() or "tf-idf" in out.lower() or "score" in out.lower()

    def test_word_in_multiple_pages_shows_all(self, capsys):
        engine = make_engine()
        engine.print_index("dog")
        out = capsys.readouterr().out
        # "dog" is in p1, p2, and p3
        assert "p1" in out
        assert "p2" in out
        assert "p3" in out

    def test_word_with_leading_trailing_spaces(self, capsys):
        engine = make_engine()
        engine.print_index("  fox  ")
        out = capsys.readouterr().out
        assert "not found" not in out.lower()

    def test_empty_index_reports_not_found(self, capsys):
        idx = Indexer()
        engine = SearchEngine(idx)
        engine.print_index("anything")
        out = capsys.readouterr().out
        assert "not found" in out.lower()


# ---------------------------------------------------------------------------
# TestFind
# ---------------------------------------------------------------------------


class TestFind:
    """Tests for SearchEngine.find."""

    # ---- Single-word searches ----

    def test_single_word_found(self, capsys):
        engine = make_engine()
        engine.find("fox")
        out = capsys.readouterr().out
        assert "p1" in out

    def test_single_word_not_found(self, capsys):
        engine = make_engine()
        engine.find("xyznonexistent")
        out = capsys.readouterr().out
        assert "no pages" in out.lower() or "not found" in out.lower()

    def test_single_word_multiple_pages(self, capsys):
        engine = make_engine()
        engine.find("dog")
        out = capsys.readouterr().out
        # "dog" appears in p1, p2, p3
        assert "p1" in out
        assert "p2" in out
        assert "p3" in out

    def test_unique_word_only_one_page(self, capsys):
        engine = make_engine()
        engine.find("wonder")
        out = capsys.readouterr().out
        assert "p4" in out
        assert "p1" not in out

    # ---- Multi-word AND searches ----

    def test_multi_word_and_both_present(self, capsys):
        engine = make_engine()
        engine.find("quick brown")
        out = capsys.readouterr().out
        # Both "quick" and "brown" are in p1 and p2
        assert "p1" in out
        assert "p2" in out

    def test_multi_word_and_excludes_partial_matches(self, capsys):
        engine = make_engine()
        engine.find("quick brown")
        out = capsys.readouterr().out
        # p3 has neither "quick" nor "brown"
        assert "p3" not in out

    def test_multi_word_no_page_has_all(self, capsys):
        engine = make_engine()
        engine.find("fox wonder")  # fox=p1 only, wonder=p4 only → no overlap
        out = capsys.readouterr().out
        assert "no pages" in out.lower() or "not found" in out.lower()

    def test_multi_word_one_missing_from_index(self, capsys):
        engine = make_engine()
        engine.find("fox zzzmissing")
        out = capsys.readouterr().out
        assert "no pages" in out.lower() or "not found" in out.lower()

    # ---- Case insensitivity ----

    def test_find_uppercase_word(self, capsys):
        engine = make_engine()
        engine.find("FOX")
        out = capsys.readouterr().out
        assert "p1" in out

    def test_find_mixed_case_query(self, capsys):
        engine = make_engine()
        engine.find("QuIcK BrOwN")
        out = capsys.readouterr().out
        assert "p1" in out

    # ---- Edge cases ----

    def test_empty_query_reports_error(self, capsys):
        engine = make_engine()
        engine.find("")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "please" in out.lower()

    def test_whitespace_only_query_reports_error(self, capsys):
        engine = make_engine()
        engine.find("   ")
        out = capsys.readouterr().out
        assert "error" in out.lower() or "please" in out.lower()

    def test_find_shows_result_count(self, capsys):
        engine = make_engine()
        engine.find("dog")
        out = capsys.readouterr().out
        # Output should mention how many pages were found (a digit)
        assert any(c.isdigit() for c in out)

    def test_find_shows_occurrence_count(self, capsys):
        engine = make_engine()
        engine.find("dog")
        out = capsys.readouterr().out
        assert "occurrence" in out.lower()

    def test_find_with_leading_trailing_spaces(self, capsys):
        engine = make_engine()
        engine.find("  fox  ")
        out = capsys.readouterr().out
        assert "p1" in out

    def test_empty_index_find_reports_not_found(self, capsys):
        idx = Indexer()
        engine = SearchEngine(idx)
        engine.find("anything")
        out = capsys.readouterr().out
        assert "no pages" in out.lower() or "not found" in out.lower()

    # ---- Ranking ----

    def test_results_are_ranked(self, capsys):
        """Pages with higher TF-IDF should appear earlier in results."""
        # Three pages: dense has highest TF for "rare", other two have lower TF.
        # A third page with no "rare" ensures IDF > 0 so scores differ.
        pages = {
            "https://example.com/dense": "rare rare rare rare rare word appears many times",
            "https://example.com/common": "rare word appears here once",
            "https://example.com/other": "nothing relevant here at all",
        }
        engine = make_engine(pages)
        engine.find("rare")
        out = capsys.readouterr().out
        # The dense page has highest TF-IDF and should appear first
        pos_dense = out.find("dense")
        pos_common = out.find("common")
        assert pos_dense < pos_common

    def test_find_three_word_query(self, capsys):
        engine = make_engine()
        engine.find("quick brown dog")
        out = capsys.readouterr().out
        # Only p2 contains all three: "quick", "brown", "dog"
        assert "p2" in out
        # p1 has quick/brown/dog too
        assert "p1" in out
        # p3 has dog but not quick/brown
        assert "p3" not in out
