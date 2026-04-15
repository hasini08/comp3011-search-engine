"""
search.py - Search logic for the COMP3011 Search Engine Tool.

Provides the SearchEngine class which implements the `print` and `find`
CLI commands against a loaded Indexer instance.

  print <word>         – display the full index entry for one word
  find <word> [...]    – return all pages containing ALL query words,
                         ranked by combined TF-IDF score
"""

import logging
from typing import List, Optional

from indexer import Indexer

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Query interface over an Indexer's inverted index.

    Attributes:
        indexer (Indexer): The Indexer instance whose .index is queried.
    """

    def __init__(self, indexer: Indexer) -> None:
        """
        Initialise the SearchEngine.

        Args:
            indexer: An Indexer object.  The index may be empty until
                     build_index() or load_index() has been called.
        """
        self.indexer = indexer

    # ------------------------------------------------------------------
    # print command
    # ------------------------------------------------------------------

    def print_index(self, word: str) -> None:
        """
        Print the full inverted-index entry for a single word.

        Displays, for every page that contains the word:
          • the page URL
          • how many times the word appears (frequency)
          • the first 10 positional indices (token positions)
          • the TF-IDF relevance score for that page

        Results are sorted by frequency (descending).

        Args:
            word: The word to look up (case-insensitive; leading/trailing
                  whitespace is stripped).
        """
        word = word.strip().lower()

        if not word:
            print("Error: please provide a word.  Usage:  print <word>")
            return

        index = self.indexer.index

        if word not in index:
            print(f"Word '{word}' not found in the index.")
            return

        postings = index[word]
        print(f"\nInverted index entry for '{word}'")
        print(f"Appears in {len(postings)} page(s):\n")

        # Sort pages by descending frequency for readability
        sorted_postings = sorted(
            postings.items(),
            key=lambda item: item[1]["frequency"],
            reverse=True,
        )

        for url, stats in sorted_postings:
            freq: int = stats["frequency"]
            positions: List[int] = stats["positions"]
            tfidf: float = self.indexer.compute_tfidf(word, url)

            # Show only the first 10 positions to keep output manageable
            displayed_positions = positions[:10]
            truncated = len(positions) > 10

            print(f"  URL      : {url}")
            print(f"  Frequency: {freq}")
            print(
                f"  Positions: {displayed_positions}"
                + (" ..." if truncated else "")
            )
            print(f"  TF-IDF   : {tfidf:.6f}")
            print()

    # ------------------------------------------------------------------
    # find command
    # ------------------------------------------------------------------

    def find(self, query: str) -> None:
        """
        Find and display all pages that contain every word in the query.

        Multi-word queries use AND semantics: a page is returned only if
        it contains *all* of the query words.  Results are ranked by
        their combined TF-IDF score (highest first).

        Args:
            query: One or more space-separated words (case-insensitive).
        """
        query = query.strip()
        words: List[str] = query.lower().split()

        if not words:
            print("Error: please provide at least one search term.")
            print("Usage:  find <word> [word2 ...]")
            return

        index = self.indexer.index

        # ----------------------------------------------------------------
        # AND intersection: start with the postings list of the first word
        # then intersect with each subsequent word
        # ----------------------------------------------------------------
        matching_pages: Optional[set] = None

        for word in words:
            if word not in index:
                print(f"No pages found containing '{word}'.")
                return

            pages_for_word = set(index[word].keys())

            if matching_pages is None:
                matching_pages = pages_for_word
            else:
                matching_pages &= pages_for_word

        # matching_pages is None only if words list was empty (handled above)
        if not matching_pages:
            print(
                f"No pages found containing all of: "
                f"{', '.join(repr(w) for w in words)}"
            )
            return

        # ----------------------------------------------------------------
        # Rank results by combined TF-IDF score across all query words
        # ----------------------------------------------------------------
        scored: List[tuple] = []
        for url in matching_pages:
            score = sum(self.indexer.compute_tfidf(w, url) for w in words)
            scored.append((url, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        query_repr = " ".join(words)
        print(f"\nFound {len(scored)} page(s) matching '{query_repr}':\n")

        for rank, (url, score) in enumerate(scored, start=1):
            print(f"  {rank}. {url}")
            for word in words:
                if word in index and url in index[word]:
                    freq = index[word][url]["frequency"]
                    print(f"       '{word}': {freq} occurrence(s)")
            print(f"     Relevance score: {score:.6f}")
            print()
