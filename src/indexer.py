"""
indexer.py - Inverted index builder for the COMP3011 Search Engine Tool.

Processes a dictionary of {URL: plain-text} pairs produced by the
crawler and builds an inverted index of the form:

    {
        "word": {
            "https://...": {
                "frequency": <int>,
                "positions": [<int>, ...]
            },
            ...
        },
        ...
    }

The index and per-page word counts are persisted to / restored from a
single JSON file on disk.

TF-IDF scoring is also provided so that search results can be ranked
by relevance rather than just presence.
"""

import json
import logging
import math
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Inner entry: statistics for one word in one page
PageStats = Dict[str, Any]          # {"frequency": int, "positions": [int]}

# Postings list: all pages that contain a word
PostingsList = Dict[str, PageStats]  # {url: PageStats}

# Full inverted index
InvertedIndex = Dict[str, PostingsList]  # {word: PostingsList}


class Indexer:
    """
    Builds, stores, and queries an inverted index.

    Attributes:
        index (InvertedIndex): The main inverted index data structure.
        page_word_counts (dict[str, int]): Total token count per URL,
            used for TF normalisation.
    """

    def __init__(self) -> None:
        self.index: InvertedIndex = {}
        self.page_word_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, pages: Dict[str, str]) -> InvertedIndex:
        """
        Build an inverted index from a mapping of URL -> plain text.

        The index is rebuilt from scratch on each call (previous data
        is discarded). Text is tokenised into lowercase alphabetic words;
        numbers and punctuation are discarded for simplicity.

        Args:
            pages: Dict produced by Crawler.crawl(), mapping URL strings
                   to the plain text extracted from each page.

        Returns:
            The newly built InvertedIndex.
        """
        self.index = {}
        self.page_word_counts = {}

        for url, text in pages.items():
            tokens = self._tokenize(text)
            self.page_word_counts[url] = len(tokens)
            self._index_page(url, tokens)
            logger.info("Indexed %s  (%d tokens)", url, len(tokens))

        logger.info(
            "Index built: %d unique words across %d pages.",
            len(self.index),
            len(self.page_word_counts),
        )
        return self.index

    def save_index(self, filepath: str) -> None:
        """
        Serialise the index to a JSON file.

        Both the inverted index and the per-page word counts are stored
        so that TF-IDF can be recomputed after loading.

        Args:
            filepath: Absolute or relative path to the output file.
        """
        payload = {
            "index": self.index,
            "page_word_counts": self.page_word_counts,
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Index saved to %s", filepath)
        print(f"Index saved to: {filepath}")

    def load_index(self, filepath: str) -> InvertedIndex:
        """
        Load a previously saved index from a JSON file.

        Args:
            filepath: Path to the JSON file created by save_index().

        Returns:
            The loaded InvertedIndex.

        Raises:
            FileNotFoundError: If filepath does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self.index = payload.get("index", {})
        self.page_word_counts = payload.get("page_word_counts", {})

        logger.info("Index loaded from %s", filepath)
        print(f"Index loaded from: {filepath}")
        return self.index

    def compute_tfidf(self, word: str, url: str) -> float:
        """
        Compute the TF-IDF relevance score for a word in a page.

        TF  = (occurrences of word in page) / (total words in page)
        IDF = log( total_pages / pages_containing_word )
        TF-IDF = TF * IDF

        Args:
            word: Lowercase word token.
            url: URL of the page.

        Returns:
            Float TF-IDF score, or 0.0 if the word / URL are not found.
        """
        if word not in self.index:
            return 0.0
        if url not in self.index[word]:
            return 0.0

        frequency: int = self.index[word][url]["frequency"]
        total_words: int = self.page_word_counts.get(url, 1)

        # Guard against division-by-zero on an empty page
        if total_words == 0:
            return 0.0

        tf = frequency / total_words

        total_docs = len(self.page_word_counts)
        docs_with_word = len(self.index[word])

        # Guard: IDF is 0 when the word appears in every document
        idf = math.log(total_docs / docs_with_word) if docs_with_word > 0 else 0.0

        return tf * idf

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """
        Convert raw text to a list of lowercase alphabetic tokens.

        Non-alphabetic characters (digits, punctuation, whitespace) are
        used as delimiters and excluded from the token list.

        Args:
            text: Raw plain-text string.

        Returns:
            Ordered list of lowercase word tokens.
        """
        words = re.findall(r"[a-zA-Z]+", text)
        return [w.lower() for w in words]

    def _index_page(self, url: str, tokens: List[str]) -> None:
        """
        Add all tokens from a page into the inverted index.

        For each token, records its frequency and the list of positions
        (0-based token indices) at which it occurs.

        Args:
            url: The URL of the page being indexed.
            tokens: Ordered list of lowercase tokens from that page.
        """
        for position, word in enumerate(tokens):
            if word not in self.index:
                self.index[word] = {}

            if url not in self.index[word]:
                self.index[word][url] = {"frequency": 0, "positions": []}

            self.index[word][url]["frequency"] += 1
            self.index[word][url]["positions"].append(position)
