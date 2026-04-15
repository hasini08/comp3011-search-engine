"""
main.py - Command-line interface for the COMP3011 Search Engine Tool.

Run this file directly to start the interactive shell:

    python main.py

Supported commands
------------------
build           Crawl quotes.toscrape.com, build the index, and save it.
load            Load a previously built index from disk.
print <word>    Print the inverted-index entry for <word>.
find <query>    Find all pages containing every word in <query>.
help            Show this help text.
quit / exit     Exit the shell.
"""

import logging
import os
import sys

# -----------------------------------------------------------------------
# Add the src/ directory to sys.path so that imports work whether this
# script is run from the project root OR from inside src/.
# -----------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from crawler import Crawler
from indexer import Indexer
from search import SearchEngine

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

TARGET_URL = "https://quotes.toscrape.com/"
POLITENESS_WINDOW = 6.0  # seconds between requests (as required)

# Index is stored in  <project_root>/data/index.json
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
INDEX_FILE = os.path.join(_PROJECT_ROOT, "data", "index.json")

logging.basicConfig(
    level=logging.WARNING,  # suppress INFO noise during normal CLI use
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# -----------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------


def cmd_build(indexer: Indexer) -> None:
    """
    Crawl the target website, build the inverted index, and save it.

    This command makes live HTTP requests to quotes.toscrape.com and
    will take several minutes due to the mandatory 6-second politeness
    window between requests.
    """
    print(f"Target: {TARGET_URL}")
    print(
        f"Politeness window: {POLITENESS_WINDOW}s  "
        "(this will take a few minutes – please wait)\n"
    )

    crawler = Crawler(TARGET_URL, politeness_window=POLITENESS_WINDOW)
    pages = crawler.crawl()

    if not pages:
        print("No pages were retrieved. Check your network connection.")
        return

    print(f"\n{len(pages)} page(s) crawled. Building index …")
    indexer.build_index(pages)

    # Make sure the data/ directory exists before writing
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    indexer.save_index(INDEX_FILE)

    print(
        f"\nDone.  Index contains {len(indexer.index)} unique words "
        f"across {len(indexer.page_word_counts)} page(s)."
    )


def cmd_load(indexer: Indexer, search_engine: SearchEngine) -> None:
    """
    Load a previously built index from disk.

    Requires that `build` has been run at least once to create the
    index file.
    """
    if not os.path.isfile(INDEX_FILE):
        print(
            f"Index file not found at:\n  {INDEX_FILE}\n"
            "Run 'build' first to create it."
        )
        return

    indexer.load_index(INDEX_FILE)
    search_engine.indexer = indexer  # Ensure SearchEngine sees new data

    print(
        f"Index loaded.  "
        f"{len(indexer.index)} unique words, "
        f"{len(indexer.page_word_counts)} page(s)."
    )


def print_help() -> None:
    """Print the help text for the CLI."""
    help_text = """
Commands
--------
  build              Crawl the website, build the index, and save to disk.
  load               Load the saved index from disk.
  print <word>       Print the inverted-index entry for a word.
  find <query>       Find pages containing all words in the query.
  help               Show this message.
  quit | exit        Exit the shell.

Examples
--------
  > build
  > load
  > print nonsense
  > find indifference
  > find good friends
"""
    print(help_text)


# -----------------------------------------------------------------------
# Main REPL
# -----------------------------------------------------------------------


def main() -> None:
    """Start the interactive search engine shell."""
    indexer = Indexer()
    search_engine = SearchEngine(indexer)

    print("=" * 50)
    print("  COMP3011 Search Engine Tool")
    print("=" * 50)
    print("Type 'help' for a list of commands.\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        # Split into command and the rest of the line
        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if command == "build":
            cmd_build(indexer)

        elif command == "load":
            cmd_load(indexer, search_engine)

        elif command == "print":
            if not args:
                print("Usage:  print <word>")
            else:
                search_engine.print_index(args)

        elif command == "find":
            if not args:
                print("Usage:  find <word> [word2 ...]")
            else:
                search_engine.find(args)

        elif command in ("help", "?", "h"):
            print_help()

        elif command in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        else:
            print(
                f"Unknown command: '{command}'.  "
                "Type 'help' to see available commands."
            )


if __name__ == "__main__":
    main()
