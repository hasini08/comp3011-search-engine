# COMP3011 Search Engine Tool

A command-line search engine that crawls [quotes.toscrape.com](https://quotes.toscrape.com/), builds an inverted index of every word on the site, and lets you search across all pages.

---

## Project Overview

The tool is split into three focused modules:

| Module | Purpose |
|---|---|
| `crawler.py` | Crawls all pages within the target domain, respecting a 6-second politeness window |
| `indexer.py` | Builds an inverted index storing word frequency and positional data; saves/loads from JSON |
| `search.py` | Queries the index with `print` and `find` commands, ranking results by TF-IDF |

Results from `find` are ranked by **TF-IDF** (Term Frequency × Inverse Document Frequency), so pages where the search terms are most relevant appear first.

---

## Installation

### Prerequisites

- Python 3.10 or later
- `pip`

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-name>

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

Start the interactive shell by running `main.py` from inside the `src/` directory:

```bash
cd src
python main.py
```

You will see a `>` prompt. Type any of the commands below.

### Commands

#### `build`

Crawls the entire target website, builds the inverted index, and saves it to `data/index.json`.

```
> build
```

> **Note:** The crawler respects a mandatory 6-second politeness window between requests, so this command takes several minutes to complete.

#### `load`

Loads a previously built index from `data/index.json`. Run this after `build` to avoid re-crawling on every session.

```
> load
```

#### `print <word>`

Prints the full inverted-index entry for a single word: every page it appears in, how many times, the token positions, and its TF-IDF score on each page.

```
> print nonsense
> print indifference
```

#### `find <query>`

Finds all pages that contain **every** word in the query (AND semantics). Results are ranked by combined TF-IDF score, highest first.

```
> find indifference
> find good friends
> find life beautiful wonder
```

#### `help`

Shows a summary of all commands.

```
> help
```

#### `quit` / `exit`

Exit the shell.

```
> quit
```

---

## Example Session

```
> build
Target: https://quotes.toscrape.com/
Politeness window: 6.0s  (this will take a few minutes – please wait)

Crawled: https://quotes.toscrape.com/  (3421 chars)
Crawled: https://quotes.toscrape.com/page/2/  (3298 chars)
...
11 page(s) crawled. Building index ...
Index saved to: data/index.json

Done.  Index contains 1842 unique words across 11 page(s).

> load
Index loaded from: data/index.json
Index loaded.  1842 unique words, 11 page(s).

> print nonsense
Inverted index entry for 'nonsense'
Appears in 1 page(s):

  URL      : https://quotes.toscrape.com/page/3/
  Frequency: 1
  Positions: [42]
  TF-IDF   : 0.003512

> find good friends
Found 2 page(s) matching 'good friends':

  1. https://quotes.toscrape.com/page/4/
       'good': 2 occurrence(s)
       'friends': 1 occurrence(s)
     Relevance score: 0.008241

  2. https://quotes.toscrape.com/page/7/
       'good': 1 occurrence(s)
       'friends': 1 occurrence(s)
     Relevance score: 0.005103
```

---

## Running the Tests

```bash
# From the project root
pytest tests/ -v
```

To see test coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test file overview

| File | What it tests |
|---|---|
| `tests/test_crawler.py` | Initialisation, text/link extraction, crawl behaviour, error handling |
| `tests/test_indexer.py` | Tokenisation, index building, persistence, TF-IDF scoring |
| `tests/test_search.py` | `print_index`, `find` (single/multi-word, AND logic, ranking, edge cases) |

All crawler tests use `unittest.mock` to mock HTTP requests — no live network access is needed.

---

## Repository Structure

```
.
├── src/
│   ├── crawler.py        # Web crawler
│   ├── indexer.py        # Inverted index builder and persistence
│   ├── search.py         # Search logic (print + find commands)
│   └── main.py           # CLI entry point
├── tests/
│   ├── test_crawler.py
│   ├── test_indexer.py
│   └── test_search.py
├── data/
│   └── index.json        # Generated after running `build`
├── requirements.txt
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `requests` | HTTP client for fetching web pages |
| `beautifulsoup4` | HTML parsing and text extraction |
| `pytest` | Test runner |
| `pytest-cov` | Code coverage reporting |
