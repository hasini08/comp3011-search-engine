"""
crawler.py - Web crawler for the COMP3011 Search Engine Tool.

Crawls all pages of a target website, respecting a configurable
politeness window between successive HTTP requests. Extracts and
returns the visible text content of each page visited.
"""

import logging
import time
from typing import Dict, List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class Crawler:
    """
    A breadth-first web crawler that stays within a single domain.

    Attributes:
        base_url (str): The starting URL and domain boundary.
        politeness_window (float): Seconds to wait between requests.
        visited (set[str]): URLs already fetched.
        pages (dict[str, str]): Mapping of URL -> extracted plain text.
    """

    def __init__(self, base_url: str, politeness_window: float = 6.0) -> None:
        """
        Initialise the crawler.

        Args:
            base_url: The URL to start crawling from.
            politeness_window: Minimum delay (seconds) between requests.
        """
        self.base_url: str = base_url
        self.politeness_window: float = politeness_window
        self.visited: set = set()
        self.pages: Dict[str, str] = {}

        # Re-use a single TCP session for keep-alive and header defaults
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "COMP3011-SearchEngine/1.0 (educational crawler)"}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self) -> Dict[str, str]:
        """
        Crawl the website starting from base_url.

        Performs a depth-first traversal of all internal links, pausing
        for at least `politeness_window` seconds between each request.

        Returns:
            A dict mapping each visited URL to its plain-text content.
        """
        logger.info("Starting crawl of %s", self.base_url)
        self._crawl_page(self.base_url)
        logger.info("Crawl complete. %d pages fetched.", len(self.pages))
        return self.pages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_url(self, url: str) -> str:
        """Strip trailing slash from a URL path for consistent deduplication."""
        parsed = urlparse(url)
        # Remove trailing slash only when the path is more than just "/"
        path = parsed.path.rstrip("/") or "/"
        return parsed._replace(path=path).geturl()

    def _crawl_page(self, url: str) -> None:
        """
        Fetch a single page, store its text, and recurse into its links.

        Args:
            url: Absolute URL of the page to fetch.
        """
        url = self._normalise_url(url)
        if url in self.visited:
            return

        self.visited.add(url)

        html = self._fetch(url)
        if html is None:
            return  # Network error – skip silently (already logged)

        soup = BeautifulSoup(html, "html.parser")
        text = self._extract_text(soup)
        self.pages[url] = text
        logger.info("Crawled: %s  (%d chars)", url, len(text))

        links = self._extract_links(soup, url)
        for link in links:
            if link not in self.visited:
                time.sleep(self.politeness_window)
                self._crawl_page(link)

    def _fetch(self, url: str) -> str | None:
        """
        Perform an HTTP GET request and return the response body.

        Args:
            url: URL to fetch.

        Returns:
            HTML string on success, or None on any network / HTTP error.
        """
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Connection error fetching %s: %s", url, exc)
        except requests.exceptions.Timeout:
            logger.warning("Request timed out: %s", url)
        except requests.exceptions.RequestException as exc:
            logger.warning("Request failed for %s: %s", url, exc)
        return None

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract visible plain text from a parsed HTML page.

        Removes <script>, <style>, <nav>, and <footer> elements before
        extracting text so that boilerplate is excluded from the index.

        Args:
            soup: Parsed BeautifulSoup object.

        Returns:
            Cleaned plain-text string.
        """
        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "head"]):
            tag.decompose()

        # join tokens with a space so words don't run together
        return soup.get_text(separator=" ", strip=True)

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """
        Extract all internal links from a page, excluding already-visited ones.

        Links are resolved to absolute URLs and filtered to the same
        domain as base_url. Fragment identifiers are stripped.

        Args:
            soup: Parsed BeautifulSoup object.
            current_url: The URL of the page being parsed (used as base
                         for resolving relative hrefs).

        Returns:
            List of unique, absolute, same-domain URLs not yet visited.
        """
        base_domain = urlparse(self.base_url).netloc
        seen_in_batch: set = set()
        links: List[str] = []

        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"].strip()

            # Skip mailto, javascript, and empty hrefs
            if not href or href.startswith(("mailto:", "javascript:", "#")):
                continue

            absolute_url = urljoin(current_url, href)
            parsed = urlparse(absolute_url)

            # Only follow http(s) links within the same domain
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc != base_domain:
                continue

            # Strip fragment to avoid treating /page#section as new URL
            clean_url = self._normalise_url(parsed._replace(fragment="").geturl())

            if clean_url not in self.visited and clean_url not in seen_in_batch:
                seen_in_batch.add(clean_url)
                links.append(clean_url)

        return links
