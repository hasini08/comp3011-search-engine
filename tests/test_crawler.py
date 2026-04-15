"""
test_crawler.py - Unit tests for the Crawler module.

Tests cover:
  - Initialisation defaults and custom values
  - Text extraction and noise removal
  - Link extraction, filtering, and deduplication
  - Crawling behaviour: politeness window, cycle avoidance, error handling
"""

import sys
import os

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import requests as req
from bs4 import BeautifulSoup
from unittest.mock import MagicMock, patch, call

from crawler import Crawler


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_mock_response(html: str, status_code: int = 200) -> MagicMock:
    """Return a mock requests.Response with the given HTML body."""
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.status_code = status_code
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError(
            f"HTTP {status_code}"
        )
    else:
        mock_resp.raise_for_status = MagicMock()
    return mock_resp


HTML_PAGE_WITH_LINKS = """
<html>
<head><title>Home</title></head>
<body>
  <p>Welcome to the home page.</p>
  <a href="/about">About</a>
  <a href="/contact">Contact</a>
</body>
</html>
"""

HTML_PAGE_NO_LINKS = """
<html>
<body>
  <p>This page has no outgoing links.</p>
</body>
</html>
"""

HTML_WITH_NOISE = """
<html>
<head><style>body { color: red; }</style></head>
<body>
  <nav>Navigation bar text</nav>
  <p>Useful content here.</p>
  <script>alert('hello');</script>
  <footer>Footer text</footer>
</body>
</html>
"""

HTML_EXTERNAL_LINKS = """
<html>
<body>
  <a href="https://external-domain.com/page">External link</a>
  <a href="/internal">Internal link</a>
  <a href="https://quotes.toscrape.com/page2">Same domain absolute</a>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# TestCrawlerInit
# ---------------------------------------------------------------------------


class TestCrawlerInit:
    """Tests for Crawler.__init__."""

    def test_default_politeness_window_is_six_seconds(self):
        crawler = Crawler("https://example.com")
        assert crawler.politeness_window == 6.0

    def test_custom_politeness_window_stored(self):
        crawler = Crawler("https://example.com", politeness_window=12.5)
        assert crawler.politeness_window == 12.5

    def test_base_url_stored(self):
        url = "https://quotes.toscrape.com/"
        crawler = Crawler(url)
        assert crawler.base_url == url

    def test_initial_visited_set_empty(self):
        crawler = Crawler("https://example.com")
        assert len(crawler.visited) == 0

    def test_initial_pages_dict_empty(self):
        crawler = Crawler("https://example.com")
        assert len(crawler.pages) == 0


# ---------------------------------------------------------------------------
# TestExtractText
# ---------------------------------------------------------------------------


class TestExtractText:
    """Tests for Crawler._extract_text."""

    def test_returns_plain_text(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup("<p>Hello world</p>", "html.parser")
        text = crawler._extract_text(soup)
        assert "Hello" in text
        assert "world" in text

    def test_script_tags_removed(self):
        crawler = Crawler("https://example.com")
        html = "<html><script>var x = 1;</script><p>Visible</p></html>"
        soup = BeautifulSoup(html, "html.parser")
        text = crawler._extract_text(soup)
        assert "var x" not in text
        assert "Visible" in text

    def test_style_tags_removed(self):
        crawler = Crawler("https://example.com")
        html = "<html><style>.red{color:red;}</style><p>Content</p></html>"
        soup = BeautifulSoup(html, "html.parser")
        text = crawler._extract_text(soup)
        assert "color" not in text
        assert "Content" in text

    def test_nav_tags_removed(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup(HTML_WITH_NOISE, "html.parser")
        text = crawler._extract_text(soup)
        assert "Navigation bar text" not in text
        assert "Useful content" in text

    def test_footer_tags_removed(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup(HTML_WITH_NOISE, "html.parser")
        text = crawler._extract_text(soup)
        assert "Footer text" not in text

    def test_empty_body_returns_empty_or_whitespace(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        text = crawler._extract_text(soup)
        assert text.strip() == ""

    def test_returns_string(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup("<p>text</p>", "html.parser")
        assert isinstance(crawler._extract_text(soup), str)


# ---------------------------------------------------------------------------
# TestExtractLinks
# ---------------------------------------------------------------------------


class TestExtractLinks:
    """Tests for Crawler._extract_links."""

    def test_extracts_relative_links(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup(
            '<a href="/about">About</a><a href="/faq">FAQ</a>', "html.parser"
        )
        links = crawler._extract_links(soup, "https://example.com/")
        assert "https://example.com/about" in links
        assert "https://example.com/faq" in links

    def test_filters_external_links(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup(
            '<a href="https://other.com/page">External</a>', "html.parser"
        )
        links = crawler._extract_links(soup, "https://example.com/")
        assert "https://other.com/page" not in links

    def test_skips_already_visited(self):
        crawler = Crawler("https://example.com")
        crawler.visited.add("https://example.com/visited")
        soup = BeautifulSoup(
            '<a href="/visited">Already seen</a>', "html.parser"
        )
        links = crawler._extract_links(soup, "https://example.com/")
        assert "https://example.com/visited" not in links

    def test_strips_fragments(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup(
            '<a href="/page#section">Section</a>', "html.parser"
        )
        links = crawler._extract_links(soup, "https://example.com/")
        assert "https://example.com/page" in links
        # The fragment version should NOT be in links
        assert "https://example.com/page#section" not in links

    def test_no_duplicate_links_returned(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup(
            '<a href="/page">P1</a><a href="/page">P1 again</a>', "html.parser"
        )
        links = crawler._extract_links(soup, "https://example.com/")
        assert links.count("https://example.com/page") == 1

    def test_accepts_absolute_internal_links(self):
        crawler = Crawler("https://quotes.toscrape.com")
        soup = BeautifulSoup(HTML_EXTERNAL_LINKS, "html.parser")
        links = crawler._extract_links(soup, "https://quotes.toscrape.com/")
        assert "https://quotes.toscrape.com/page2" in links

    def test_skips_mailto_hrefs(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup(
            '<a href="mailto:test@example.com">Email</a>', "html.parser"
        )
        links = crawler._extract_links(soup, "https://example.com/")
        assert not any("mailto" in l for l in links)

    def test_skips_javascript_hrefs(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup(
            '<a href="javascript:void(0)">Click</a>', "html.parser"
        )
        links = crawler._extract_links(soup, "https://example.com/")
        assert len(links) == 0

    def test_empty_page_returns_empty_list(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup("<p>No links here</p>", "html.parser")
        links = crawler._extract_links(soup, "https://example.com/")
        assert links == []

    def test_returns_list(self):
        crawler = Crawler("https://example.com")
        soup = BeautifulSoup("", "html.parser")
        result = crawler._extract_links(soup, "https://example.com/")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestCrawlBehaviour
# ---------------------------------------------------------------------------


class TestCrawlBehaviour:
    """Integration-style tests for crawling behaviour using mocked HTTP."""

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_returns_pages_dict(self, MockSession, mock_sleep):
        mock_session = MagicMock()
        MockSession.return_value = mock_session
        mock_session.get.return_value = make_mock_response(HTML_PAGE_NO_LINKS)

        crawler = Crawler("https://example.com", politeness_window=0)
        result = crawler.crawl()

        assert isinstance(result, dict)
        assert "https://example.com" in result

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_politeness_sleep_called(self, MockSession, mock_sleep):
        mock_session = MagicMock()
        MockSession.return_value = mock_session

        # Home page links to /page2
        responses = [
            make_mock_response(HTML_PAGE_WITH_LINKS),
            make_mock_response(HTML_PAGE_NO_LINKS),
            make_mock_response(HTML_PAGE_NO_LINKS),  # /about
            make_mock_response(HTML_PAGE_NO_LINKS),  # /contact
        ]
        mock_session.get.side_effect = responses

        crawler = Crawler("https://example.com", politeness_window=6.0)
        crawler.crawl()

        # sleep should have been called with 6.0
        mock_sleep.assert_called_with(6.0)

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_does_not_revisit_pages(self, MockSession, mock_sleep):
        """A page linked from itself should only be fetched once."""
        mock_session = MagicMock()
        MockSession.return_value = mock_session

        # Page links to itself
        self_link_html = '<html><body><a href="/">Home</a><p>text</p></body></html>'
        mock_session.get.return_value = make_mock_response(self_link_html)

        crawler = Crawler("https://example.com", politeness_window=0)
        crawler.crawl()

        assert mock_session.get.call_count == 1

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_handles_http_error_gracefully(self, MockSession, mock_sleep):
        mock_session = MagicMock()
        MockSession.return_value = mock_session
        mock_session.get.return_value = make_mock_response("", status_code=404)

        crawler = Crawler("https://example.com", politeness_window=0)
        result = crawler.crawl()  # Must not raise

        assert result == {}

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_handles_connection_error_gracefully(self, MockSession, mock_sleep):
        mock_session = MagicMock()
        MockSession.return_value = mock_session
        mock_session.get.side_effect = req.exceptions.ConnectionError("unreachable")

        crawler = Crawler("https://example.com", politeness_window=0)
        result = crawler.crawl()

        assert result == {}

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_crawls_linked_pages(self, MockSession, mock_sleep):
        """Starting page links to two children; all three should be crawled."""
        mock_session = MagicMock()
        MockSession.return_value = mock_session

        mock_session.get.side_effect = [
            make_mock_response(HTML_PAGE_WITH_LINKS),  # home  -> /about, /contact
            make_mock_response(HTML_PAGE_NO_LINKS),     # /about
            make_mock_response(HTML_PAGE_NO_LINKS),     # /contact
        ]

        crawler = Crawler("https://example.com", politeness_window=0)
        pages = crawler.crawl()

        assert len(pages) == 3

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_page_text_stored(self, MockSession, mock_sleep):
        mock_session = MagicMock()
        MockSession.return_value = mock_session
        mock_session.get.return_value = make_mock_response(
            "<html><body><p>unique marker text</p></body></html>"
        )

        crawler = Crawler("https://example.com", politeness_window=0)
        pages = crawler.crawl()

        assert any("unique marker text" in text for text in pages.values())

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_timeout_error_handled(self, MockSession, mock_sleep):
        mock_session = MagicMock()
        MockSession.return_value = mock_session
        mock_session.get.side_effect = req.exceptions.Timeout("timed out")

        crawler = Crawler("https://example.com", politeness_window=0)
        result = crawler.crawl()

        assert result == {}
