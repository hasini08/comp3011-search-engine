"""
test_main.py - Unit tests for the main CLI module.

Tests cover:
  - cmd_load: missing file, successful load
  - cmd_build: no pages returned, successful build
  - print_help: output contains all commands
  - main REPL: unknown command, quit, empty input, help
"""

import os
import sys
import json
import tempfile

import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexer import Indexer
from search import SearchEngine
import main as main_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_indexer_and_engine():
    idx = Indexer()
    engine = SearchEngine(idx)
    return idx, engine


def make_temp_index(pages=None):
    """Build an index and save it to a temp file. Returns (indexer, path)."""
    if pages is None:
        pages = {"https://example.com/p1": "hello world foo bar"}
    idx = Indexer()
    idx.build_index(pages)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        path = f.name
    idx.save_index(path)
    return idx, path


# ---------------------------------------------------------------------------
# TestCmdLoad
# ---------------------------------------------------------------------------


class TestCmdLoad:
    """Tests for the cmd_load function."""

    def test_load_missing_file_prints_error(self, capsys):
        idx, engine = make_indexer_and_engine()
        with patch.object(main_module, "INDEX_FILE", "/nonexistent/path.json"):
            main_module.cmd_load(idx, engine)
        out = capsys.readouterr().out
        assert "not found" in out.lower() or "build" in out.lower()

    def test_load_valid_file_populates_index(self, capsys):
        _, path = make_temp_index()
        idx, engine = make_indexer_and_engine()
        try:
            with patch.object(main_module, "INDEX_FILE", path):
                main_module.cmd_load(idx, engine)
            assert len(idx.index) > 0
        finally:
            os.unlink(path)

    def test_load_updates_search_engine(self, capsys):
        _, path = make_temp_index()
        idx, engine = make_indexer_and_engine()
        try:
            with patch.object(main_module, "INDEX_FILE", path):
                main_module.cmd_load(idx, engine)
            assert engine.indexer is idx
        finally:
            os.unlink(path)

    def test_load_prints_word_count(self, capsys):
        _, path = make_temp_index()
        idx, engine = make_indexer_and_engine()
        try:
            with patch.object(main_module, "INDEX_FILE", path):
                main_module.cmd_load(idx, engine)
            out = capsys.readouterr().out
            assert any(c.isdigit() for c in out)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TestCmdBuild
# ---------------------------------------------------------------------------


class TestCmdBuild:
    """Tests for the cmd_build function."""

    @patch("main.Crawler")
    def test_build_with_no_pages_prints_warning(self, MockCrawler, capsys, tmp_path):
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {}
        MockCrawler.return_value = mock_crawler

        idx = Indexer()
        with patch.object(main_module, "INDEX_FILE", str(tmp_path / "index.json")):
            main_module.cmd_build(idx)

        out = capsys.readouterr().out
        assert "no pages" in out.lower() or "network" in out.lower()

    @patch("main.Crawler")
    def test_build_calls_crawler(self, MockCrawler, capsys, tmp_path):
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {
            "https://example.com": "hello world"
        }
        MockCrawler.return_value = mock_crawler

        idx = Indexer()
        with patch.object(main_module, "INDEX_FILE", str(tmp_path / "index.json")):
            main_module.cmd_build(idx)

        mock_crawler.crawl.assert_called_once()

    @patch("main.Crawler")
    def test_build_saves_index_file(self, MockCrawler, capsys, tmp_path):
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {
            "https://example.com": "hello world"
        }
        MockCrawler.return_value = mock_crawler

        idx = Indexer()
        index_path = str(tmp_path / "index.json")
        with patch.object(main_module, "INDEX_FILE", index_path):
            main_module.cmd_build(idx)

        assert os.path.exists(index_path)

    @patch("main.Crawler")
    def test_build_populates_index(self, MockCrawler, capsys, tmp_path):
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = {
            "https://example.com": "hello world foo"
        }
        MockCrawler.return_value = mock_crawler

        idx = Indexer()
        with patch.object(main_module, "INDEX_FILE", str(tmp_path / "index.json")):
            main_module.cmd_build(idx)

        assert "hello" in idx.index


# ---------------------------------------------------------------------------
# TestPrintHelp
# ---------------------------------------------------------------------------


class TestPrintHelp:
    """Tests for the print_help function."""

    def test_help_contains_build(self, capsys):
        main_module.print_help()
        assert "build" in capsys.readouterr().out

    def test_help_contains_load(self, capsys):
        main_module.print_help()
        assert "load" in capsys.readouterr().out

    def test_help_contains_print(self, capsys):
        main_module.print_help()
        assert "print" in capsys.readouterr().out

    def test_help_contains_find(self, capsys):
        main_module.print_help()
        assert "find" in capsys.readouterr().out

    def test_help_contains_quit(self, capsys):
        main_module.print_help()
        assert "quit" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# TestMainREPL
# ---------------------------------------------------------------------------


class TestMainREPL:
    """Tests for the main() REPL loop."""

    def test_quit_exits_cleanly(self, capsys):
        with patch("builtins.input", side_effect=["quit"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "goodbye" in out.lower()

    def test_exit_exits_cleanly(self, capsys):
        with patch("builtins.input", side_effect=["exit"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "goodbye" in out.lower()

    def test_unknown_command_prints_message(self, capsys):
        with patch("builtins.input", side_effect=["invalidcmd", "quit"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "unknown" in out.lower()

    def test_help_command_works(self, capsys):
        with patch("builtins.input", side_effect=["help", "quit"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "build" in out

    def test_empty_input_does_not_crash(self, capsys):
        with patch("builtins.input", side_effect=["", "   ", "quit"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "goodbye" in out.lower()

    def test_keyboard_interrupt_exits_cleanly(self, capsys):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            main_module.main()
        out = capsys.readouterr().out
        assert "goodbye" in out.lower()

    def test_print_without_args_shows_usage(self, capsys):
        with patch("builtins.input", side_effect=["print", "quit"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "usage" in out.lower() or "print" in out.lower()

    def test_find_without_args_shows_usage(self, capsys):
        with patch("builtins.input", side_effect=["find", "quit"]):
            main_module.main()
        out = capsys.readouterr().out
        assert "usage" in out.lower() or "find" in out.lower()
