"""
conftest.py - Pytest configuration for the COMP3011 Search Engine project.

Adds the src/ directory to sys.path so that pytest can import the
source modules (crawler, indexer, search) when running tests from the
project root directory.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
