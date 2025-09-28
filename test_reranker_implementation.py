#!/usr/bin/env python3
"""Simple test script to verify the reranker implementation."""

import sys
import os
sys.path.append('src')

from tools.reranker_vllm import get_reranker, SearchResultRanked
from tools.web_search import SearchResult

def create_mock_search_results():
    """Create mock SearchResult objects for testing."""
    return [
        SearchResult(
            text="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            id="1",
            sid="1_1",
            dump="test_dump_1",
            url="https://example.com/ml-basics",
            date="2024-01-01",
            file_path="/path/to/file1",
            language="en",
