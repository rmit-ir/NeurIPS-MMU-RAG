#!/usr/bin/env python3
"""Test script for the async reranker implementation."""

import asyncio
import time
from typing import List
from tools.reranker_vllm import get_reranker, SearchResultRanked
from tools.web_search import SearchResult

# Mock search results for testing
def create_mock_search_results() -> List[SearchResult]:
    """Create mock search results for testing."""
    return [
        SearchResult(
            text="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            id="1",
            sid="s1",
            dump="dump1",
            url="https://example.com/ml-basics",
            date="2024-01-01",
