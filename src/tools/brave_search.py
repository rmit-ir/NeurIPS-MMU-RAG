from __future__ import annotations

import asyncio
import os
from typing import List, Optional, TypedDict

import aiohttp
from aiohttp import ClientError, ServerTimeoutError, ClientConnectionError

from tools.logging_utils import get_logger
from tools.retry_utils import retry

logger = get_logger('brave_search')

BRAVE_SEARCH_URL = os.getenv(
    "BRAVE_SEARCH_URL", "https://api.search.brave.com/res/v1/web/search")


class BraveSearchError(Exception):
    """Custom exception for Brave search related errors."""


class BraveSearchResult(TypedDict):
    """Result structure for Brave Search API response."""
    url: str
    title: Optional[str]
    description: Optional[str]


@retry(max_retries=4, retry_on=(ClientError, ServerTimeoutError, ClientConnectionError, BraveSearchError))
async def brave_search(
    query: str,
    count: int = 10,
    api_key: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 30.0,
) -> List[BraveSearchResult]:
    """Search using Brave Search API.

    Args:
        query: Search query string
        count: Number of results to return (max 20)
        api_key: Brave API key. If None, uses BRAVE_API_KEY env var
        session: Optional existing aiohttp session
        timeout: Total request timeout in seconds (default: 30.0)

    Returns:
        List of BraveSearchResult objects

    Raises:
        ValueError: If query is empty or API key is not provided
        BraveSearchError: If the API request fails
    """
    if not query:
        raise ValueError("query must be non-empty")

    key = api_key or os.getenv("BRAVE_API_KEY")
    if not key:
        raise ValueError(
            "Brave API key not provided. Set BRAVE_API_KEY or pass api_key."
        )

    close_session = False
    if session is None:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        session = aiohttp.ClientSession(timeout=timeout_cfg)
        close_session = True
    
    # truncate query to less than 400 characters or 50 words.
    if len(query) > 400:
        query = query[:400]
    if query.count(' ') > 50:
        query = ' '.join(query.split()[:50])

    try:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "X-Subscription-Token": key,
        }
        params = {
            "q": query,
            "count": str(min(count, 20)),  # Brave API max is 20
        }

        start_time = asyncio.get_event_loop().time()

        async with session.get(BRAVE_SEARCH_URL, headers=headers, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise BraveSearchError(
                    f"Brave search request failed: {resp.status} {text[:200]}"
                )
            data = await resp.json()

        results: List[BraveSearchResult] = []
        if "web" in data and "results" in data["web"]:
            for result in data["web"]["results"]:
                results.append({
                    "url": result.get("url", ""),
                    "title": result.get("title"),
                    "description": result.get("description"),
                })

        took_time = asyncio.get_event_loop().time() - start_time
        logger.info(
            "Brave search completed",
            query=query,
            num_results=len(results),
            duration=took_time
        )

        return results
    finally:
        if close_session:
            await session.close()


async def main():
    """Test function for Brave search."""
    results = await brave_search("latest AI research papers 2024", count=5)
    print(f"Retrieved {len(results)} results\n")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(
            f"   Description: {result['description'][:100] if result['description'] else 'N/A'}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
