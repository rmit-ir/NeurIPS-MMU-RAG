from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import aiohttp
from aiohttp import ClientError, ServerTimeoutError, ClientConnectionError

from tools.logging_utils import get_logger
from tools.retry_utils import retry
from tools.web_search import SearchResult
from tools.brave_search import BraveSearchResult

logger = get_logger('jina_retriever')

JINA_READER_BASE = os.getenv("JINA_READER_BASE", "https://r.jina.ai/")
JINA_REFERER = os.getenv("JINA_REFERER", "https://search.chai-research.au/")


class JinaRetrieverError(Exception):
    """Custom exception for Jina retriever related errors."""


async def _fetch_single_url(
    session: aiohttp.ClientSession,
    url: str,
    index: int,
    headers: dict,
    timeout: float,
    id_prefix: Optional[str],
    search_title: Optional[str],
    search_description: Optional[str],
) -> Optional[SearchResult]:
    """Fetch a single URL using Jina AI Reader API.

    Args:
        session: aiohttp session
        url: URL to fetch
        index: Index for sid generation
        headers: Request headers
        timeout: Request timeout in seconds
        id_prefix: Optional prefix for sid
        search_title: Title from search results (fallback if Jina doesn't return one)
        search_description: Description from search results

    Returns:
        SearchResult if successful, None if failed
    """
    jina_url = f"{JINA_READER_BASE}{url}"

    try:
        async with session.get(
            jina_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if 200 <= response.status < 300:
                response_data = await response.json()
                data = response_data.get("data", {})

                content = data.get("content") or ""
                title = data.get("title") or search_title or ""
                final_url = data.get("url") or url
                description = data.get(
                    "description") or search_description or ""

                # Prepend title to text
                if title:
                    text = f"# {title}\n\n{content}"
                else:
                    text = content

                result = SearchResult(
                    type="brave_jina",
                    url=final_url,
                    id=url,  # Original URL as ID
                    text=text,
                    language=data.get("metadata", {}).get("lang"),
                    token_count=text.count(" ") + 1 if text.strip() else 0,
                    sid=f"{id_prefix}_{index}" if id_prefix else str(index),
                    metadata={
                        "title": title,
                        "description": description,
                        "original_url": url,
                        "published_time": data.get("publishedTime"),
                    },
                    date=data.get("publishedTime"),
                    dump=None,
                    file_path=None,
                    language_score=None,
                    score=None,
                )
                logger.debug(
                    "Jina fetch successful",
                    url=url,
                    content_length=len(content)
                )
                return result
            else:
                logger.warning(
                    "Jina fetch failed",
                    url=url,
                    status=response.status
                )
                return None
    except asyncio.TimeoutError:
        logger.warning("Jina fetch timeout", url=url)
        return None
    except Exception as e:
        logger.warning("Jina fetch error", url=url, error=str(e))
        return None


@retry(max_retries=2, retry_on=(ClientError, ServerTimeoutError, ClientConnectionError, JinaRetrieverError))
async def retrieve_urls(
    urls: List[str],
    api_key: Optional[str] = None,
    max_concurrent: int = 10,
    timeout: int = 5,
    id_prefix: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None,
    search_metadata: Optional[List[BraveSearchResult]] = None,
) -> List[SearchResult]:
    """Batch retrieve content from URLs using Jina AI Reader API.

    Args:
        urls: List of URLs to fetch
        api_key: Jina API key. If None, uses JINA_API_KEY env var
        max_concurrent: Maximum number of concurrent requests (default: 10)
        timeout: Request timeout per URL in seconds (default: 5)
        id_prefix: Optional prefix for sid field
        session: Optional existing aiohttp session
        search_metadata: Optional list of BraveSearchResult with title/description

    Returns:
        List of SearchResult objects (only successful fetches)

    Raises:
        ValueError: If API key is not provided
    """
    if not urls:
        return []

    key = api_key or os.getenv("JINA_API_KEY")
    if not key:
        raise ValueError(
            "Jina API key not provided. Set JINA_API_KEY or pass api_key."
        )

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {key}",
        "X-Engine": "direct",
        "X-Referer": JINA_REFERER,
        "X-Retain-Images": "none",
        "X-Return-Format": "markdown",
        "X-Timeout": str(int(timeout)),
        "X-Token-Budget": "100000",
    }

    close_session = False
    if session is None:
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        session = aiohttp.ClientSession(connector=connector)
        close_session = True

    try:
        start_time = asyncio.get_event_loop().time()

        # Create tasks for batch fetching
        tasks = []
        for i, url in enumerate(urls, start=1):
            search_title = None
            search_description = None
            if search_metadata and i - 1 < len(search_metadata):
                meta = search_metadata[i - 1]
                search_title = meta.get("title")
                search_description = meta.get("description")

            tasks.append(
                _fetch_single_url(
                    session=session,
                    url=url,
                    index=i,
                    headers=headers,
                    timeout=timeout,
                    id_prefix=id_prefix,
                    search_title=search_title,
                    search_description=search_description,
                )
            )

        results_raw = await asyncio.gather(*tasks)

        # Filter out None results
        results = [r for r in results_raw if r is not None]

        took_time = asyncio.get_event_loop().time() - start_time
        logger.info(
            "Jina batch retrieval completed",
            total_urls=len(urls),
            successful=len(results),
            failed=len(urls) - len(results),
            duration=took_time
        )

        return results
    finally:
        if close_session:
            await session.close()


async def main():
    """Test function for Jina retriever."""
    test_urls = [
        "https://simonwillison.net/2024/Jun/16/jina-ai-reader/",
        "https://example.com/",
    ]

    results = await retrieve_urls(test_urls, timeout=10)
    print(f"Retrieved {len(results)} results\n")

    for result in results:
        print(f"URL: {result.url}")
        print(f"Title: {result.metadata.get('title', 'N/A')}")
        print(f"Text preview: {result.text[:200]}...")
        print(f"Token count: {result.token_count}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
