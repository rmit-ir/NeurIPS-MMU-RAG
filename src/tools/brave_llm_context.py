"""
Brave LLM Context API client.
Uses the /res/v1/llm/context endpoint to get pre-extracted, relevance-scored
content optimized for LLM/RAG consumption.

Unlike regular web search, this returns actual page content (text chunks)
ready for LLM use, without needing a separate scraper like Jina.

See https://api-dashboard.search.brave.com/documentation/services/llm-context
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import aiohttp
from aiohttp import ClientError, ServerTimeoutError, ClientConnectionError

from tools.logging_utils import get_logger
from tools.retry_utils import retry
from tools.web_search import SearchResult

logger = get_logger('brave_llm_context')

# Brave Search API pricing — AWS Marketplace "Data for AI" tier is $5 CPM
# ($0.005 per call). Override via env if you're on a different plan.
COST_PER_CALL_USD = float(os.getenv("BRAVE_COST_PER_CALL_USD", "0.005"))


class BraveLLMContextError(Exception):
    """Custom exception for Brave LLM Context API errors."""


@retry(max_retries=4, retry_on=(ClientError, ServerTimeoutError, ClientConnectionError, BraveLLMContextError))
async def brave_llm_context(
    query: str,
    count: int = 20,
    max_tokens: int = 8192,
    max_urls: int = 20,
    freshness: Optional[str] = None,
    threshold_mode: str = "balanced",
    excluded_domains: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 30.0,
) -> List[SearchResult]:
    """Search using Brave LLM Context API.

    Returns pre-extracted content as SearchResult objects, compatible with
    the existing search pipeline (reranking, chunking, etc.).

    Args:
        query: Search query string.
        count: Number of search results to consider (1-50, default 20).
        max_tokens: Approximate max tokens in response (1024-32768, default 8192).
        max_urls: Max URLs in response (1-50, default 20).
        freshness: Filter by freshness (pd/pw/pm/py or YYYY-MM-DDtoYYYY-MM-DD).
        threshold_mode: Relevance threshold (strict/balanced/lenient/disabled).
        excluded_domains: Domains to exclude from results (e.g. ["wikipedia.org"]).
            Filtered client-side after API response.
        api_key: Brave API key. If None, uses BRAVE_API_KEY env var.
        session: Optional existing aiohttp session.
        timeout: Request timeout in seconds.

    Returns:
        List of SearchResult objects with extracted content.
    """
    if not query:
        raise ValueError("query must be non-empty")

    key = api_key or os.getenv("BRAVE_API_KEY")
    if not key:
        raise ValueError(
            "Brave API key not provided. Set BRAVE_API_KEY or pass api_key."
        )

    # Truncate query to stay within Brave API limits
    if len(query) > 300:
        query = query[:300]
    if query.count(' ') > 40:
        query = ' '.join(query.split()[:40])

    close_session = False
    if session is None:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        session = aiohttp.ClientSession(timeout=timeout_cfg)
        close_session = True

    try:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "X-Subscription-Token": key,
        }
        params = {
            "q": query,
            "count": str(min(count, 50)),
            "maximum_number_of_urls": str(min(max_urls, 50)),
            "maximum_number_of_tokens": str(min(max(max_tokens, 1024), 32768)),
            "context_threshold_mode": threshold_mode,
        }
        if freshness:
            params["freshness"] = freshness

        start_time = asyncio.get_event_loop().time()

        async with session.get(
            "https://api.search.brave.com/res/v1/llm/context",
            headers=headers,
            params=params,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.warning(
                    "Brave LLM Context request failed, returning empty results",
                    status=resp.status, query=query, detail=text[:200],
                )
                return []
            data = await resp.json()

        results = _parse_response(data)

        # Client-side domain filtering (API doesn't support -site: operators)
        if excluded_domains:
            before = len(results)
            results = [
                r for r in results
                if not any(d in r.url for d in excluded_domains)
            ]
            logger.info("Filtered excluded domains",
                        excluded=excluded_domains,
                        before=before, after=len(results))

        took_time = asyncio.get_event_loop().time() - start_time
        logger.info(
            "Brave LLM Context search completed",
            query=query,
            num_results=len(results),
            duration=took_time,
        )

        return results
    finally:
        if close_session:
            await session.close()


def _parse_response(data: dict) -> List[SearchResult]:
    """Parse Brave LLM Context response into SearchResult objects."""
    results: List[SearchResult] = []
    grounding = data.get("grounding", {})
    sources = data.get("sources", {})
    generic = grounding.get("generic", [])

    idx = 1
    for item in generic:
        url = item.get("url", "")
        title = item.get("title", "")
        snippets = item.get("snippets", [])

        if not snippets:
            continue

        # Join all snippets into a single text block
        text = "\n\n".join(snippets)

        # Get source metadata
        source_meta = sources.get(url, {})
        age_list = source_meta.get("age", [])
        date = age_list[1] if len(age_list) > 1 else None  # ISO date format

        results.append(SearchResult(
            type="brave_jina",  # reuse existing type for compatibility
            text=text,
            id=f"brave_llm_ctx_{idx}",
            sid=str(idx),
            url=url,
            token_count=len(text.split()),
            metadata={"title": title, "hostname": source_meta.get("hostname", "")},
            dump=None,
            date=date,
            file_path=None,
            language=None,
            language_score=None,
            score=None,
        ))
        idx += 1

    return results


async def main():
    """Test function for Brave LLM Context."""
    import sys

    from dotenv import load_dotenv
    load_dotenv()

    query = sys.argv[1] if len(sys.argv) > 1 else "Lunisolar calendar"
    results = await brave_llm_context(query, count=10, max_tokens=4096)
    print(f"Retrieved {len(results)} results\n")
    for r in results:
        print("-" * 80)
        print(f"[{r.sid}] {r.url}")
        print(f"    Date: {r.date}")
        print(f"    Words: {r.token_count}")
        print(f"    Text: {r.text}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
