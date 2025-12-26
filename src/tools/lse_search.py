from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import ClientError, ServerTimeoutError, ClientConnectionError

from tools.logging_utils import get_logger
from tools.retry_utils import retry
from tools.web_search import SearchResult

logger = get_logger('lse_search')

LSE_SEARCH_BASE_URL = "https://search-cw22b-diskann-minicpm.rankun.org/search"


class LSESearchError(Exception):
    """Custom exception for LSE search related errors."""


def _parse_lse_results(json_payload: Dict[str, Any], id_prefix: Optional[str] = None) -> List[SearchResult]:
    """Parse LSE search API response into SearchResult objects.

    Args:
        json_payload: The JSON response from LSE search API
        id_prefix: Optional prefix for the sid field, used by caller to distinguish requests
    """
    raw_results = json_payload.get("results", []) or []
    results: List[SearchResult] = []

    for idx, item in enumerate(raw_results, start=1):
        doc = item.get("doc", {})
        if not isinstance(doc, dict):
            logger.warning("Skipping empty document", index=idx, doc=doc)
            continue

        result = SearchResult(
            type="clue_web",
            url=doc.get("URL", "").strip(),
            id=doc.get("ClueWeb22-ID", ""),
            text=doc.get("Clean-Text", ""),
            language=doc.get("Language", ""),
            token_count=doc.get("Clean-Text", "").count(" "),
            sid=f"{id_prefix}_{idx}" if id_prefix else str(idx),
            metadata={
                "url_hash": doc.get("URL-hash", ""),
                "docid": item.get("docid", ""),
                "distance": item.get("distance", 0.0),
            },
            date=None,
            dump=None,
            file_path=None,
            language_score=None,
            score=item.get("distance", None),
        )
        results.append(result)

    return results


@retry(max_retries=4, retry_on=(ClientError, ServerTimeoutError, ClientConnectionError, LSESearchError))
async def search_clueweb(
    query: str,
    k: int = 10,
    id_prefix: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 60.0,
    cw22_a: bool = False,
) -> List[SearchResult]:
    """Search ClueWeb22 using LSE DiskANN + MiniCPM search API.

    Args:
        query: Search query string
        k: Number of documents to retrieve (default: 10)
        id_prefix: Optional prefix for the sid field (e.g., "1" results in "1_1", "1_2", etc.)
        session: Optional existing aiohttp session
        timeout: Total request timeout in seconds (default: 60.0)
        cw22_a: Kept for backward compatibility, ignored here

    Returns:
        List of SearchResult objects

    Raises:
        ValueError: If query is empty or k is invalid
        LSESearchError: If the API request fails
    """
    if not query:
        raise ValueError("query must be non-empty")
    if k <= 0:
        raise ValueError("k must be > 0")

    close_session = False
    if session is None:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        session = aiohttp.ClientSession(timeout=timeout_cfg)
        close_session = True

    try:
        payload = {"query": query, "k": k}
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        start_time = asyncio.get_event_loop().time()

        async with session.post(LSE_SEARCH_BASE_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise LSESearchError(
                    f"LSE search request failed: {resp.status} {text[:200]}"
                )
            json_resp = await resp.json(content_type=None)

        results = _parse_lse_results(json_resp, id_prefix)
        took_time = asyncio.get_event_loop().time() - start_time

        logger.info(
            "LSE ClueWeb search completed",
            query=query,
            num_results=len(results),
            duration=took_time
        )

        return results
    finally:
        if close_session:
            await session.close()


async def main():
    """Test function for LSE ClueWeb search."""
    start_time = asyncio.get_event_loop().time()
    results = await search_clueweb(
        "Wikipedia",
        k=10
    )
    took_time = asyncio.get_event_loop().time() - start_time

    print(f"LSE ClueWeb Search completed in {took_time:.4f} seconds")
    print(f"Retrieved {len(results)} results\n")

    for i, doc in enumerate(results, start=1):
        if isinstance(doc, SearchResult):
            print(f"{i}. {doc.id}")
            print(f"   URL: {doc.url}")
            print(f"   Distance: {doc.metadata.get('distance', 'N/A')}")
            print(f"   Language: {doc.language}")
            print(f"   Text preview: {doc.text[:200]}...")
            print()
        else:
            print(f"{i}. Error: {doc.error}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
