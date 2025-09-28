from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional, NamedTuple

import aiohttp

from tools.logging_utils import get_logger

logger = get_logger('web_search')


class SearchResult(NamedTuple):
    """Typed result from external search sources."""
    text: str
    id: str
    sid: str
    """short identifier, e.g. 1_0, 1_1, etc."""
    dump: str
    url: str
    date: str
    file_path: str
    language: str
    language_score: float
    token_count: int
    score: float | None


class SearchError(NamedTuple):
    """Error result from search operations."""
    error: str
    value: Optional[Any] = None


FINEWEB_BASE_URL = "https://clueweb22.us/fineweb/search"
CLUEWEB_BASE_URL = "https://clueweb22.us/search"


class WebSearchError(Exception):
    """Custom exception for web search related errors."""


def _decode_results(json_payload: Dict[str, Any], id_prefix: Optional[str] = None) -> List[SearchResult | SearchError]:
    """Decode the Base64 JSON documents in the 'results' field.

    Any decoding / JSON errors are handled gracefully: problematic entries are
    returned as SearchError objects.
    """
    raw_results = json_payload.get("results", []) or []
    results: List[SearchResult | SearchError] = []
    idx = 1
    for item in raw_results:
        if not isinstance(item, str):
            continue
        try:
            binary = base64.b64decode(item, validate=True)
        except Exception as e:  # noqa: BLE001
            results.append(SearchError(error=f"base64_decode_failed: {e}"))
            continue
        try:
            obj = json.loads(binary.decode("utf-8", errors="replace"))
            if isinstance(obj, dict) and "_error" not in obj:
                # Convert to typed SearchResult
                try:
                    result = SearchResult(
                        text=obj.get("text", ""),
                        id=obj.get("id", ""),
                        sid=f"{id_prefix}_{idx}" if id_prefix else str(idx),
                        dump=obj.get("dump", ""),
                        url=obj.get("url", ""),
                        date=obj.get("date", ""),
                        file_path=obj.get("file_path", ""),
                        language=obj.get("language", ""),
                        language_score=float(obj.get("language_score", 0.0)),
                        token_count=int(obj.get("token_count", 0)),
                        score=obj.get("score", None)
                    )
                    results.append(result)
                except (ValueError, TypeError) as e:
                    results.append(SearchError(
                        error=f"type_conversion_failed: {e}", value=obj))
            else:
                results.append(SearchError(
                    error="decoded_not_dict", value=obj))
        except Exception as e:  # noqa: BLE001
            results.append(SearchError(error=f"json_parse_failed: {e}"))
        idx += 1
    return results


async def _make_search_request(
    url: str,
    params: Dict[str, str],
    headers: Optional[Dict[str, str]] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 20.0,
    service_name: str = "Search"
) -> Dict[str, Any]:
    """Make a search request to the specified URL with common error handling."""
    close_session = False
    if session is None:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        session = aiohttp.ClientSession(timeout=timeout_cfg)
        close_session = True

    try:
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise WebSearchError(
                    f"{service_name} request failed: {resp.status} {text[:200]}"
                )
            json_resp = await resp.json(content_type=None)
        return json_resp
    finally:
        if close_session:
            await session.close()


async def search_fineweb(
    query: str,
    k: int = 5,
    id_prefix: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 20.0,
) -> List[SearchResult | SearchError]:
    """Search the FineWeb dataset (no API key required).

    Args:
        query: Search query string.
        k: Number of documents to retrieve.
        id_prefix: e.g. "1" results in "1_0", "1_1", "1_2", etc.
        session: Optional existing aiohttp session.
        timeout: Total request timeout in seconds.

    Returns:
        List of SearchResult objects.
    """
    if not query:
        raise ValueError("query must be non-empty")
    if k <= 0:
        raise ValueError("k must be > 0")

    params = {"query": query, "k": str(k)}
    json_resp = await _make_search_request(
        FINEWEB_BASE_URL, params, None, session, timeout, "FineWeb"
    )
    results = _decode_results(json_resp, id_prefix)
    return results


async def search_clueweb(
    query: str,
    k: int = 5,
    api_key: Optional[str] = None,
    cw22_a: bool = False,
    id_prefix: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 30.0,
) -> List[SearchResult | SearchError]:
    """Search the ClueWeb-22 collection (API key required).

    Args:
        query: Search query string.
        k: Number of documents to retrieve.
        api_key: ClueWeb retriever API key. If None, will attempt
                 CLUEWEB_API_KEY or RAG_CLUEWEB_API_KEY environment vars.
        cw22_a: If True, use ClueWeb22-A instead of default B.
        session: Optional existing aiohttp session.
        timeout: Total request timeout in seconds.

    Returns:
        List of SearchResult objects.
    """
    if not query:
        raise ValueError("query must be non-empty")
    if k <= 0:
        raise ValueError("k must be > 0")

    key = api_key or os.getenv(
        "CLUEWEB_API_KEY") or os.getenv("RAG_CLUEWEB_API_KEY")
    if not key:
        raise WebSearchError(
            "ClueWeb API key not provided. Set CLUEWEB_API_KEY or pass api_key."
        )

    params = {"query": query, "k": str(k)}
    if cw22_a:
        params["cw22_a"] = "true"
    headers = {"x-api-key": key}

    json_resp = await _make_search_request(
        CLUEWEB_BASE_URL, params, headers, session, timeout, "ClueWeb"
    )
    logger.info("TODO: ClueWeb not tested yet")

    results = _decode_results(json_resp, id_prefix)
    return results


def _sync_wrapper(async_func, *args, **kwargs) -> List[SearchResult | SearchError]:
    """Generic synchronous wrapper for async search functions."""
    try:
        loop = asyncio.get_running_loop()  # Will raise if no loop
    except RuntimeError:
        return asyncio.run(async_func(*args, **kwargs))
    else:
        # If already in an event loop, the caller should use the async version.
        raise RuntimeError(
            f"{async_func.__name__}_sync called from within an existing event loop; "
            f"use await {async_func.__name__}(...)."
        )


def search_fineweb_sync(query: str, k: int = 5, timeout: float = 20.0) -> List[SearchResult | SearchError]:
    """Synchronous wrapper for fineweb_search (creates its own loop if needed)."""
    return _sync_wrapper(search_fineweb, query=query, k=k, timeout=timeout)


def search_clueweb_sync(
    query: str,
    k: int = 5,
    api_key: Optional[str] = None,
    cw22_a: bool = False,
    timeout: float = 30.0,
) -> List[SearchResult | SearchError]:
    """Synchronous wrapper for clueweb_search (creates its own loop if needed)."""
    return _sync_wrapper(
        search_clueweb, query=query, k=k, api_key=api_key, cw22_a=cw22_a, timeout=timeout
    )


# Backward compatibility aliases
fineweb_search = search_fineweb
clueweb_search = search_clueweb


async def main():
    results = await search_fineweb("machine learning", k=10)
    for i, doc in enumerate(results):
        print(f"Result {i+1}", doc)


if __name__ == "__main__":
    asyncio.run(main())
