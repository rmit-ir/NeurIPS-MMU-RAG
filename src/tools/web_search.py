from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional

import aiohttp

FINEWEB_BASE_URL = "https://clueweb22.us/fineweb/search"
CLUEWEB_BASE_URL = "https://clueweb22.us/search"

__all__ = [
    "fineweb_search",
    "clueweb_search",
    "fineweb_search_sync",
    "clueweb_search_sync",
]


class WebSearchError(Exception):
    """Custom exception for web search related errors."""


async def _decode_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Decode the Base64 JSON documents in the 'results' field.

    Any decoding / JSON errors are handled gracefully: problematic entries are
    skipped (unless partial decode is available) and an "_error" field is
    attached to the partial record when possible.
    """

    raw_results = payload.get("results", []) or []
    decoded: List[Dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, str):
            continue
        try:
            binary = base64.b64decode(item, validate=True)
        except Exception as e:  # noqa: BLE001
            decoded.append({"_error": f"base64_decode_failed: {e}"})
            continue
        try:
            obj = json.loads(binary.decode("utf-8", errors="replace"))
            if isinstance(obj, dict):
                decoded.append(obj)
            else:
                decoded.append({"_error": "decoded_not_dict", "value": obj})
        except Exception as e:  # noqa: BLE001
            decoded.append({"_error": f"json_parse_failed: {e}"})
    return decoded


async def fineweb_search(
    query: str,
    k: int = 5,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 20.0,
) -> List[Dict[str, Any]]:
    """Search the FineWeb dataset (no API key required).

    Args:
        query: Search query string.
        k: Number of documents to retrieve.
        session: Optional existing aiohttp session.
        timeout: Total request timeout in seconds.

    Returns:
        List of decoded document dicts.
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
        params = {"query": query, "k": str(k)}
        async with session.get(FINEWEB_BASE_URL, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise WebSearchError(
                    f"FineWeb request failed: {resp.status} {text[:200]}"
                )
            data = await resp.json(content_type=None)
        return await _decode_results(data)
    finally:
        if close_session:
            await session.close()


async def clueweb_search(
    query: str,
    k: int = 5,
    api_key: Optional[str] = None,
    cw22_a: bool = False,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
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
        List of decoded document dicts.
    """
    if not query:
        raise ValueError("query must be non-empty")
    if k <= 0:
        raise ValueError("k must be > 0")

    key = api_key or os.getenv("CLUEWEB_API_KEY") or os.getenv("RAG_CLUEWEB_API_KEY")
    if not key:
        raise WebSearchError(
            "ClueWeb API key not provided. Set CLUEWEB_API_KEY or pass api_key."
        )

    close_session = False
    if session is None:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        session = aiohttp.ClientSession(timeout=timeout_cfg)
        close_session = True
    try:
        params = {"query": query, "k": str(k)}
        if cw22_a:
            params["cw22_a"] = "true"
        headers = {"x-api-key": key}
        async with session.get(CLUEWEB_BASE_URL, params=params, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise WebSearchError(
                    f"ClueWeb request failed: {resp.status} {text[:200]}"
                )
            data = await resp.json(content_type=None)
        return await _decode_results(data)
    finally:
        if close_session:
            await session.close()


# ---------- Sync convenience wrappers ----------

def fineweb_search_sync(query: str, k: int = 5, timeout: float = 20.0) -> List[Dict[str, Any]]:
    """Synchronous wrapper for fineweb_search (creates its own loop if needed)."""
    try:
        loop = asyncio.get_running_loop()  # Will raise if no loop
    except RuntimeError:
        return asyncio.run(fineweb_search(query=query, k=k, timeout=timeout))
    else:
        # If already in an event loop, the caller should use the async version.
        raise RuntimeError(
            "fineweb_search_sync called from within an existing event loop; "
            "use await fineweb_search(...)."
        )


def clueweb_search_sync(
    query: str,
    k: int = 5,
    api_key: Optional[str] = None,
    cw22_a: bool = False,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Synchronous wrapper for clueweb_search (creates its own loop if needed)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            clueweb_search(query=query, k=k, api_key=api_key, cw22_a=cw22_a, timeout=timeout)
        )
    else:
        raise RuntimeError(
            "clueweb_search_sync called from within an existing event loop; "
            "use await clueweb_search(...)."
        )


