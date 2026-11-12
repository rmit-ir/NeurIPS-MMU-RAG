"""Keenable page text in the existing RAG SearchResult format."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import os
from urllib.parse import urlsplit

from keenable import AsyncKeenable, KeenableRateLimitError

from tools.web_search import SearchResult


async def keenable_context(query: str, count: int = 20, max_tokens: int = 8192,
                           max_urls: int = 20, freshness: str | None = None) -> list[SearchResult]:
    """Return extracted pages, keeping the agent's URL and approximate token budgets."""
    if not query.strip():
        raise ValueError("query must be non-empty")
    if not os.getenv("KEENABLE_API_KEY"):
        raise ValueError("KEENABLE_API_KEY is required")
    count = max(1, min(count, max_urls, 50))
    filters = {}
    if freshness:
        days = {"pd": 1, "pw": 7, "pm": 30, "py": 365}[freshness]
        filters["published_after"] = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    async with AsyncKeenable(client_source="ASE AI Overview") as client:
        try:
            response = await client.search(query, mode="pro", max_results=count,
                                           snippet_max_length=min(10000, max(180, max_tokens * 4 // count)),
                                           **filters)
        except KeenableRateLimitError:
            await asyncio.sleep(1)
            response = await client.search(query, mode="pro", max_results=count,
                                           snippet_max_length=min(10000, max(180, max_tokens * 4 // count)),
                                           **filters)
    results = []
    for result in response:
        url = urlsplit(result.url)
        # shortcut: omit expiring cards until the SDK exposes disable_cards.
        if not result.snippet or url.scheme not in ("https", "http") or not url.hostname or url.hostname == "cards.keenable.ai":
            continue
        sid = str(len(results) + 1)
        results.append(SearchResult(
            type="keenable", text=result.snippet, id=f"keenable_{sid}", sid=sid,
            url=result.url, token_count=len(result.snippet.split()),
            metadata={"title": result.title, "hostname": url.hostname, "provider": "keenable"},
            dump=None, date=result.published_at, file_path=None,
            language=None, language_score=None, score=None,
        ))
    return results
