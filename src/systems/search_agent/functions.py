"""Tool implementations invoked by SearchAgent in response to model tool calls.

Each function takes the parsed arguments dict plus a citations registry it
may mutate (assigning stable [N] IDs to newly seen URLs) and returns a
plain-text string to feed back into the model via function_call_output.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from systems.rag_interface import CitationItem
from tools.brave_llm_context import COST_PER_CALL_USD, brave_llm_context
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url
from tools.posthog_client import capture_ai_span
from tools.web_search import SearchResult


logger = get_logger("search_agent.functions")

# Per-result snippet cap so a single search can't dominate the context.
_MAX_SNIPPET_CHARS = 4000


async def exec_web_search(
    args: Dict[str, Any],
    citations_by_url: Dict[str, CitationItem],
    default_count: int = 5,
    max_tokens: int = 4096,
    *,
    trace_id: Optional[str] = None,
    distinct_id: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> str:
    """Run a Brave LLM Context search and return a model-facing summary.

    Mutates citations_by_url: any new URL is assigned a stable sid (the
    next integer) and stored as a CitationItem. Existing URLs reuse
    their prior sid so the model's [N] references stay stable across
    turns.

    If trace_id + distinct_id are provided, emits a $ai_span event for
    PostHog so the Brave HTTP call shows up in the trace tree alongside
    the LLM turns.
    """
    query = (args.get("query") or "").strip()
    if not query:
        return "Error: empty query."

    count = int(args.get("count") or default_count)
    count = max(1, min(count, 10))
    freshness = args.get("freshness") or None  # null sentinel from strict mode

    started_at = time.time()
    is_error = False
    error_msg: Optional[str] = None
    results: List[SearchResult] = []
    try:
        results = await brave_llm_context(
            query=query,
            count=count,
            max_tokens=max_tokens,
            max_urls=count,
            freshness=freshness,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("web_search failed", error=str(e), query=query)
        is_error = True
        error_msg = str(e)

    if trace_id and distinct_id:
        capture_ai_span(
            distinct_id=distinct_id,
            trace_id=trace_id,
            span_name="brave_llm_context",
            started_at=started_at,
            parent_id=parent_id,
            is_error=is_error,
            error=error_msg,
            input_state={
                "query": query,
                "count": count,
                "freshness": freshness,
            },
            output_state={
                "num_results": len(results),
                "urls": [r.url for r in results],
            },
            # Every successful API call costs the same flat rate regardless
            # of num_results returned. Skip cost for errors (we don't pay
            # for failed requests).
            cost_usd=None if is_error else COST_PER_CALL_USD,
        )

    if is_error:
        return f"Error running web_search: {error_msg}"
    if not results:
        return f"No results for query: {query!r}."

    return _format_results(query, results, citations_by_url)


def _format_results(
    query: str,
    results: List[SearchResult],
    citations_by_url: Dict[str, CitationItem],
) -> str:
    lines: List[str] = [f"Search results for: {query!r}\n"]
    for r in results:
        existing = citations_by_url.get(r.url)
        if existing and existing.get("sid"):
            sid = existing["sid"] or ""
        else:
            sid = str(len(citations_by_url) + 1)
            title = (r.metadata or {}).get("title") if r.metadata else None
            citations_by_url[r.url] = CitationItem(
                url=r.url,
                icon_url=to_icon_url(r.url),
                date=str(r.date) if r.date else None,
                title=title,
                text=r.text,
                sid=sid,
                chunk_idx=None,
            )

        title = (r.metadata or {}).get("title", "") if r.metadata else ""
        date = f" ({r.date})" if r.date else ""
        text = r.text or ""
        snippet = text if len(text) < _MAX_SNIPPET_CHARS else text[:_MAX_SNIPPET_CHARS] + "…"
        lines.append(f"[{sid}] {title}{date}\n{r.url}\n{snippet}\n")

    return "\n".join(lines)
