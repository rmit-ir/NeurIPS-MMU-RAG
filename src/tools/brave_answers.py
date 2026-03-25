"""
Brave Answers API client.
Uses the /res/v1/chat/completions endpoint (OpenAI-compatible) to get
AI-generated answers grounded in real-time web search results.

See https://api-dashboard.search.brave.com/app/documentation/ai-grounding
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from openai import AsyncOpenAI

from tools.logging_utils import get_logger
from tools.retry_utils import retry

logger = get_logger('brave_answers')


@dataclass
class BraveAnswerCitation:
    """A citation from a Brave Answers response."""
    number: int
    url: str
    snippet: Optional[str] = None
    favicon: Optional[str] = None
    start_index: int = 0
    end_index: int = 0


@dataclass
class BraveAnswerResult:
    """Result from the Brave Answers API."""
    content: str
    citations: List[BraveAnswerCitation] = field(default_factory=list)
    queries_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    total_cost: float = 0.0


class BraveAnswersError(Exception):
    """Custom exception for Brave Answers API errors."""


@retry(max_retries=4, retry_on=(Exception,))
async def brave_answers(
    query: str,
    api_key: Optional[str] = None,
    enable_citations: bool = True,
    enable_research: bool = False,
) -> BraveAnswerResult:
    """Get an AI-generated answer from Brave Answers API.

    Args:
        query: The question to answer.
        api_key: Brave API key. If None, uses BRAVE_API_KEY env var.
        enable_citations: Include inline citations (requires streaming internally).
        enable_research: Enable multi-search research mode (slower, more thorough).

    Returns:
        BraveAnswerResult with content, citations, and usage metadata.
    """
    if not query:
        raise ValueError("query must be non-empty")

    key = api_key or os.getenv("BRAVE_API_KEY")
    if not key:
        raise ValueError(
            "Brave API key not provided. Set BRAVE_API_KEY or pass api_key."
        )

    client = AsyncOpenAI(
        api_key=key,
        base_url="https://api.search.brave.com/res/v1",
    )

    start_time = asyncio.get_event_loop().time()

    # Citations/research require streaming
    use_streaming = enable_citations or enable_research

    if use_streaming:
        return await _brave_answers_streaming(
            client, query, enable_citations, enable_research, start_time
        )
    else:
        return await _brave_answers_sync(client, query, start_time)


async def _brave_answers_sync(
    client: AsyncOpenAI,
    query: str,
    start_time: float,
) -> BraveAnswerResult:
    """Non-streaming Brave Answers request."""
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": query}],
        model="brave",
        stream=False,
    )

    content = response.choices[0].message.content or ""
    took = asyncio.get_event_loop().time() - start_time
    logger.info("Brave answers completed (sync)", query=query, duration=took)

    return BraveAnswerResult(content=content)


async def _brave_answers_streaming(
    client: AsyncOpenAI,
    query: str,
    enable_citations: bool,
    enable_research: bool,
    start_time: float,
) -> BraveAnswerResult:
    """Streaming Brave Answers request to capture citations and usage."""
    text_parts: list[str] = []
    citations: list[BraveAnswerCitation] = []
    usage_data: dict = {}

    async for chunk in await client.chat.completions.create(
        messages=[{"role": "user", "content": query}],
        model="brave",
        stream=True,
        extra_body={
            # Research mode doesn't support enable_citations
            **({"enable_citations": True} if enable_citations and not enable_research else {}),
            **({"enable_research": True} if enable_research else {}),
        },
    ):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if not delta:
            continue

        if delta.startswith("<citation>") and delta.endswith("</citation>"):
            data = json.loads(delta.removeprefix("<citation>").removesuffix("</citation>"))
            citations.append(BraveAnswerCitation(
                number=data.get("number", 0),
                url=data.get("url", ""),
                snippet=data.get("snippet"),
                favicon=data.get("favicon"),
                start_index=data.get("start_index", 0),
                end_index=data.get("end_index", 0),
            ))
        elif delta.startswith("<usage>") and delta.endswith("</usage>"):
            usage_data = json.loads(delta.removeprefix("<usage>").removesuffix("</usage>"))
        elif delta.startswith("<enum_item>") and delta.endswith("</enum_item>"):
            # Entity items - skip for now
            pass
        else:
            text_parts.append(delta)

    took = asyncio.get_event_loop().time() - start_time
    logger.info(
        "Brave answers completed (streaming)",
        query=query,
        num_citations=len(citations),
        duration=took,
    )

    return BraveAnswerResult(
        content="".join(text_parts),
        citations=citations,
        queries_used=usage_data.get("X-Request-Queries", 0),
        tokens_in=usage_data.get("X-Request-Tokens-In", 0),
        tokens_out=usage_data.get("X-Request-Tokens-Out", 0),
        total_cost=usage_data.get("X-Request-Total-Cost", 0.0),
    )


async def main():
    """Test function for Brave Answers."""
    result = await brave_answers("What are the latest advances in multimodal RAG?")
    print(f"Answer ({result.queries_used} queries, ${result.total_cost:.4f}):\n")
    print(result.content)
    if result.citations:
        print(f"\nCitations ({len(result.citations)}):")
        for c in result.citations:
            print(f"  [{c.number}] {c.url}")


if __name__ == "__main__":
    asyncio.run(main())
