#!/usr/bin/env python
"""Test script for Brave Answers API (AI Overview via /res/v1/chat/completions)."""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


async def test_streaming(query: str = "What are the latest advances in multimodal RAG?"):
    """Test streaming mode with citations and entities."""
    api_key = os.environ["BRAVE_API_KEY"]

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.search.brave.com/res/v1",
    )

    print(f"=== Streaming: {query} ===\n")

    citations = []
    async for chunk in await client.chat.completions.create(
        messages=[{"role": "user", "content": query}],
        model="brave",
        stream=True,
        extra_body={
            "enable_citations": True,
            "enable_entities": True,
            "enable_research": False,
        },
    ):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if not delta:
            continue

        if delta.startswith("<citation>") and delta.endswith("</citation>"):
            citation = json.loads(delta.removeprefix("<citation>").removesuffix("</citation>"))
            citations.append(citation)
            print(f"[{citation['number']}]", end="", flush=True)

        elif delta.startswith("<enum_item>") and delta.endswith("</enum_item>"):
            item = json.loads(delta.removeprefix("<enum_item>").removesuffix("</enum_item>"))
            print(f"\n  * {item['original_tokens']}", end="", flush=True)

        elif delta.startswith("<usage>") and delta.endswith("</usage>"):
            usage = json.loads(delta.removeprefix("<usage>").removesuffix("</usage>"))
            print(f"\n\n--- Usage ---")
            print(f"  Queries: {usage.get('X-Request-Queries', '?')}")
            print(f"  Tokens in: {usage.get('X-Request-Tokens-In', '?')}")
            print(f"  Tokens out: {usage.get('X-Request-Tokens-Out', '?')}")
            print(f"  Total cost: ${usage.get('X-Request-Total-Cost', '?')}")

        else:
            print(delta, end="", flush=True)

    if citations:
        print(f"\n\n--- Citations ({len(citations)}) ---")
        for c in citations:
            print(f"  [{c['number']}] {c['url']}")

    print()


async def test_non_streaming(query: str = "What is the capital of Australia?"):
    """Test non-streaming mode (no citations/entities available)."""
    api_key = os.environ["BRAVE_API_KEY"]

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.search.brave.com/res/v1",
    )

    print(f"=== Non-streaming: {query} ===\n")

    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": query}],
        model="brave",
        stream=False,
    )

    print(response.choices[0].message.content)
    print()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Brave Answers API")
    parser.add_argument("query", nargs="?", default="What is the capital of Australia?")
    parser.add_argument("--no-stream", action="store_true", help="Use non-streaming mode")
    args = parser.parse_args()

    if args.no_stream:
        await test_non_streaming(args.query)
    else:
        await test_streaming(args.query)


if __name__ == "__main__":
    asyncio.run(main())
